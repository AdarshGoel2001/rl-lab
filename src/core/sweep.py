"""
Hyperparameter Sweep Orchestration

This module implements a comprehensive hyperparameter optimization system using Optuna.
It provides automatic hyperparameter tuning for RL experiments with pruning, resuming,
and advanced optimization algorithms.

Key features:
- Config-driven sweep definitions
- Multiple optimization algorithms (TPE, CmaEs, Random, etc.)
- Automatic pruning of poor trials
- Resume interrupted sweeps
- Integration with existing trainer and logging systems
- Multi-objective optimization support
"""

import os
import time
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import yaml
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner

from src.utils.config import Config, load_config, apply_config_overrides

logger = logging.getLogger(__name__)


@dataclass
class SweepConfig:
    """Configuration for hyperparameter sweep"""
    name: str
    study_name: str
    storage: str
    direction: str
    base_config: str
    parameters: Dict[str, Dict[str, Any]]
    sampler: Dict[str, Any]
    pruner: Dict[str, Any]
    overrides: Dict[str, Any]
    execution: Dict[str, Any]
    objective: Dict[str, Any]
    analysis: Dict[str, Any]
    advanced: Dict[str, Any]


class ParameterSampler:
    """Helper class for sampling hyperparameters from Optuna trial"""
    
    @staticmethod
    def suggest_parameter(trial: optuna.Trial, name: str, config: Dict[str, Any]) -> Any:
        """
        Suggest a parameter value based on its configuration.
        
        Args:
            trial: Optuna trial object
            name: Parameter name
            config: Parameter configuration dict
            
        Returns:
            Suggested parameter value
        """
        param_type = config.get('type', 'uniform')
        
        if param_type == 'uniform':
            return trial.suggest_float(name, config['low'], config['high'])
            
        elif param_type == 'log_uniform':
            return trial.suggest_float(name, config['low'], config['high'], log=True)
            
        elif param_type == 'int':
            return trial.suggest_int(name, config['low'], config['high'])
            
        elif param_type == 'categorical':
            return trial.suggest_categorical(name, config['choices'])
            
        elif param_type == 'discrete':
            return trial.suggest_discrete_uniform(
                name, config['low'], config['high'], config.get('q', 1.0)
            )
            
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")


class SweepOrchestrator:
    """
    Main orchestrator for hyperparameter sweeps using Optuna.
    
    Manages the complete sweep lifecycle from configuration loading
    through optimization and result analysis.
    """
    
    def __init__(self, sweep_config_path: str):
        """
        Initialize sweep orchestrator.
        
        Args:
            sweep_config_path: Path to sweep configuration file
        """
        self.config_path = Path(sweep_config_path)
        self.sweep_config = self._load_sweep_config()
        self.study: Optional[optuna.Study] = None
        self.best_trial: Optional[optuna.Trial] = None
        
        # Setup experiment directory
        self.experiment_dir = Path(self.sweep_config.advanced.get(
            'experiment_root', 'experiments/sweeps'
        )) / self.sweep_config.name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Sweep orchestrator initialized: {self.sweep_config.name}")
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _load_sweep_config(self) -> SweepConfig:
        """Load and validate sweep configuration"""
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract main sections with defaults
        sweep_data = config_dict.get('sweep', {})
        
        return SweepConfig(
            name=config_dict.get('sweep', {}).get('name', 'unnamed_sweep'),
            study_name=sweep_data.get('study_name', f"study_{int(time.time())}"),
            storage=sweep_data.get('storage', 'sqlite:///sweep_studies.db'),
            direction=sweep_data.get('direction', 'maximize'),
            base_config=config_dict.get('base_config', ''),
            parameters=config_dict.get('parameters', {}),
            sampler=config_dict.get('sweep', {}).get('sampler', {'type': 'TPE'}),
            pruner=config_dict.get('sweep', {}).get('pruner', {'type': 'MedianPruner'}),
            overrides=config_dict.get('overrides', {}),
            execution=config_dict.get('execution', {}),
            objective=config_dict.get('objective', {}),
            analysis=config_dict.get('analysis', {}),
            advanced=config_dict.get('advanced', {})
        )
    
    def _setup_logging(self):
        """Setup sweep-specific logging"""
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create file handler for sweep logs
        log_file = log_dir / "sweep.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        sweep_logger = logging.getLogger('sweep')
        sweep_logger.addHandler(file_handler)
        sweep_logger.setLevel(logging.INFO)
        
        logger.info(f"Sweep logging setup complete. Log file: {log_file}")
    
    def create_study(self) -> optuna.Study:
        """
        Create or load an Optuna study.
        
        Returns:
            Configured Optuna study
        """
        # Create sampler
        sampler = self._create_sampler()
        
        # Create pruner  
        pruner = self._create_pruner()
        
        # Create or load study
        study = optuna.create_study(
            study_name=self.sweep_config.study_name,
            storage=self.sweep_config.storage,
            direction=self.sweep_config.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=self.sweep_config.advanced.get('load_if_exists', True)
        )
        
        logger.info(f"Study created/loaded: {self.sweep_config.study_name}")
        logger.info(f"Storage: {self.sweep_config.storage}")
        logger.info(f"Direction: {self.sweep_config.direction}")
        logger.info(f"Sampler: {type(sampler).__name__}")
        logger.info(f"Pruner: {type(pruner).__name__}")
        
        return study
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on configuration"""
        sampler_config = self.sweep_config.sampler
        sampler_type = sampler_config.get('type', 'TPE')
        
        if sampler_type == 'TPE':
            return TPESampler(
                n_startup_trials=sampler_config.get('n_startup_trials', 10),
                n_ei_candidates=sampler_config.get('n_ei_candidates', 24),
                seed=42  # For reproducibility
            )
        elif sampler_type == 'Random':
            return RandomSampler(seed=42)
        elif sampler_type == 'CmaEs':
            return CmaEsSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
    
    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Create Optuna pruner based on configuration"""
        pruner_config = self.sweep_config.pruner
        pruner_type = pruner_config.get('type', 'MedianPruner')
        
        if pruner_type == 'MedianPruner':
            return MedianPruner(
                n_startup_trials=pruner_config.get('n_startup_trials', 5),
                n_warmup_steps=pruner_config.get('n_warmup_steps', 5000),
                interval_steps=pruner_config.get('interval_steps', 2500)
            )
        elif pruner_type == 'SuccessiveHalvingPruner':
            return SuccessiveHalvingPruner()
        elif pruner_type == 'HyperbandPruner':
            return HyperbandPruner()
        elif pruner_type is None or pruner_type == 'None':
            return optuna.pruners.NopPruner()
        else:
            raise ValueError(f"Unknown pruner type: {pruner_type}")
    
    def objective_function(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function that trains a model and returns the metric to optimize.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value to optimize
        """
        # Sample hyperparameters
        sampled_params = {}
        for param_name, param_config in self.sweep_config.parameters.items():
            value = ParameterSampler.suggest_parameter(trial, param_name, param_config)
            sampled_params[param_name] = value
        
        logger.info(f"Trial {trial.number}: Sampled parameters = {sampled_params}")
        
        try:
            # Load base configuration
            base_config = load_config(self.sweep_config.base_config)
            
            # Apply sampled parameters and overrides
            all_overrides = {**sampled_params, **self.sweep_config.overrides}
            modified_config = apply_config_overrides(base_config, all_overrides)
            
            # Create unique experiment directory for this trial
            trial_name = self.sweep_config.advanced.get(
                'trial_name_template', 'trial_{trial_number:03d}'
            ).format(trial_number=trial.number)
            
            trial_dir = self.experiment_dir / trial_name
            
            # Create trainer directly with modified config
            from src.core.trainer import Trainer
            trainer = Trainer(modified_config, trial_dir)
            
            # Run training with pruning support
            results = self._run_training_with_pruning(trainer, trial)
            
            # Extract objective value
            objective_metric = self.sweep_config.objective.get('metric', 'eval_return_mean')
            objective_value = results.get(objective_metric, -float('inf'))
            
            # Apply constraints
            if not self._check_constraints(results):
                logger.info(f"Trial {trial.number}: Failed constraints, pruning")
                raise optuna.TrialPruned()
            
            # Cleanup if not keeping all trials
            if not self.sweep_config.advanced.get('keep_all_trials', True):
                self._cleanup_trial_if_not_best(trial_dir, objective_value)
            
            logger.info(f"Trial {trial.number}: Objective value = {objective_value}")
            return objective_value
            
        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number}: Pruned")
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number}: Failed with error = {e}")
            # Return worst possible value for failed trials
            if self.sweep_config.direction == 'maximize':
                return -float('inf')
            else:
                return float('inf')
    
    def _run_training_with_pruning(self, trainer, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Run training with Optuna pruning support.
        
        Args:
            trainer: RL trainer instance
            trial: Optuna trial for pruning
            
        Returns:
            Final training results
        """
        # Store trial reference for pruning callback
        trainer._optuna_trial = trial
        trainer._optuna_pruning_metric = self.sweep_config.objective.get('metric', 'eval_return_mean')
        
        # Add pruning callback to trainer
        original_evaluation_step = trainer._evaluation_step
        
        def evaluation_step_with_pruning():
            eval_results = original_evaluation_step()
            
            # Report intermediate value for pruning
            metric_value = eval_results.get(trainer._optuna_pruning_metric)
            if metric_value is not None:
                trial.report(metric_value, trainer.step)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return eval_results
        
        trainer._evaluation_step = evaluation_step_with_pruning
        
        # Run training
        results = trainer.train()
        
        return results
    
    def _check_constraints(self, results: Dict[str, Any]) -> bool:
        """
        Check if trial results satisfy constraints.
        
        Args:
            results: Training results dictionary
            
        Returns:
            True if all constraints are satisfied
        """
        constraints = self.sweep_config.objective.get('constraints', [])
        
        for constraint in constraints:
            metric = constraint['metric']
            condition = constraint['condition']
            value = constraint['value']
            
            result_value = results.get(metric)
            if result_value is None:
                return False
            
            if condition == '>' and result_value <= value:
                return False
            elif condition == '<' and result_value >= value:
                return False
            elif condition == '==' and result_value != value:
                return False
            elif condition == '>=' and result_value < value:
                return False
            elif condition == '<=' and result_value > value:
                return False
        
        return True
    
    def _cleanup_trial_if_not_best(self, trial_dir: Path, objective_value: float):
        """Clean up trial directory if not among best trials"""
        # Simple cleanup logic - only keep if better than current best
        if self.best_trial is None or self._is_better_value(objective_value, self.best_trial.value):
            # Keep this trial, remove old best if exists
            if self.best_trial is not None:
                old_best_dir = self.experiment_dir / f"trial_{self.best_trial.number:03d}"
                if old_best_dir.exists():
                    import shutil
                    shutil.rmtree(old_best_dir, ignore_errors=True)
        else:
            # Remove this trial
            if trial_dir.exists():
                import shutil
                shutil.rmtree(trial_dir, ignore_errors=True)
    
    def _is_better_value(self, value1: float, value2: float) -> bool:
        """Check if value1 is better than value2 based on optimization direction"""
        if self.sweep_config.direction == 'maximize':
            return value1 > value2
        else:
            return value1 < value2
    
    def run_sweep(self) -> optuna.Study:
        """
        Run the complete hyperparameter sweep.
        
        Returns:
            Completed Optuna study
        """
        logger.info(f"Starting sweep: {self.sweep_config.name}")
        
        # Create study
        self.study = self.create_study()
        
        # Save sweep configuration
        self._save_sweep_config()
        
        # Run optimization
        execution_config = self.sweep_config.execution
        
        try:
            self.study.optimize(
                self.objective_function,
                n_trials=execution_config.get('n_trials', 100),
                timeout=execution_config.get('timeout', None),
                n_jobs=execution_config.get('n_jobs', 1)
            )
        except KeyboardInterrupt:
            logger.info("Sweep interrupted by user")
        
        # Get best trial
        self.best_trial = self.study.best_trial
        
        logger.info(f"Sweep completed. Best trial: {self.best_trial.number}")
        logger.info(f"Best value: {self.best_trial.value}")
        logger.info(f"Best parameters: {self.best_trial.params}")
        
        # Generate analysis
        self._generate_analysis()
        
        return self.study
    
    def _save_sweep_config(self):
        """Save sweep configuration for reproducibility"""
        config_dir = self.experiment_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        
        # Save original config
        import shutil
        shutil.copy2(self.config_path, config_dir / "sweep_config.yaml")
        
        # Save metadata
        metadata = {
            'sweep_name': self.sweep_config.name,
            'study_name': self.sweep_config.study_name,
            'created_at': datetime.now().isoformat(),
            'config_path': str(self.config_path),
            'experiment_dir': str(self.experiment_dir)
        }
        
        import json
        with open(config_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _generate_analysis(self):
        """Generate sweep analysis plots and reports"""
        if self.study is None:
            return
        
        analysis_dir = self.experiment_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        try:
            # Import optuna visualization (optional dependency)
            import optuna.visualization as vis
            import plotly
            
            # Generate plots specified in config
            plots = self.sweep_config.analysis.get('plots', [])
            
            for plot_type in plots:
                try:
                    if plot_type == "optimization_history":
                        fig = vis.plot_optimization_history(self.study)
                        fig.write_html(analysis_dir / "optimization_history.html")
                        
                    elif plot_type == "param_importances":
                        fig = vis.plot_param_importances(self.study)
                        fig.write_html(analysis_dir / "param_importances.html")
                        
                    elif plot_type == "parallel_coordinate":
                        fig = vis.plot_parallel_coordinate(self.study)
                        fig.write_html(analysis_dir / "parallel_coordinate.html")
                        
                    elif plot_type == "slice_plot":
                        fig = vis.plot_slice(self.study)
                        fig.write_html(analysis_dir / "slice_plot.html")
                        
                except Exception as e:
                    logger.warning(f"Failed to generate {plot_type}: {e}")
            
            logger.info(f"Analysis plots saved to {analysis_dir}")
            
        except ImportError:
            logger.warning("Optuna visualization not available. Skipping plots.")
        
        # Export best trials
        self._export_best_trials()
    
    def _export_best_trials(self):
        """Export best trial results"""
        if self.study is None:
            return
        
        analysis_dir = self.experiment_dir / "analysis"
        
        # Get best trials
        n_best = self.sweep_config.analysis.get('export_best_n', 10)
        best_trials = sorted(
            self.study.trials, 
            key=lambda t: t.value if t.value is not None else -float('inf'),
            reverse=(self.sweep_config.direction == 'maximize')
        )[:n_best]
        
        # Export to JSON
        best_trials_data = []
        for trial in best_trials:
            trial_data = {
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
            }
            best_trials_data.append(trial_data)
        
        import json
        with open(analysis_dir / "best_trials.json", 'w') as f:
            json.dump(best_trials_data, f, indent=2)
        
        # Export best parameters as config override
        if best_trials:
            best_params = best_trials[0].params
            with open(analysis_dir / "best_params_override.yaml", 'w') as f:
                yaml.dump(best_params, f, default_flow_style=False)
        
        logger.info(f"Best trials exported to {analysis_dir}")


def create_sweep_orchestrator(config_path: str) -> SweepOrchestrator:
    """
    Create sweep orchestrator from configuration file.
    
    Args:
        config_path: Path to sweep configuration file
        
    Returns:
        Initialized sweep orchestrator
    """
    return SweepOrchestrator(config_path)


def resume_sweep(study_name: str, storage: str) -> optuna.Study:
    """
    Resume an existing sweep study.
    
    Args:
        study_name: Name of study to resume
        storage: Storage URL
        
    Returns:
        Resumed study
    """
    study = optuna.load_study(
        study_name=study_name,
        storage=storage
    )
    
    logger.info(f"Resumed study: {study_name}")
    logger.info(f"Completed trials: {len(study.trials)}")
    
    return study


def list_studies(storage: str) -> List[str]:
    """
    List all studies in storage.
    
    Args:
        storage: Storage URL
        
    Returns:
        List of study names
    """
    if storage.startswith('sqlite'):
        # Extract database path
        db_path = storage.replace('sqlite:///', '')
        
        if not os.path.exists(db_path):
            return []
        
        # Query database for studies
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT study_name FROM studies")
            studies = [row[0] for row in cursor.fetchall()]
            return studies
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()
    
    else:
        # For other storage types, would need different implementation
        logger.warning(f"Listing studies not implemented for storage: {storage}")
        return []