#!/usr/bin/env python3
"""
Sweep Analysis and Visualization Script

Provides comprehensive analysis tools for hyperparameter sweep results.
Can analyze existing Optuna studies and generate detailed reports and visualizations.

Usage examples:
  # Analyze study and generate all plots
  python scripts/analyze_sweep.py --study ppo_cartpole_sweep_v1 --storage sqlite:///experiments/sweeps/ppo_cartpole.db
  
  # Generate specific plots only
  python scripts/analyze_sweep.py --study ppo_cartpole_sweep_v1 --storage sqlite:///experiments/sweeps/ppo_cartpole.db --plots optimization_history param_importances
  
  # Export best parameters
  python scripts/analyze_sweep.py --study ppo_cartpole_sweep_v1 --storage sqlite:///experiments/sweeps/ppo_cartpole.db --export-best 10
  
  # Compare multiple studies
  python scripts/analyze_sweep.py --compare study1 study2 --storage sqlite:///experiments/sweeps/comparison.db
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
import pandas as pd
import numpy as np


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Reduce noise from libraries
    logging.getLogger('optuna').setLevel(logging.WARNING)


def load_study(study_name: str, storage: str) -> optuna.Study:
    """
    Load Optuna study from storage.
    
    Args:
        study_name: Name of the study to load
        storage: Storage URL
        
    Returns:
        Loaded Optuna study
    """
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        logging.info(f"Loaded study: {study_name}")
        logging.info(f"Total trials: {len(study.trials)}")
        logging.info(f"Best value: {study.best_value if study.best_trial else 'None'}")
        return study
    except Exception as e:
        logging.error(f"Failed to load study '{study_name}': {e}")
        raise


def generate_study_summary(study: optuna.Study) -> Dict[str, Any]:
    """
    Generate comprehensive summary of study results.
    
    Args:
        study: Optuna study to analyze
        
    Returns:
        Dictionary containing study summary
    """
    trials = study.trials
    completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
    
    summary = {
        'study_name': study.study_name,
        'direction': study.direction.name,
        'total_trials': len(trials),
        'completed_trials': len(completed_trials),
        'pruned_trials': len(pruned_trials),
        'failed_trials': len(failed_trials),
        'pruning_rate': len(pruned_trials) / len(trials) if trials else 0,
    }
    
    if study.best_trial:
        summary.update({
            'best_trial_number': study.best_trial.number,
            'best_value': study.best_trial.value,
            'best_params': study.best_trial.params,
        })
        
        # Duration information
        if study.best_trial.datetime_complete and study.best_trial.datetime_start:
            duration = study.best_trial.datetime_complete - study.best_trial.datetime_start
            summary['best_trial_duration'] = duration.total_seconds()
    
    # Performance statistics for completed trials
    if completed_trials:
        values = [t.value for t in completed_trials]
        summary.update({
            'mean_performance': np.mean(values),
            'std_performance': np.std(values),
            'min_performance': np.min(values),
            'max_performance': np.max(values),
            'median_performance': np.median(values),
        })
    
    return summary


def generate_plots(study: optuna.Study, output_dir: Path, plot_types: List[str]):
    """
    Generate visualization plots for the study.
    
    Args:
        study: Optuna study to visualize
        output_dir: Directory to save plots
        plot_types: List of plot types to generate
    """
    try:
        import optuna.visualization as vis
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for plot_type in plot_types:
            try:
                if plot_type == "optimization_history":
                    fig = vis.plot_optimization_history(study)
                    fig.write_html(output_dir / "optimization_history.html")
                    logging.info("Generated optimization history plot")
                    
                elif plot_type == "param_importances":
                    fig = vis.plot_param_importances(study)
                    fig.write_html(output_dir / "param_importances.html")
                    logging.info("Generated parameter importances plot")
                    
                elif plot_type == "parallel_coordinate":
                    fig = vis.plot_parallel_coordinate(study)
                    fig.write_html(output_dir / "parallel_coordinate.html")
                    logging.info("Generated parallel coordinate plot")
                    
                elif plot_type == "slice_plot":
                    fig = vis.plot_slice(study)
                    fig.write_html(output_dir / "slice_plot.html")
                    logging.info("Generated slice plot")
                    
                elif plot_type == "contour_plot":
                    # Generate contour plot for top 2 most important parameters
                    if len(study.best_params) >= 2:
                        param_names = list(study.best_params.keys())[:2]
                        fig = vis.plot_contour(study, params=param_names)
                        fig.write_html(output_dir / "contour_plot.html")
                        logging.info("Generated contour plot")
                    
                elif plot_type == "edf":
                    fig = vis.plot_edf(study)
                    fig.write_html(output_dir / "edf_plot.html")
                    logging.info("Generated EDF plot")
                    
                else:
                    logging.warning(f"Unknown plot type: {plot_type}")
                    
            except Exception as e:
                logging.error(f"Failed to generate {plot_type}: {e}")
                
    except ImportError:
        logging.error("Optuna visualization not available. Install with: pip install optuna[visualization]")


def export_best_trials(study: optuna.Study, output_dir: Path, n_best: int):
    """
    Export best trial results to various formats.
    
    Args:
        study: Optuna study
        output_dir: Output directory
        n_best: Number of best trials to export
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get best trials
    direction_multiplier = 1 if study.direction == optuna.study.StudyDirection.MAXIMIZE else -1
    best_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: direction_multiplier * t.value,
        reverse=True
    )[:n_best]
    
    # Export to JSON
    best_trials_data = []
    for i, trial in enumerate(best_trials):
        trial_data = {
            'rank': i + 1,
            'trial_number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        }
        
        if trial.datetime_complete and trial.datetime_start:
            duration = trial.datetime_complete - trial.datetime_start
            trial_data['duration_seconds'] = duration.total_seconds()
        
        best_trials_data.append(trial_data)
    
    # Save as JSON
    json_path = output_dir / f"best_{n_best}_trials.json"
    with open(json_path, 'w') as f:
        json.dump(best_trials_data, f, indent=2)
    logging.info(f"Exported best trials to {json_path}")
    
    # Save as CSV
    try:
        df = pd.DataFrame(best_trials_data)
        
        # Flatten parameters into separate columns
        param_df = pd.json_normalize(df['params'])
        param_df.columns = [f"param_{col}" for col in param_df.columns]
        
        # Combine with main data
        df_expanded = pd.concat([df.drop('params', axis=1), param_df], axis=1)
        
        csv_path = output_dir / f"best_{n_best}_trials.csv"
        df_expanded.to_csv(csv_path, index=False)
        logging.info(f"Exported best trials to {csv_path}")
        
    except Exception as e:
        logging.warning(f"Failed to export CSV: {e}")
    
    # Export best parameters as YAML config override
    if best_trials:
        import yaml
        best_params = best_trials[0].params
        
        yaml_path = output_dir / "best_params_override.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(best_params, f, default_flow_style=False)
        logging.info(f"Exported best parameters to {yaml_path}")


def generate_parameter_analysis(study: optuna.Study, output_dir: Path):
    """
    Generate detailed parameter analysis.
    
    Args:
        study: Optuna study
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        logging.warning("No completed trials found for parameter analysis")
        return
    
    # Extract parameter values and performances
    param_data = []
    for trial in completed_trials:
        row = {'trial_number': trial.number, 'value': trial.value}
        row.update(trial.params)
        param_data.append(row)
    
    df = pd.DataFrame(param_data)
    
    # Parameter statistics
    param_stats = {}
    for param in study.best_params.keys():
        if param in df.columns:
            param_values = df[param].dropna()
            
            if param_values.dtype in ['int64', 'float64']:
                # Numerical parameter
                param_stats[param] = {
                    'type': 'numerical',
                    'mean': float(param_values.mean()),
                    'std': float(param_values.std()),
                    'min': float(param_values.min()),
                    'max': float(param_values.max()),
                    'median': float(param_values.median()),
                    'best_value': study.best_params[param]
                }
            else:
                # Categorical parameter
                value_counts = param_values.value_counts()
                param_stats[param] = {
                    'type': 'categorical',
                    'unique_values': list(value_counts.index),
                    'frequencies': value_counts.to_dict(),
                    'best_value': study.best_params[param]
                }
    
    # Save parameter statistics
    stats_path = output_dir / "parameter_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(param_stats, f, indent=2)
    logging.info(f"Exported parameter statistics to {stats_path}")
    
    # Generate correlation matrix for numerical parameters
    numerical_params = [p for p, stats in param_stats.items() if stats['type'] == 'numerical']
    if len(numerical_params) > 1:
        corr_matrix = df[numerical_params + ['value']].corr()
        
        corr_path = output_dir / "parameter_correlations.csv"
        corr_matrix.to_csv(corr_path)
        logging.info(f"Exported parameter correlations to {corr_path}")


def compare_studies(study_names: List[str], storage: str, output_dir: Path):
    """
    Compare multiple studies and generate comparison report.
    
    Args:
        study_names: List of study names to compare
        storage: Storage URL
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all studies
    studies = []
    for name in study_names:
        try:
            study = load_study(name, storage)
            studies.append(study)
        except Exception as e:
            logging.error(f"Failed to load study {name}: {e}")
            continue
    
    if not studies:
        logging.error("No studies could be loaded for comparison")
        return
    
    # Generate comparison data
    comparison_data = []
    for study in studies:
        summary = generate_study_summary(study)
        comparison_data.append(summary)
    
    # Create comparison DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    csv_path = output_dir / "study_comparison.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Study comparison exported to {csv_path}")
    
    # Save detailed comparison
    json_path = output_dir / "study_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    logging.info(f"Detailed comparison exported to {json_path}")


def main():
    """Main entry point for sweep analysis script"""
    parser = argparse.ArgumentParser(
        description="Analyze hyperparameter sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis of a study
  python scripts/analyze_sweep.py --study ppo_cartpole_sweep_v1 --storage sqlite:///experiments/sweeps/ppo_cartpole.db
  
  # Generate specific plots only
  python scripts/analyze_sweep.py --study ppo_cartpole_sweep_v1 --storage sqlite:///experiments/sweeps/ppo_cartpole.db --plots optimization_history param_importances
  
  # Export best trials
  python scripts/analyze_sweep.py --study ppo_cartpole_sweep_v1 --storage sqlite:///experiments/sweeps/ppo_cartpole.db --export-best 10
  
  # Compare multiple studies
  python scripts/analyze_sweep.py --compare study1 study2 --storage sqlite:///experiments/sweeps/comparison.db
        """
    )
    
    # Study specification
    study_group = parser.add_mutually_exclusive_group(required=True)
    study_group.add_argument(
        '--study',
        type=str,
        help="Study name to analyze"
    )
    study_group.add_argument(
        '--compare',
        nargs='+',
        help="Compare multiple studies"
    )
    
    # Storage configuration
    parser.add_argument(
        '--storage',
        type=str,
        required=True,
        help="Optuna storage URL"
    )
    
    # Output configuration
    parser.add_argument(
        '--output',
        type=str,
        default="experiments/analysis",
        help="Output directory for analysis results"
    )
    
    # Analysis options
    parser.add_argument(
        '--plots',
        nargs='*',
        default=['optimization_history', 'param_importances', 'parallel_coordinate', 'slice_plot'],
        choices=['optimization_history', 'param_importances', 'parallel_coordinate', 
                'slice_plot', 'contour_plot', 'edf'],
        help="Types of plots to generate"
    )
    
    parser.add_argument(
        '--export-best',
        type=int,
        default=10,
        help="Number of best trials to export"
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help="Skip plot generation"
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose logging"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Create output directory
    output_dir = Path(args.output)
    
    try:
        if args.study:
            # Single study analysis
            study = load_study(args.study, args.storage)
            
            # Create study-specific output directory
            study_output_dir = output_dir / args.study / datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate summary
            summary = generate_study_summary(study)
            
            print(f"\n{'='*60}")
            print(f"STUDY ANALYSIS: {study.study_name}")
            print(f"{'='*60}")
            print(f"Direction: {summary['direction']}")
            print(f"Total trials: {summary['total_trials']}")
            print(f"Completed: {summary['completed_trials']}")
            print(f"Pruned: {summary['pruned_trials']} ({summary['pruning_rate']:.1%})")
            print(f"Failed: {summary['failed_trials']}")
            
            if 'best_value' in summary:
                print(f"\nBest trial: #{summary['best_trial_number']}")
                print(f"Best value: {summary['best_value']:.4f}")
                print("Best parameters:")
                for key, value in summary['best_params'].items():
                    print(f"  {key}: {value}")
            
            if 'mean_performance' in summary:
                print(f"\nPerformance statistics:")
                print(f"  Mean: {summary['mean_performance']:.4f}")
                print(f"  Std:  {summary['std_performance']:.4f}")
                print(f"  Min:  {summary['min_performance']:.4f}")
                print(f"  Max:  {summary['max_performance']:.4f}")
            
            print(f"\nResults will be saved to: {study_output_dir}")
            print(f"{'='*60}")
            
            # Save summary
            summary_path = study_output_dir / "study_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Generate plots
            if not args.no_plots and args.plots:
                generate_plots(study, study_output_dir / "plots", args.plots)
            
            # Export best trials
            if args.export_best > 0:
                export_best_trials(study, study_output_dir / "exports", args.export_best)
            
            # Generate parameter analysis
            generate_parameter_analysis(study, study_output_dir / "analysis")
            
        elif args.compare:
            # Multi-study comparison
            comparison_output_dir = output_dir / "comparison" / datetime.now().strftime("%Y%m%d_%H%M%S")
            compare_studies(args.compare, args.storage, comparison_output_dir)
            
            print(f"\nStudy comparison results saved to: {comparison_output_dir}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)