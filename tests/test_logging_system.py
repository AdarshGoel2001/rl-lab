"""
Comprehensive Test Suite for Standardized Logging System

Tests logging functionality across single and parallel environments with
validation of metric standards, output formatting, and backend integration.
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
import json
import yaml

# Import our logging components
from src.utils.logger import ExperimentLogger, LoggingConfig, create_logger
from src.utils.logging_standards import (
    StandardizedMetrics, MetricValidator, LoggingOutputFormatter, 
    EnvironmentCompatibilityChecker, MetricNamingStandards
)


class TestStandardizedMetrics:
    """Test metric standards and specifications"""
    
    def test_metric_specifications(self):
        """Test that all required metrics are properly defined"""
        all_specs = StandardizedMetrics.get_all_specs()
        required_specs = StandardizedMetrics.get_required_specs()
        
        # Ensure we have core training metrics
        assert "step" in all_specs
        assert "episode" in all_specs
        assert "time_elapsed" in all_specs
        assert "steps_per_second" in all_specs
        
        # Ensure algorithm performance metrics
        assert "policy_loss" in all_specs
        assert "value_loss" in all_specs
        assert "total_loss" in all_specs
        
        # Ensure environment metrics
        assert "reward_mean" in all_specs
        assert "episode_length_mean" in all_specs
        
        # Check required vs optional
        assert all_specs["step"].required == True
        assert all_specs["debug/actor_mean_avg"].required == False
        
        print(f"✓ Found {len(all_specs)} total metrics, {len(required_specs)} required")


class TestMetricValidator:
    """Test metric validation against standards"""
    
    def setup_method(self):
        self.validator = MetricValidator()
    
    def test_valid_training_metrics(self):
        """Test validation of complete valid training metrics"""
        metrics = {
            "step": 1000,
            "episode": 50,
            "time_elapsed": 120.5,
            "steps_per_second": 8.3,
            "policy_loss": 0.05,
            "value_loss": 0.02,
            "entropy_loss": 0.01,
            "total_loss": 0.08,
            "grad_norm": 0.5,
            "reward_mean": 15.2,
            "reward_std": 3.1,
            "episode_length_mean": 200.0,
            "episode_length_std": 25.0,
        }
        
        result = self.validator.validate_metrics(metrics, "training")
        
        assert result["is_valid"] == True
        assert len(result["errors"]) == 0
        assert len(result["metrics"]) == len(metrics)
        print("✓ Valid training metrics passed validation")
    
    def test_invalid_metrics(self):
        """Test validation catches invalid metrics"""
        metrics = {
            "step": -1,  # Invalid: negative step
            "policy_loss": float('nan'),  # Invalid: NaN value
            "reward_mean": float('inf'),  # Invalid: Inf value
            "clip_ratio": 1.5,  # Invalid: above max value (should be 0-1)
        }
        
        result = self.validator.validate_metrics(metrics, "training")
        
        assert result["is_valid"] == False
        assert len(result["errors"]) > 0
        print(f"✓ Invalid metrics caught {len(result['errors'])} errors")
    
    def test_missing_required_metrics(self):
        """Test validation catches missing required metrics"""
        metrics = {
            "step": 1000,  # Only one required metric
        }
        
        result = self.validator.validate_metrics(metrics, "training")
        
        assert result["is_valid"] == False
        assert len(result["errors"]) > 0
        print(f"✓ Missing required metrics caught {len(result['errors'])} errors")


class TestEnvironmentCompatibilityChecker:
    """Test environment type detection and metric normalization"""
    
    def setup_method(self):
        self.checker = EnvironmentCompatibilityChecker()
    
    def test_single_environment_detection(self):
        """Test detection of single environment metrics"""
        metrics = {
            "reward": 10.5,
            "episode_length": 200,
            "policy_loss": 0.05
        }
        
        env_type = self.checker.detect_environment_type(metrics)
        assert env_type == "single"
        print("✓ Single environment correctly detected")
    
    def test_parallel_environment_detection(self):
        """Test detection of parallel environment metrics"""
        metrics = {
            "reward_batch": [10.5, 8.2, 12.1, 9.8],
            "num_envs": 4,
            "policy_loss": 0.05
        }
        
        env_type = self.checker.detect_environment_type(metrics)
        assert env_type == "parallel"
        print("✓ Parallel environment correctly detected")
    
    def test_single_environment_normalization(self):
        """Test normalization of single environment metrics"""
        metrics = {
            "reward": 10.5,
            "episode_length": 200,
            "policy_loss": 0.05,
            "tensor_metric": torch.tensor(1.5)
        }
        
        normalized = self.checker.normalize_metrics(metrics, "single")
        
        assert isinstance(normalized["reward"], float)
        assert isinstance(normalized["episode_length"], float)
        assert isinstance(normalized["policy_loss"], float)
        assert isinstance(normalized["tensor_metric"], float)
        print("✓ Single environment metrics normalized")
    
    def test_parallel_environment_normalization(self):
        """Test normalization of parallel environment metrics"""
        metrics = {
            "reward": np.array([10.5, 8.2, 12.1, 9.8]),
            "episode_length": np.array([200, 180, 220, 195]),
            "policy_loss": 0.05
        }
        
        normalized = self.checker.normalize_metrics(metrics, "parallel")
        
        # Should aggregate array metrics
        assert "reward_mean" in normalized
        assert "reward_std" in normalized
        assert "episode_length_mean" in normalized
        assert "episode_length_std" in normalized
        assert isinstance(normalized["reward_mean"], float)
        assert normalized["reward_mean"] == pytest.approx(10.15)
        print("✓ Parallel environment metrics normalized with aggregation")


class TestLoggingOutputFormatter:
    """Test output formatting for different backends"""
    
    def setup_method(self):
        self.formatter = LoggingOutputFormatter()
    
    def test_bash_formatting(self):
        """Test bash output formatting for Claude Code readability"""
        metrics = {
            "train/step": 1000,
            "train/episode": 50,
            "train/reward_mean": 15.234,
            "train/policy_loss": 0.0567,
            "train/steps_per_second": 8.3
        }
        
        bash_output = self.formatter.format_for_bash(metrics, 1000)
        
        assert "Step 1000" in bash_output
        assert "Reward: 15.2340" in bash_output
        assert "PiLoss: 0.0567" in bash_output
        assert "SPS: 8.3" in bash_output
        print(f"✓ Bash output: {bash_output}")
    
    def test_tensorboard_formatting(self):
        """Test TensorBoard formatting with standardized names"""
        metrics = {
            "reward_mean": 15.234,
            "policy_loss": 0.0567
        }
        
        tb_formatted = self.formatter.format_for_tensorboard(metrics, 1000, "train")
        
        assert "train/reward_mean" in tb_formatted
        assert "train/policy_loss" in tb_formatted
        print("✓ TensorBoard formatting applied")
    
    def test_wandb_formatting(self):
        """Test W&B formatting with dot notation"""
        metrics = {
            "train/reward_mean": 15.234,
            "train/policy_loss": 0.0567
        }
        
        wandb_formatted = self.formatter.format_for_wandb(metrics, 1000)
        
        assert "step" in wandb_formatted
        assert wandb_formatted["step"] == 1000
        assert "train.reward_mean" in wandb_formatted
        assert "train.policy_loss" in wandb_formatted
        print("✓ W&B formatting applied")


class TestExperimentLogger:
    """Test the main experiment logger with all backends"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.experiment_dir = Path(self.temp_dir) / "test_experiment"
        self.experiment_dir.mkdir(parents=True)
    
    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
    
    def test_logger_initialization(self):
        """Test logger initializes with standardized components"""
        config = LoggingConfig(
            terminal=True,
            tensorboard=False,
            wandb_enabled=False
        )
        
        logger = ExperimentLogger(self.experiment_dir, config)
        
        # Check standardized components are initialized
        assert hasattr(logger, 'metric_validator')
        assert hasattr(logger, 'output_formatter')
        assert hasattr(logger, 'env_checker')
        assert hasattr(logger, 'bash_log_file')
        assert logger.detected_env_type is None
        print("✓ Logger initialized with standardized components")
    
    @patch('src.utils.logger.SummaryWriter')
    def test_single_environment_logging(self, mock_summary_writer):
        """Test logging with single environment metrics"""
        config = LoggingConfig(
            terminal=True,
            tensorboard=True,
            wandb_enabled=False,
            log_frequency=1
        )
        
        logger = ExperimentLogger(self.experiment_dir, config)
        
        # Single environment metrics
        metrics = {
            "step": 1000,
            "episode": 50,
            "reward_mean": 15.2,
            "policy_loss": 0.05,
            "value_loss": 0.02,
            "total_loss": 0.07,
            "grad_norm": 0.5,
            "time_elapsed": 120.0,
            "steps_per_second": 8.3,
            "episode_length_mean": 200.0,
            "reward_std": 3.1,
            "episode_length_std": 25.0,
            "entropy_loss": 0.01
        }
        
        logger.log_metrics(metrics, 1000, prefix="train")
        
        # Check bash log file was created and written
        assert logger.bash_log_file.exists()
        with open(logger.bash_log_file, 'r') as f:
            content = f.read()
            assert "Step 1000" in content
            assert "Reward: 15.2000" in content
        
        # Check environment type was detected
        assert logger.detected_env_type == "single"
        print("✓ Single environment logging successful")
    
    @patch('src.utils.logger.SummaryWriter')
    def test_parallel_environment_logging(self, mock_summary_writer):
        """Test logging with parallel environment metrics"""
        config = LoggingConfig(
            terminal=True,
            tensorboard=True,
            wandb_enabled=False,
            log_frequency=1
        )
        
        logger = ExperimentLogger(self.experiment_dir, config)
        
        # Parallel environment metrics
        metrics = {
            "step": 1000,
            "episode": 50,
            "reward": np.array([10.5, 8.2, 12.1, 9.8]),  # Will be normalized
            "episode_length": np.array([200, 180, 220, 195]),
            "policy_loss": 0.05,
            "value_loss": 0.02,
            "total_loss": 0.07,
            "grad_norm": 0.5,
            "time_elapsed": 120.0,
            "steps_per_second": 33.2,  # Higher due to parallelization
            "entropy_loss": 0.01,
            "num_envs": 4
        }
        
        logger.log_metrics(metrics, 1000, prefix="train")
        
        # Check environment type was detected
        assert logger.detected_env_type == "parallel"
        
        # Check bash log file was created
        assert logger.bash_log_file.exists()
        with open(logger.bash_log_file, 'r') as f:
            content = f.read()
            assert "Step 1000" in content
        
        print("✓ Parallel environment logging successful")
    
    def test_metric_validation_warnings(self):
        """Test that metric validation warnings are logged"""
        config = LoggingConfig(terminal=True, tensorboard=False, wandb_enabled=False)
        logger = ExperimentLogger(self.experiment_dir, config)
        
        # Incomplete metrics (missing required ones)
        metrics = {
            "step": 1000,
            "reward_mean": 15.2,
            # Missing many required metrics
        }
        
        with patch('src.utils.logger.logger') as mock_logger:
            logger.log_metrics(metrics, 1000, prefix="train")
            
            # Should have logged warnings about missing metrics
            warning_calls = [call for call in mock_logger.warning.call_args_list 
                           if "Missing critical metrics" in str(call)]
            assert len(warning_calls) > 0
        
        print("✓ Metric validation warnings properly logged")


class TestIntegrationWithTrainer:
    """Test integration with the trainer system"""
    
    @pytest.mark.integration
    def test_end_to_end_single_environment(self):
        """Test complete logging flow with single environment"""
        # This would test with actual trainer but requires full setup
        # For now, simulate the key components
        
        temp_dir = tempfile.mkdtemp()
        try:
            experiment_dir = Path(temp_dir) / "integration_test"
            config = LoggingConfig(
                terminal=True,
                tensorboard=False,
                wandb_enabled=False,
                log_frequency=10
            )
            
            logger = ExperimentLogger(experiment_dir, config)
            
            # Simulate training loop with realistic PPO metrics
            for step in range(0, 100, 10):
                metrics = {
                    "step": step,
                    "episode": step // 10,
                    "reward_mean": 10.0 + np.random.normal(0, 2),
                    "episode_length_mean": 200.0 + np.random.normal(0, 20),
                    "policy_loss": 0.05 + np.random.normal(0, 0.01),
                    "value_loss": 0.02 + np.random.normal(0, 0.005),
                    "entropy_loss": 0.01 + np.random.normal(0, 0.002),
                    "total_loss": 0.08 + np.random.normal(0, 0.01),
                    "grad_norm": 0.5 + np.random.normal(0, 0.1),
                    "time_elapsed": step * 1.2,
                    "steps_per_second": 8.3,
                    "reward_std": 3.0,
                    "episode_length_std": 25.0,
                }
                
                logger.log_metrics(metrics, step, prefix="train")
            
            # Check bash log file exists and has content
            assert logger.bash_log_file.exists()
            with open(logger.bash_log_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 10  # Should have logged multiple times
            
            logger.finish()
            print("✓ End-to-end single environment integration test passed")
            
        finally:
            shutil.rmtree(temp_dir)
    
    @pytest.mark.integration 
    def test_end_to_end_parallel_environment(self):
        """Test complete logging flow with parallel environment"""
        temp_dir = tempfile.mkdtemp()
        try:
            experiment_dir = Path(temp_dir) / "parallel_integration_test"
            config = LoggingConfig(
                terminal=True,
                tensorboard=False,
                wandb_enabled=False,
                log_frequency=10
            )
            
            logger = ExperimentLogger(experiment_dir, config)
            num_envs = 8
            
            # Simulate parallel training loop
            for step in range(0, 100, 10):
                # Simulate vectorized rewards and episode lengths
                rewards = np.random.normal(10.0, 2.0, num_envs)
                episode_lengths = np.random.normal(200.0, 20.0, num_envs)
                
                metrics = {
                    "step": step,
                    "episode": step // 10 * num_envs,
                    "reward": rewards,  # Will be normalized to reward_mean, reward_std
                    "episode_length": episode_lengths,
                    "policy_loss": 0.05 + np.random.normal(0, 0.01),
                    "value_loss": 0.02 + np.random.normal(0, 0.005),
                    "entropy_loss": 0.01 + np.random.normal(0, 0.002),
                    "total_loss": 0.08 + np.random.normal(0, 0.01),
                    "grad_norm": 0.5 + np.random.normal(0, 0.1),
                    "time_elapsed": step * 1.2,
                    "steps_per_second": 33.2,  # Higher due to parallelization
                    "num_envs": num_envs
                }
                
                logger.log_metrics(metrics, step, prefix="train")
            
            # Environment should be detected as parallel
            assert logger.detected_env_type == "parallel"
            
            # Check bash log file
            assert logger.bash_log_file.exists()
            with open(logger.bash_log_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 10
            
            logger.finish()
            print("✓ End-to-end parallel environment integration test passed")
            
        finally:
            shutil.rmtree(temp_dir)


def run_logging_tests():
    """Run all logging tests"""
    print("=" * 60)
    print("COMPREHENSIVE LOGGING SYSTEM TEST SUITE")
    print("=" * 60)
    
    # Test suites
    test_classes = [
        TestStandardizedMetrics,
        TestMetricValidator, 
        TestEnvironmentCompatibilityChecker,
        TestLoggingOutputFormatter,
        TestExperimentLogger,
        TestIntegrationWithTrainer
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        test_instance = test_class()
        
        # Get test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                # Setup if exists
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test
                method = getattr(test_instance, method_name)
                method()
                passed_tests += 1
                
                # Teardown if exists
                if hasattr(test_instance, 'teardown_method'):
                    test_instance.teardown_method()
                    
            except Exception as e:
                print(f"✗ {method_name}: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")
    print("=" * 60)


if __name__ == "__main__":
    run_logging_tests()