# Test Suite Report
PyTest Summary: 0 tests, 0 failures, 0 errors, 0 skipped, time 0.00s

## Issue Summary
- By Category: call_graph: 48, code_quality: 36, dependency: 6, performance: 1, benchmark: 1, regression: 1, resources: 1
- By Severity: info: 94

## Recent Issues (up to 50)
- [info] call_graph @ ?: Function defined but not referenced: save_best_checkpoint 
- [info] call_graph @ ?: Function defined but not referenced: save_model 
- [info] call_graph @ ?: Function defined but not referenced: summary 
- [info] call_graph @ ?: Function defined but not referenced: unfreeze_parameters 
- [info] call_graph @ ?: Function defined but not referenced: update_config 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/buffers/trajectory.py: Unused import 'Experience' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/buffers/trajectory.py: Unused import 'Tuple' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/buffers/base.py: Unused import 'List' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/buffers/base.py: Unused import 'Tuple' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/buffers/base.py: Unused import 'deque' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/core/sweep.py: Unused import 'Callable' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/core/sweep.py: Unused import 'Config' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/core/sweep.py: Unused import 'Union' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/core/sweep.py: Unused import 'create_trainer_from_config' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/core/sweep.py: Unused import 'np' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/core/sweep.py: Unused import 'plotly' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/core/trainer.py: Unused import 'ConfigManager' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/core/trainer.py: Unused import 'Tuple' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/core/trainer.py: Unused import 'os' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/environments/__init__.py: Unused import 'BaseEnvironment' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/environments/__init__.py: Unused import 'ParallelEnvironmentManager' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/environments/minigrid_wrapper.py: Unused import 'Union' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/environments/minigrid_wrapper.py: Unused import 'minigrid' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/environments/minigrid_wrapper.py: Unused import 'torch' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/environments/gym_wrapper.py: Unused import 'Union' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/algorithms/random.py: Unused import 'np' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/algorithms/ppo.py: Unused import 'Optional' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/algorithms/ppo.py: Unused import 'nn' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/algorithms/base.py: Unused import 'Optional' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/utils/checkpoint.py: Unused import 'os' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/utils/logger.py: Unused import 'os' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/networks/cnn.py: Unused import 'Optional' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/networks/cnn.py: Unused import 'Tuple' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/networks/cnn.py: Unused import 'np' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/networks/mlp.py: Unused import 'List' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/networks/mlp.py: Unused import 'Optional' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/networks/mlp.py: Unused import 'Union' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/networks/base.py: Unused import 'Tuple' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/networks/base.py: Unused import 'Union' 
- [info] code_quality @ /Users/martian/Documents/Code/rl-lab/src/networks/base.py: Unused import 'np' 
- [info] dependency @ ?: Module not imported elsewhere: src.algorithms.ppo 
- [info] dependency @ ?: Module not imported elsewhere: src.algorithms.random 
- [info] dependency @ ?: Module not imported elsewhere: src.buffers.trajectory 
- [info] dependency @ ?: Module not imported elsewhere: src.core.sweep 
- [info] dependency @ ?: Module not imported elsewhere: src.environments.gym_wrapper 
- [info] dependency @ ?: Module not imported elsewhere: src.networks.mlp 
- [info] performance @ ?: Top allocation diff (bytes) 
- [info] benchmark @ ?: MLP forward benchmark 
- [info] regression @ ?: Forward pass baseline 
- [info] resources @ ?: Device availability 