#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/Users/martian/Documents/Code/rl-lab')

import numpy as np
import torch
from src.buffers.trajectory import TrajectoryBuffer
from src.utils.config import load_config

def test_buffer():
    print("Testing trajectory buffer...")
    
    # Load config
    config = load_config('configs/experiments/ppo_atari_pong.yaml')
    
    # Create buffer with dict config
    buffer_config = {
        'capacity': 1024,
        'batch_size': 1024,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'compute_returns': True,
        'normalize_advantages': True
    }
    buffer = TrajectoryBuffer(buffer_config)
    
    # Create fake vectorized trajectory data
    num_steps = 128  # More steps to get enough data
    num_envs = 8
    obs_shape = (84, 84, 4)
    
    trajectory = {
        'observations': np.random.rand(num_steps, num_envs, *obs_shape),
        'actions': np.random.randint(0, 6, size=(num_steps, num_envs)),
        'rewards': np.random.rand(num_steps, num_envs),
        'old_values': np.random.rand(num_steps, num_envs),
        'old_log_probs': np.random.rand(num_steps, num_envs),
        'dones': np.random.random((num_steps, num_envs)) < 0.1
    }
    
    print("Adding trajectory to buffer...")
    print("Trajectory keys:", list(trajectory.keys()))
    print("Actions shape:", trajectory['actions'].shape)
    
    # Add single trajectory should be enough
    buffer.add(trajectory=trajectory)
    
    print("Buffer size:", buffer._size)
    print("Buffer ready:", buffer.ready())
    
    # Debug: Check what's in the trajectory after adding
    if buffer.trajectories:
        traj = buffer.trajectories[0]
        print(f"First trajectory keys: {list(traj.keys())}")
        for key, val in traj.items():
            if hasattr(val, 'shape'):
                print(f"  {key}: shape {val.shape}")
            else:
                print(f"  {key}: {type(val)}")
    
    # Sample from buffer (force sampling for debug)
    print("\nSampling from buffer...")
    
    # Let's debug the sampling process by manually checking what all_data looks like
    from collections import defaultdict
    all_data = defaultdict(list)
    
    for trajectory in buffer.trajectories:
        for key, values in trajectory.items():
            # Skip bootstrap_value as it's trajectory-level metadata, not step data
            if key == 'bootstrap_value':
                continue
                
            if isinstance(values, np.ndarray):
                if values.ndim == 3:
                    # Vectorized observations: (T, B, obs_dim) -> flatten to list of individual obs
                    for t in range(values.shape[0]):
                        for b in range(values.shape[1]):
                            all_data[key].append(values[t, b])
                elif values.ndim == 2:
                    # Vectorized scalar values: (T, B) -> flatten to list of individual scalars
                    for t in range(values.shape[0]):
                        for b in range(values.shape[1]):
                            all_data[key].append(values[t, b])
                elif values.ndim == 1:
                    # Could be single env case (T,) or bootstrap values (B,)
                    if key == 'bootstrap_values':
                        # Skip bootstrap values - they're trajectory-level metadata
                        continue
                    else:
                        # Single env case: (T,) -> extend as list
                        all_data[key].extend(values.tolist())
                else:
                    # Other cases
                    all_data[key].extend(values.flatten().tolist())
            else:
                all_data[key].append(values)
    
    # Debug lengths
    total_steps = len(all_data['observations']) if 'observations' in all_data else 0
    print(f"Total steps: {total_steps}")
    for key, values in all_data.items():
        print(f"  {key}: {len(values)} items")
        if len(values) != total_steps:
            print(f"    ^ This key will be SKIPPED!")
    
    try:
        batch = buffer.sample(batch_size=100)  # Sample a smaller batch
        print("Batch keys:", list(batch.keys()))
        if 'actions' in batch:
            print("Actions tensor shape:", batch['actions'].shape)
        else:
            print("ERROR: 'actions' key missing from batch!")
            
        return batch
    except Exception as e:
        print(f"Sampling failed: {e}")
        return None

if __name__ == "__main__":
    batch = test_buffer()