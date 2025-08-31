#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.trainer import Trainer
from src.utils.config import load_config

def debug_training():
    config_path = "configs/experiments/ppo_pendulum_5M.yaml"
    config = load_config(config_path)
    
    # Override to run enough steps for full buffer
    config.training.total_timesteps = 5000
    config.logging.log_frequency = 100
    
    trainer = Trainer(config)
    
    print(f"Buffer capacity: {trainer.buffer.capacity}")
    print(f"Buffer batch_size: {trainer.buffer.batch_size}")
    print(f"Initial buffer size: {trainer.buffer._size}")
    print(f"Buffer ready: {trainer.buffer.ready()}")
    
    # Manually test one training step
    print("\n--- Testing one training step ---")
    trajectory = trainer._collect_experience()
    
    if trajectory is not None:
        print(f"Collected trajectory with {len(trajectory['observations'])} steps")
        trainer.buffer.add(trajectory=trajectory)
        print(f"Buffer size after add: {trainer.buffer._size}")
        print(f"Buffer ready after add: {trainer.buffer.ready()}")
        
        if trainer.buffer.ready():
            print("Buffer is ready - attempting update...")
            batch = trainer.buffer.sample()
            print(f"Sampled batch keys: {list(batch.keys())}")
            print(f"Batch size: {batch['observations'].shape[0]}")
            
            # Try the update
            try:
                update_metrics = trainer.algorithm.update(batch)
                print(f"Update successful! Metrics: {list(update_metrics.keys())}")
            except Exception as e:
                print(f"Update failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Buffer still not ready")
    else:
        print("No trajectory collected")

if __name__ == "__main__":
    debug_training()