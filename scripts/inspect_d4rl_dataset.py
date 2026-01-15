"""
Script to inspect D4RL dataset files.
"""

import h5py
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "d4rl"


def inspect_dataset(filepath: Path):
    """Inspect a single D4RL dataset file."""
    print(f"\n{'='*70}")
    print(f"Dataset: {filepath.name}")
    print(f"{'='*70}")

    with h5py.File(filepath, 'r') as f:
        print("\nKeys in HDF5 file:")
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                shape = item.shape
                dtype = item.dtype
                print(f"  {key:20s} shape={shape}, dtype={dtype}")
            else:
                print(f"  {key:20s} (group)")

        # Get data
        observations = f['observations'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        terminals = f['terminals'][:]

        # Basic stats
        print(f"\nDataset Statistics:")
        print(f"  Total transitions: {len(observations):,}")
        print(f"  Observation dim:   {observations.shape[1]}")
        print(f"  Action dim:        {actions.shape[1]}")

        # Reward stats
        print(f"\n  Reward statistics:")
        print(f"    Mean:   {rewards.mean():8.3f}")
        print(f"    Std:    {rewards.std():8.3f}")
        print(f"    Min:    {rewards.min():8.3f}")
        print(f"    Max:    {rewards.max():8.3f}")
        print(f"    Median: {np.median(rewards):8.3f}")

        # Episode stats
        num_episodes = terminals.sum()
        print(f"\n  Episodes: {int(num_episodes)}")

        if num_episodes > 0:
            # Calculate episode returns
            episode_starts = np.where(np.concatenate([[True], terminals[:-1]]))[0]
            episode_ends = np.where(terminals)[0]

            episode_returns = []
            episode_lengths = []

            for start, end in zip(episode_starts, episode_ends):
                episode_return = rewards[start:end+1].sum()
                episode_length = end - start + 1
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)

            episode_returns = np.array(episode_returns)
            episode_lengths = np.array(episode_lengths)

            print(f"\n  Episode Returns:")
            print(f"    Mean:   {episode_returns.mean():8.1f}")
            print(f"    Std:    {episode_returns.std():8.1f}")
            print(f"    Min:    {episode_returns.min():8.1f}")
            print(f"    Max:    {episode_returns.max():8.1f}")
            print(f"    Median: {np.median(episode_returns):8.1f}")

            print(f"\n  Episode Lengths:")
            print(f"    Mean:   {episode_lengths.mean():8.1f}")
            print(f"    Std:    {episode_lengths.std():8.1f}")
            print(f"    Min:    {int(episode_lengths.min())}")
            print(f"    Max:    {int(episode_lengths.max())}")

        # Action stats
        print(f"\n  Action statistics:")
        print(f"    Mean:   {actions.mean(axis=0)}")
        print(f"    Std:    {actions.std(axis=0)}")
        print(f"    Min:    {actions.min(axis=0)}")
        print(f"    Max:    {actions.max(axis=0)}")


def main():
    """Inspect D4RL datasets."""
    if len(sys.argv) > 1:
        # Inspect specific dataset
        dataset_name = sys.argv[1]
        if not dataset_name.endswith('.hdf5'):
            dataset_name += '.hdf5'

        filepath = DATASET_DIR / dataset_name
        if not filepath.exists():
            print(f"Error: Dataset not found: {filepath}")
            print(f"\nAvailable datasets:")
            for f in sorted(DATASET_DIR.glob("*.hdf5")):
                print(f"  - {f.name}")
            return

        inspect_dataset(filepath)
    else:
        # List all datasets
        print("Available D4RL Datasets")
        print("="*70)

        datasets = sorted(DATASET_DIR.glob("*.hdf5"))
        if not datasets:
            print("No datasets found. Run download_d4rl_hf.py first.")
            return

        print(f"\nFound {len(datasets)} datasets in {DATASET_DIR}:\n")

        for f in datasets:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.stem:35s} {size_mb:7.1f} MB")

        print(f"\nUsage: python {Path(__file__).name} <dataset_name>")
        print(f"Example: python {Path(__file__).name} hopper_expert-v2")


if __name__ == "__main__":
    main()
