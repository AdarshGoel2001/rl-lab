"""
Script to download D4RL datasets from HuggingFace.
This avoids the need for mujoco_py installation.
"""

import urllib.request
import os
from pathlib import Path
from tqdm import tqdm

# Set up directories
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "d4rl"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace repository base URL
HF_BASE_URL = "https://huggingface.co/datasets/imone/D4RL/resolve/main"

# Datasets to download (files are in root of repo)
DATASETS = [
    # Hopper (easiest - good starting point)
    "hopper_expert-v2.hdf5",
    "hopper_medium-v2.hdf5",
    "hopper_medium_expert-v2.hdf5",

    # HalfCheetah (medium difficulty)
    "halfcheetah_expert-v2.hdf5",
    "halfcheetah_medium-v2.hdf5",
    "halfcheetah_medium_expert-v2.hdf5",

    # Walker2d (harder)
    "walker2d_expert-v2.hdf5",
    "walker2d_medium-v2.hdf5",
    "walker2d_medium_expert-v2.hdf5",
]


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> bool:
    """Download a file with progress bar."""
    try:
        with DownloadProgressBar(
            unit='B',
            unit_scale=True,
            miniters=1,
            desc=output_path.name
        ) as t:
            urllib.request.urlretrieve(
                url,
                filename=output_path,
                reporthook=t.update_to
            )
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def get_file_info(file_path: Path) -> dict:
    """Get info about a downloaded file."""
    try:
        import h5py

        with h5py.File(file_path, 'r') as f:
            info = {
                'observations': f['observations'].shape,
                'actions': f['actions'].shape,
                'rewards': f['rewards'].shape,
                'terminals': f['terminals'].shape,
                'total_transitions': len(f['observations']),
            }

            # Calculate stats
            import numpy as np
            rewards = f['rewards'][:]
            terminals = f['terminals'][:]

            info['mean_reward'] = float(np.mean(rewards))
            info['std_reward'] = float(np.std(rewards))
            info['num_episodes'] = int(np.sum(terminals))

        return info
    except Exception as e:
        return {'error': str(e)}


def main():
    """Download D4RL datasets from HuggingFace."""
    print("D4RL Dataset Downloader (HuggingFace)")
    print("="*60)
    print(f"\nDatasets will be saved in: {DATASET_DIR}\n")

    # Check existing files
    existing_files = list(DATASET_DIR.glob("*.hdf5"))
    if existing_files:
        print(f"Found {len(existing_files)} existing dataset(s):")
        for f in existing_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        print()

    # Download each dataset
    success_count = 0
    for filename in DATASETS:
        output_path = DATASET_DIR / filename

        # Skip if already exists
        if output_path.exists():
            print(f"⊙ Skipping {filename} (already exists)")
            success_count += 1
            continue

        print(f"\nDownloading: {filename}")
        url = f"{HF_BASE_URL}/{filename}"

        if download_file(url, output_path):
            print(f"  ✓ Downloaded successfully")

            # Try to get info
            info = get_file_info(output_path)
            if 'error' not in info:
                print(f"    Observations: {info['observations']}")
                print(f"    Actions: {info['actions']}")
                print(f"    Total transitions: {info['total_transitions']}")
                print(f"    Mean reward: {info['mean_reward']:.2f}")
                print(f"    Episodes: {info['num_episodes']}")

            success_count += 1
        else:
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()

    # Summary
    print(f"\n{'='*60}")
    print(f"Download Complete!")
    print(f"{'='*60}")
    print(f"Successfully downloaded: {success_count}/{len(DATASETS)} datasets")

    # List all files
    all_files = sorted(DATASET_DIR.glob("*.hdf5"))
    if all_files:
        print(f"\nAll available datasets ({len(all_files)}):")
        total_size = 0
        for f in all_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        print(f"\nTotal size: {total_size:.1f} MB")


if __name__ == "__main__":
    main()
