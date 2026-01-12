"""
Checkpoint Management System

Pure I/O layer for saving and loading training checkpoints.
The orchestrator is responsible for assembling checkpoint state;
this module only handles disk operations, symlinks, and cleanup.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages saving and loading training checkpoints.

    Pure I/O layer - accepts pre-assembled state dicts and handles:
    - Saving to disk with torch.save
    - Loading from disk with torch.load
    - Symlink management (latest.pt, best.pt)
    - Old checkpoint cleanup
    - RNG state capture/restore
    """

    def __init__(
        self,
        experiment_dir: Union[str, Path],
        max_checkpoints: int = 5,
        compress: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            experiment_dir: Directory to store experiment files
            max_checkpoints: Maximum number of checkpoints to keep (0 = unlimited)
            compress: Whether to compress checkpoint files
        """
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.max_checkpoints = max_checkpoints
        self.compress = compress

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Checkpoint manager initialized at {self.checkpoint_dir}")

    def save(
        self,
        state: Dict[str, Any],
        step: int,
        name: Optional[str] = None,
        is_best: bool = False,
    ) -> Path:
        """
        Save pre-assembled state dict to disk.

        Args:
            state: Pre-assembled checkpoint state dictionary
            step: Current training step
            name: Optional name for checkpoint (auto-generated if None)
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint file
        """
        if name is None:
            name = f"step_{step}"

        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        # Add metadata and RNG states (step comes from state["global_step"])
        checkpoint_data = {
            **state,
            "timestamp": datetime.now().isoformat(),
            "rng_states": self._get_rng_states(),
        }

        # Save checkpoint
        try:
            if self.compress:
                torch.save(
                    checkpoint_data,
                    checkpoint_path,
                    _use_new_zipfile_serialization=True,
                )
            else:
                torch.save(checkpoint_data, checkpoint_path)

            logger.info(f"Checkpoint saved: {checkpoint_path} (step {step})")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

        # Update symbolic links
        self._update_latest_link(checkpoint_path)
        if is_best:
            self._update_best_link(checkpoint_path)

        # Clean up old checkpoints
        if self.max_checkpoints > 0:
            self._cleanup_old_checkpoints()

        # Save checkpoint metadata
        self._save_checkpoint_metadata(
            {"step": step, "name": name, "is_best": is_best}, checkpoint_path
        )

        return checkpoint_path

    def load(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_latest: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from file.

        Args:
            checkpoint_path: Path to specific checkpoint file (None for auto-detection)
            load_latest: If True and no path given, load latest checkpoint

        Returns:
            Checkpoint data dictionary or None if no checkpoint found
        """
        if checkpoint_path is None:
            if load_latest:
                checkpoint_path = self.get_latest_path()
            else:
                return None

        if checkpoint_path is None:
            logger.info("No checkpoint found to load")
            return None

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return None

        try:
            # PyTorch 2.6+ defaults to weights_only=True; our checkpoints include full state.
            try:
                checkpoint_data = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
            except TypeError:
                # Older torch without weights_only argument
                checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

            logger.info(
                f"Checkpoint loaded: {checkpoint_path} (step {checkpoint_data.get('global_step', 'unknown')})"
            )
            return checkpoint_data

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise

    def get_latest_path(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        latest_link = self.checkpoint_dir / "latest.pt"

        if latest_link.exists():
            return latest_link

        # Fallback: find most recent checkpoint file
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        if checkpoint_files:
            # Sort by modification time
            latest_file = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            return latest_file

        return None

    def restore_rng_states(self, checkpoint_data: Dict[str, Any]) -> None:
        """Restore RNG states from checkpoint data."""
        rng_states = checkpoint_data.get("rng_states", {})
        if rng_states:
            self._restore_rng_states(rng_states)

    # ------------------------------------------------------------------
    # RNG state helpers
    # ------------------------------------------------------------------

    def _get_rng_states(self) -> Dict[str, Any]:
        """Get current random number generator states."""
        rng_states = {
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

        # PyTorch CPU RNG state
        try:
            rng_states["torch_cpu"] = torch.get_rng_state()
        except Exception as e:
            logger.warning(f"Could not get PyTorch CPU RNG state: {e}")

        # PyTorch CUDA RNG states (if available)
        try:
            if torch.cuda.is_available():
                rng_states["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception as e:
            logger.warning(f"Could not get PyTorch CUDA RNG state: {e}")

        # PyTorch MPS RNG state (if available)
        try:
            if torch.backends.mps.is_available():
                rng_states["torch_mps"] = torch.mps.get_rng_state()
        except Exception as e:
            logger.warning(f"Could not get PyTorch MPS RNG state: {e}")

        return rng_states

    def _restore_rng_states(self, rng_states: Dict[str, Any]) -> None:
        """Restore random number generator states."""
        try:
            if "numpy" in rng_states:
                np.random.set_state(rng_states["numpy"])
        except Exception as e:
            logger.warning(f"Could not restore NumPy RNG state: {e}")

        try:
            if "python" in rng_states:
                random.setstate(rng_states["python"])
        except Exception as e:
            logger.warning(f"Could not restore Python RNG state: {e}")

        try:
            if "torch_cpu" in rng_states:
                torch.set_rng_state(rng_states["torch_cpu"])
        except Exception as e:
            logger.warning(f"Could not restore PyTorch CPU RNG state: {e}")

        try:
            if "torch_cuda" in rng_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_states["torch_cuda"])
        except Exception as e:
            logger.warning(f"Could not restore PyTorch CUDA RNG state: {e}")

        try:
            if "torch_mps" in rng_states and torch.backends.mps.is_available():
                torch.mps.set_rng_state(rng_states["torch_mps"])
        except Exception as e:
            logger.warning(f"Could not restore PyTorch MPS RNG state: {e}")

    # ------------------------------------------------------------------
    # Symlink and cleanup helpers
    # ------------------------------------------------------------------

    def _update_latest_link(self, checkpoint_path: Path) -> None:
        """Update symbolic link to latest checkpoint."""
        latest_link = self.checkpoint_dir / "latest.pt"

        try:
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(checkpoint_path.name)
        except Exception as e:
            logger.warning(f"Could not update latest checkpoint link: {e}")

    def _update_best_link(self, checkpoint_path: Path) -> None:
        """Update symbolic link to best checkpoint."""
        best_link = self.checkpoint_dir / "best.pt"

        try:
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            best_link.symlink_to(checkpoint_path.name)
        except Exception as e:
            logger.warning(f"Could not update best checkpoint link: {e}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to free disk space."""
        checkpoint_files = [
            f
            for f in self.checkpoint_dir.glob("*.pt")
            if not f.is_symlink() and f.name not in ["latest.pt", "best.pt"]
        ]

        if len(checkpoint_files) <= self.max_checkpoints:
            return

        # Sort by modification time, keep most recent
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        files_to_remove = checkpoint_files[self.max_checkpoints :]

        for file_path in files_to_remove:
            try:
                file_path.unlink()
                # Also remove associated metadata file if it exists
                metadata_file = file_path.with_suffix(".json")
                if metadata_file.exists():
                    metadata_file.unlink()
                logger.debug(f"Removed old checkpoint: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove old checkpoint {file_path}: {e}")

    def _save_checkpoint_metadata(
        self, metadata: Dict[str, Any], checkpoint_path: Path
    ) -> None:
        """Save checkpoint metadata to JSON file."""
        metadata_path = checkpoint_path.with_suffix(".json")

        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save checkpoint metadata: {e}")

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with metadata.

        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("*.pt"):
            if checkpoint_file.is_symlink():
                continue

            metadata_file = checkpoint_file.with_suffix(".json")
            metadata = {}

            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(
                        f"Could not read metadata for {checkpoint_file}: {e}"
                    )

            checkpoints.append(
                {
                    "path": str(checkpoint_file),
                    "size_mb": checkpoint_file.stat().st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(
                        checkpoint_file.stat().st_mtime
                    ).isoformat(),
                    **metadata,
                }
            )

        # Sort by step or modification time
        checkpoints.sort(key=lambda x: x.get("step", 0), reverse=True)
        return checkpoints
