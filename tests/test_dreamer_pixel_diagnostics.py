import numpy as np
import torch

from scripts.research.diagnostics.diagnose_dreamer_pixel_checkpoint import (
    build_reconstruction_grid,
    summarize_reconstructions,
)


def test_summarize_reconstructions_reports_pixel_errors():
    actual = torch.zeros(2, 3, 3, 4, 4)
    recon = torch.ones(2, 3, 3, 4, 4) * 0.25

    summary = summarize_reconstructions(actual, recon)

    assert summary["schema_version"] == 1
    assert summary["num_frames"] == 6
    assert summary["reconstruction_mse"] == 0.0625
    assert summary["reconstruction_mae"] == 0.25
    assert summary["actual_min"] == 0.0
    assert summary["actual_max"] == 0.0
    assert summary["reconstruction_min"] == 0.25
    assert summary["reconstruction_max"] == 0.25


def test_build_reconstruction_grid_interleaves_actual_and_reconstruction_rows():
    actual = torch.zeros(1, 2, 3, 4, 4)
    recon = torch.ones(1, 2, 3, 4, 4)

    grid = build_reconstruction_grid(actual, recon, max_frames=2)

    assert grid.shape == (8, 8, 3)
    assert grid.dtype == np.float32
    assert np.allclose(grid[:4], 0.0)
    assert np.allclose(grid[4:], 1.0)
