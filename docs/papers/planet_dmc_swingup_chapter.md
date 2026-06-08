# PlaNet DMC Cartpole Swingup Chapter

This is the current completed PlaNet-style chapter for the repo narrative.
Use it as the handoff point before starting Dreamer.

## Claim

The repo has a state-observation PlaNet-style RSSM + CEM implementation running
on DMC `cartpole_swingup`.

Run:

- id: `planet_dmc_swingup_paper_authentic_20260602_123034`
- remote run dir:
  `/home/omkar/adarsh/rl-lab/experiments/planet_dmc_swingup_paper_authentic-20260602_123034`
- local artifact dir:
  `remote_artifacts/wsl/planet_dmc_swingup_paper_authentic-20260602_123034`
- final/best eval mean: `637.53`
- eval std: `67.19`
- env steps: `178000`
- world-model updates: `50000`
- primary loss: `0.4587`

This does not solve the Layer 2 target yet. The current Layer 2 target is to
reach the high-control regime, roughly `>=800` mean return, on DMC
`cartpole_swingup`.

## Implementation

Main implementation:

- workflow: `src/workflows/planet.py`
- experiment config: `configs/experiment/planet_dmc_cartpole_swingup.yaml`
- paper-authentic budget: `configs/budget/planet_dmc_paper_authentic.yaml`
- planner family: CEM over latent RSSM rollouts
- observation modality: state observations, not pixels

This chapter is PlaNet-style rather than a pixel-faithful reproduction. It is
still useful because it exercises the same research loop shape: learn latent
dynamics from replay, plan with CEM, collect planner-guided data, repeat.

## Evidence

Durable summary:

- `reports/world_model_runs.csv`

Pulled local evidence:

- `.hydra/config.yaml`
- `.hydra/overrides.yaml`
- `pre_run/resolved_config.yaml`
- `pre_run/working_tree.diff`
- `train.log`
- `checkpoint_manifest.json`
- `diagnostics/planet_reward_open_loop_smoke/diagnostics_summary.json`
- `diagnostics/planet_reward_open_loop_smoke/reward_calibration.csv`
- `diagnostics/planet_reward_open_loop_smoke/open_loop_horizon_metrics.csv`

Some large files stayed on the GPU because the current Tailscale/SSH path stalls
on larger payloads. `pull_skipped_files.tsv` records skipped TensorBoard and PNG
files.

## Diagnostics Smoke

The diagnostic smoke was run on the GPU-side checkpoint:

`experiments/planet_dmc_swingup_paper_authentic-20260602_123034/checkpoints/best_snapshots/best_snapshot_step_178000.pt`

Key numbers:

- checkpoint global step: `178000`
- reward MAE: `0.0430`
- reward correlation: `0.9981`
- open-loop return MAE at horizon 1: `0.2770`
- open-loop return MAE at horizon 3: `0.7484`
- open-loop return MAE at horizon 5: `1.0857`
- open-loop return MAE at horizon 8: `1.3750`

Interpretation: the reward predictor and short open-loop model behavior look
reasonable on this smoke sample. That does not imply the planner solves the
environment. The benchmark score remains the authority for the chapter claim.

## Known Deviations And Problems

- This is state-observation PlaNet, not pixel PlaNet.
- The run did not reach the Layer 2 high-control target.
- A checkpoint retention bug meant the best evidence came from
  `checkpoints/best_snapshots/` rather than the normal immutable best pointer.
- Heavy raw artifacts are not fully pulled locally. Keep checkpoints and large
  TensorBoard files on the GPU unless a local inspection needs them.

## What Dreamer Inherits

The Dreamer chapter should inherit the proven infrastructure rather than
rebuild it:

- config-first experiment entry through `scripts/train.py`;
- orchestrator/workflow boundary;
- run artifact contract;
- manifest row before narrative claims;
- GPU-side diagnostics for checkpoint-heavy inspection;
- tiny live run snapshots before full artifact pulls.
