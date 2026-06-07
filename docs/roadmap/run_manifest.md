# Run Manifest

`reports/world_model_runs.csv` is the graphable index for the world-model
chronology. It is the bridge between the narrative docs and the actual run
artifacts.

The manifest should answer one question:

```text
What exact run supports this claim in the chronology?
```

It should stay small. It is not a database, not a replacement for TensorBoard,
and not a place to paste long analysis.

## Why It Exists

The chronology compares small implementations across a fixed evaluation ladder.
Official paper scores are not directly comparable, so this repo needs its own
controlled result table.

The manifest makes that table reproducible:

- each row points to the paper idea;
- each row points to the implementation and config;
- each row points to the environment and eval layer;
- each row points to the remote or local run directory;
- each completed row records score, budget, artifacts, and the claim it
  supports.

Do not use the manifest as a substitute for logs. The manifest should point to
logs and summarize only the fields needed for comparison and plotting.

The manifest is filled from the run artifact contract in
`docs/contracts/run_artifacts.md`. Do not add claims without a run directory or
an explicit failure note.

## File

```text
reports/world_model_runs.csv
```

This CSV is intentionally machine-readable so it can later drive plots such as:

- score by chronology year;
- score by benchmark layer;
- GPU time by model family;
- rollout/prediction quality by model family;
- solved/not-solved status across the evaluation ladder.

## Status Values

Use exactly one of:

| Status | Meaning |
| --- | --- |
| `planned` | We know the run should exist, but it has not started. |
| `running` | The run is currently active or incomplete. |
| `complete` | The run finished and has a final score or final failure record. |
| `failed` | The run crashed or produced an invalid result. |
| `superseded` | A newer row replaces this one, but the old row is kept for history. |

## Important Columns

| Column | Meaning |
| --- | --- |
| `run_id` | Stable unique id. Prefer `<experiment>_<timestamp>`. |
| `status` | One of the allowed status values. |
| `chapter` | The chronology chapter this run supports. |
| `paper` | Paper or idea being represented. |
| `year` | Paper year, used for chronology plots. |
| `model_family` | Short family label, such as `RSSM+CEM` or `VAE+MDN-RNN`. |
| `implementation` | Main workflow or component file. |
| `config` | Hydra experiment config used for the run. |
| `environment` | Human-readable environment name. |
| `modality` | `state`, `pixels`, `tokens`, `latent`, or mixed description. |
| `eval_layer` | Layer from `docs/roadmap/eval_ladder.md`. |
| `budget_tier` | `local smoke`, `WSL tiny`, `WSL chapter`, or `stretch`. |
| `remote_host` | Remote machine if used. |
| `gpu` | GPU model if used. |
| `run_dir` | Original experiment directory. This may be remote. |
| `artifact_dir` | Local copied artifact directory, once pulled back. |
| `eval_return_mean` | Final mean eval return. Leave blank until final. |
| `best_eval_return_mean` | Best eval mean if tracked. |
| `primary_loss` | Main final training loss if useful. |
| `claim` | One short sentence this run supports. |
| `failure_notes` | Short failure note if incomplete, failed, or below target. |
| `next_action` | Concrete next step. |

## Update Rules

For a new run:

1. Add a row as `running` when the run starts.
2. Fill `run_dir`, `remote_host`, `gpu`, config, environment, and eval layer.
3. Leave score fields blank until the run finishes.
4. Pull artifacts back with `scripts/GPU/gpu_pull_latest.sh` when the run is
   done or worth preserving.
5. Update the row to `complete` or `failed`.
6. Fill `artifact_dir`, score fields, wall time, claim, failure notes, and next
   action.

Completed rows should be boring and factual. Analysis belongs in the chapter
writeup, not in the CSV.

## Current Anchor Rows

The first completed chronology row is the paper-aligned PlaNet DMC Swingup run:

```text
planet_dmc_swingup_paper_authentic_20260602_123034
```

It is a state-observation PlaNet-style run, not a full pixel reproduction. Use
it as the Layer 2 state world-model benchmark anchor for RSSM+CEM until a newer
completed row supersedes it.

Older rows may remain as `superseded` to preserve the run history that led to a
clean completed result.
