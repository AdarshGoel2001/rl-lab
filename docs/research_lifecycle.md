# Research Lifecycle

This is the end-to-end path for one world-model chapter. It is written for
agents that need to move from a paper idea to a reproducible repo claim.

```text
paper brief -> implementation plan -> tiny validation -> GPU run -> diagnostics -> manifest row -> narrative update -> review
```

## 1. Paper Brief

Owner: Paper Agent

Inputs:

```text
docs/roadmap/world_model_chronology.md
docs/roadmap/eval_ladder.md
docs/papers/
```

Output:

```text
research_notes/rough_notes/<chapter>_brief.md
```

Minimum brief:

```text
paper:
core idea:
small repo version:
required components:
required workflow behavior:
target eval rung:
expected compute:
known deviations from paper:
stop conditions:
```

## 2. Implementation Plan

Owner: Reviewer Agent plus Workflow/Component/Config agents

Output:

```text
research_notes/clean_plans/<chapter>_implementation_plan.md
```

The plan should identify the files each role owns and the first tiny test that
would prove the implementation can run.

## 3. Tiny Local Validation

Owner: Workflow, Component, and Config agents

Required commands:

```bash
python scripts/validate_experiment.py <experiment> --budget planet_tiny
python -m pytest <focused tests> -q
python scripts/train.py +experiment=<experiment> budget=planet_tiny
```

For a new chapter, `planet_tiny` may be replaced by a new chapter-specific tiny
budget. The budget must still be local, short, and deterministic enough to
debug.

## 4. GPU Run Preparation

Owner: GPU Ops Agent

Before launch:

```bash
scripts/GPU/gpu_status.sh
scripts/GPU/gpu_sync_patch.sh --paths "src scripts tests configs docs"
```

The Mac repo remains source of truth. If the remote worker has code edits, pull
them back only through `scripts/GPU/gpu_pull_patch.sh` and review before
applying.

## 5. Serious Run

Owner: GPU Ops Agent

Launch with an explicit session and budget:

```bash
scripts/GPU/gpu_run.sh --session <session> --experiment <experiment> --budget <budget> -- <override...>
```

During the run, poll cheap status first:

```bash
scripts/GPU/gpu_status.sh
scripts/GPU/gpu_tail.sh <session>
scripts/GPU/gpu_metrics.sh --run experiments/<run_name>
```

Stop and explain before continuing if eval return contradicts the expected
trend, losses become unusable, checkpoints are ambiguous, or the run cannot
produce required artifacts.

## 6. Artifact Pullback

Owner: GPU Ops Agent

Pull the named run:

```bash
scripts/GPU/gpu_pull_latest.sh --run experiments/<run_name> --analyze
```

The local run folder should follow `docs/contracts/run_artifacts.md`.

## 7. Diagnostics

Owner: Diagnostics Agent

Use model-specific scripts:

```bash
python scripts/research/diagnostics/diagnose_planet_checkpoint.py --help
python scripts/research/export_tensorboard_run.py --help
```

Diagnostics should write outputs into the run artifact tree, TensorBoard, or a
clearly named rough-notes artifact directory.

## 8. Manifest Row

Owner: Manifest And Narrative Agent

Update:

```text
reports/world_model_runs.csv
```

Follow:

```text
docs/roadmap/run_manifest.md
```

Do this before narrative claims. The manifest is the graphable chronology table.

## 9. Narrative Update

Owner: Manifest And Narrative Agent

Update roadmap or chapter notes only after the manifest row exists and points to
evidence. Narrative should say what improved, what failed, what was changed from
the paper, and what the next chapter inherits.

## 10. Review

Owner: Reviewer Agent

Review checklist:

```text
contracts respected:
tests passed:
run artifacts present:
manifest row added:
diagnostics sufficient:
known deviations recorded:
next action clear:
```

If any item is missing, leave a handoff packet instead of pretending the chapter
is complete.
