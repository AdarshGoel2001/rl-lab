# Agent Team Operating Model

This document describes how multiple coding-agent instances should work
together in RL Lab. Use it when a chapter is too large for one agent but still
belongs to one paper or one research claim.

## Prime Rule

Parallelize roles, not unrelated papers.

One chapter may have several agents working at once, but they should all serve
the same paper implementation or the same experiment question. Do not run five
paper implementations in parallel unless the user explicitly asks for that.

## Chapter Lifecycle

```text
paper brief -> implementation plan -> tiny validation -> GPU run
-> diagnostics -> manifest row -> narrative update -> review
```

Each stage must leave an artifact that another agent can inspect without hidden
chat context.

## Roles

### Paper Agent

Purpose: turn a paper into repo-local requirements.

Reads:

- `docs/roadmap/world_model_chronology.md`
- `docs/roadmap/eval_ladder.md`
- paper PDF/text/notes under `docs/papers/`

Writes:

- paper notes under `docs/papers/`
- a short chapter brief in `research_notes/rough_notes/` or a clean plan when
  the direction is approved

Must answer:

- What is the smallest faithful version for this repo?
- Which eval ladder rung can show progress?
- What compute assumptions from the paper are impossible here?

### Workflow Agent

Purpose: implement algorithm-specific training logic.

Owns:

- `src/workflows/<algorithm>.py`
- workflow-specific tests
- workflow-specific diagnostics hooks only when needed

Must not own:

- checkpoint I/O
- TensorBoard setup
- experiment directory policy
- generic buffer routing

The workflow owns losses, imagination, planner interaction, collection details,
and workflow-specific checkpoint state.

### Component Agent

Purpose: implement reusable model pieces.

Owns:

- `src/components/`
- `src/buffers/` only when the data contract is component-specific
- matching config groups under `configs/components/`, `configs/controller/`,
  or `configs/buffer/`

Must provide focused tests for tensor shapes, finite losses, and backward passes
when parameters are trainable.

### Config And Eval Agent

Purpose: make the experiment runnable and comparable.

Owns:

- `configs/experiment/`
- `configs/budget/`
- `configs/environment/`
- config-resolution tests

Must ensure:

- tiny budget resolves and runs locally;
- eval episode counts match vectorized env rules;
- serious budgets have explicit checkpoint and eval cadence;
- the run can later support a manifest row.

### GPU Ops Agent

Purpose: run serious experiments on the WSL GPU worker without making WSL the
source of truth.

Owns:

- `scripts/GPU/`
- remote run commands
- artifact pullback
- run status checks

Must follow:

- Mac repo is source of truth.
- Code moves to WSL by patch or git pull.
- Artifacts move back through `scripts/GPU/gpu_pull_latest.sh`.
- Remote edits come back only through reviewed patches.

### Diagnostics Agent

Purpose: inspect a run deeply enough to explain what happened.

Owns:

- `scripts/research/diagnostics/`
- TensorBoard export/plot helpers
- run-specific diagnostic outputs

Diagnostics may be model-specific. Prefer useful scripts over a shared framework
until two models genuinely need the same interface.

### Manifest And Narrative Agent

Purpose: turn run evidence into chronology.

Owns:

- `reports/world_model_runs.csv`
- `docs/roadmap/run_manifest.md`
- narrative notes that cite manifest rows and run artifacts

Must not make a chapter claim unless it can point to run artifacts.

### Reviewer Agent

Purpose: protect boundaries before work is merged.

Checks:

- Does orchestration still own infrastructure?
- Does workflow code still own algorithm logic?
- Are tests focused and executable?
- Are docs linked from an entrypoint?
- Can another agent reproduce the claim from artifacts?

## Handoff Packet

When handing work to another agent, leave this information:

```text
Goal:
Current repo state:
Files touched:
Commands run:
Passing tests:
Run ids or experiment dirs:
Artifacts produced:
Open questions:
Stop conditions:
```

For GPU runs, include the remote session name, experiment directory, checkpoint
path, TensorBoard log directory, and pullback command.

## Coordination Rules

- One agent should own a file at a time unless the user explicitly coordinates a
  merge.
- Shared contracts should be edited before dependent agents start coding.
- GPU Ops should not silently change code on WSL.
- Diagnostics should not mutate training code during an active run.
- Narrative should not outrun the manifest.
- If a result contradicts expectation, stop and explain before launching the
  next long run.

## Recommended Chapter Team

For the next paper chapter, use this sequence:

1. Paper Agent writes the chapter brief.
2. Reviewer Agent checks whether the brief fits the eval ladder.
3. Workflow and Component Agents implement the tiny version.
4. Config Agent makes local and WSL budgets.
5. Diagnostics Agent adds run-specific inspection scripts.
6. GPU Ops Agent runs the serious experiment.
7. Manifest Agent records the result.
8. Reviewer Agent verifies code, docs, and artifacts.
