# Agent Handoff Template

Use this when handing work to another agent or returning from a long run.

```text
Goal:

Current repo state:

Files touched:

Commands run:

Passing tests:

Failing tests or warnings:

Run ids or experiment dirs:

Remote session names:

Checkpoints:

TensorBoard or scalar paths:

Diagnostics produced:

Manifest rows changed:

Narrative docs changed:

Open questions:

Stop conditions:

Recommended next action:
```

## Rules

- Include exact paths, not vague references.
- Include command outcomes, not only commands.
- If a GPU run is active, include the session name and status command.
- If a claim depends on a run, include the run artifact path.
- If code was changed on WSL, include the patch path and whether it was applied
  locally.
