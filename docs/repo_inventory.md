# Repo Inventory

This inventory classifies repo surfaces for agents. It should be updated when a
chapter graduates from future material to active implementation.

## Status Labels

```text
active: used by the current PlaNet chapter or core training loop
support: tested utility code that can be reused after inspection
future: planned or exploratory material, not a runnable workflow
stale: remove, rewrite, or move into notes before using
artifact: evidence from runs, not source code
```

## Active Surfaces

| Surface | Status | Why agents should care |
| --- | --- | --- |
| `AGENTS.md` | active | Single agent entrypoint. |
| `README.md` | active | Human overview and quick commands. |
| `docs/repo_map.md` | active | First discovery map. |
| `docs/agentic_workflow.md` | active | Single-agent extension loop. |
| `docs/agent_team_operating_model.md` | active | Multi-agent role split and handoff rules. |
| `docs/research_lifecycle.md` | active | Paper-to-narrative lifecycle. |
| `docs/contracts/*.md` | active | Boundary contracts. |
| `docs/roadmap/*.md` | active | Chronology, eval ladder, and manifest semantics. |
| `reports/world_model_runs.csv` | active | Graphable chronology manifest. |
| `scripts/train.py` | active | Hydra training entrypoint. |
| `scripts/validate_experiment.py` | active | Config-resolution check. |
| `src/orchestration/` | active | Loop, phase scheduling, logging, checkpointing, resume. |
| `src/workflows/planet.py` | active | Only live algorithm workflow. |
| `configs/workflow/planet.yaml` | active | Only live workflow config. |
| `configs/experiment/planet_cartpole.yaml` | active | Local smoke baseline. |
| `configs/experiment/planet_dmc_cartpole_swingup.yaml` | active | Current DMC chapter baseline. |
| `configs/budget/planet_*.yaml` | active | Current PlaNet budget family. |
| `scripts/GPU/` | active | WSL worker sync, run, status, and pullback tools. |
| `scripts/research/diagnostics/diagnose_planet_checkpoint.py` | active | Current PlaNet diagnostic example. |

## Active PlaNet Components

| Surface | Status | Notes |
| --- | --- | --- |
| `src/components/representation_learners/rssm.py` | active | PlaNet latent dynamics model. |
| `src/components/prediction_heads/mlp.py` | active | Reward, continuation, and observation heads. |
| `src/components/controllers/mpc_planner.py` | active | CEM planner used by PlaNet. |
| `src/components/controllers/random_policy.py` | active | Seed data collection. |
| `src/buffers/world_model_sequence.py` | active | Small world-model replay buffer. |
| `src/environments/dmc_wrapper.py` | active | DMC state env wrapper. |
| `src/environments/dmc_vectorized_wrapper.py` | active | Vectorized DMC state env wrapper. |
| `src/environments/vectorized_gym_wrapper.py` | active | Local CartPole vectorized smoke env. |

## Support Material

These files have tests or useful local structure, but they are not current
runnable workflows.

| Surface | Status | Required before use |
| --- | --- | --- |
| `src/components/representation_learners/conv_vae.py` | support | Revalidate for a new pixel workflow. |
| `src/components/representation_learners/identity.py` | support | Revalidate dimensions against the target workflow. |
| `src/components/dynamics/` | support | Treat as future world-model components, not an active OG workflow. |
| `src/components/return_computers/` | support | Use only when a workflow needs real-data returns. |
| `src/buffers/disk_buffer.py` | support | Finish persistence semantics before serious replay reuse. |
| `src/buffers/offline.py` | support | Use for custom offline loaders. |
| `configs/environment/` | support | Environment configs must be paired with a live experiment. |

## Future Or Planned Material

These are useful direction markers, not proof of implementation:

```text
docs/roadmap/world_model_chronology.md
docs/roadmap/eval_ladder.md
docs/papers/
research_notes/clean_plans/
```

## Stale Or Removed Surface

The active workflow cleanup removed stale runnable entrypoints for:

```text
OG World Models workflow
Dreamer workflow
TD-MPC workflow
Diffusion Policy workflow
CMA-ES controller scaffold
Diffusion policy controller scaffold
Dreamer actor/critic controller scaffold
old pixel FPS environment scaffold
D4RL sequence loader
```

Do not recreate those names as compatibility shims. Reimplement future chapters
from the paper brief, current contracts, and current tests.

## Artifact Surface

These are evidence, not source:

```text
experiments/
reports/world_model_runs.csv
research_notes/rough_notes/
research_notes/rough_notes/diagnostics_*/
reports/session_*/
```

Artifacts should support claims. They should not become the only place where an
interface or command is documented.
