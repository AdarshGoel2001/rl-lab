# Executable Documentation Audit

This file maps important agent-facing claims to executable checks. It is not a
complete test plan; it is the first place to look when a doc claim should be
kept honest by code.

## Entrypoints

| Claim | Doc | Check |
| --- | --- | --- |
| `AGENTS.md` is the single agent entrypoint. | `AGENTS.md` | `tests/test_docs_entrypoints.py` |
| `README.md` points to the current research loop. | `README.md` | `tests/test_docs_entrypoints.py` |
| Repo map separates live and future material. | `docs/repo_map.md` | `tests/test_docs_entrypoints.py` |
| Agent team roles and handoff packet are documented. | `docs/agent_team_operating_model.md` | `tests/test_docs_entrypoints.py` |

## Config And Workflow

| Claim | Doc | Check |
| --- | --- | --- |
| PlaNet CartPole tiny config resolves. | `docs/agentic_workflow.md` | `tests/test_validate_experiment.py` |
| PlaNet DMC budgets resolve and preserve intended scale. | `docs/roadmap/eval_ladder.md` | `tests/test_dmc_cartpole_swingup_config.py` |
| Evaluation episode accounting belongs above the workflow. | `docs/contracts/workflow_data_contract.md` | `tests/test_orchestrator_evaluation_contract.py` |
| PlaNet evaluate logs vectorized episode accounting. | `docs/contracts/workflow_data_contract.md` | `tests/test_planet_tiny.py` |

## Artifacts And Resume

| Claim | Doc | Check |
| --- | --- | --- |
| Run artifacts have a required folder/status shape. | `docs/contracts/run_artifacts.md` | `tests/test_run_artifact_contract.py` |
| Best checkpoint pointers resolve to immutable files. | `docs/contracts/run_artifacts.md` | `tests/test_best_checkpoint_retention.py` |
| Resume modes distinguish exact and warm starts. | `AGENTS.md`, `docs/contracts/buffers.md` | `tests/test_resume_modes.py` |
| Run status is machine-readable for polling agents. | `docs/contracts/run_artifacts.md` | `tests/test_orchestrator_run_status.py` |

## GPU And Diagnostics

| Claim | Doc | Check |
| --- | --- | --- |
| GPU helper scripts expose bounded commands. | `docs/contracts/run_artifacts.md` | `tests/test_gpu_scripts_contract.py` |
| TensorBoard scalars can be exported without a browser. | `docs/contracts/run_artifacts.md` | `tests/test_tensorboard_export.py` |
| PlaNet checkpoint diagnostics can write artifacts. | `docs/repo_map.md` | `tests/test_planet_diagnostics.py` |

## Components And Buffers

| Claim | Doc | Check |
| --- | --- | --- |
| RSSM, MLP heads, and MPC planner support the PlaNet workflow. | `docs/contracts/component_interfaces.md` | `tests/test_planet_tiny.py` |
| `WorldModelSequenceBuffer` supports sequence sampling and persistence. | `docs/contracts/buffers.md` | `tests/test_world_model_sequence_buffer.py` |
| Support dynamics and representation components still have smoke coverage. | `docs/repo_inventory.md` | `tests/test_gaussian_gru_dynamics.py`, `tests/test_world_model_system.py` |
| Hydra component configs can be instantiated in isolation. | `docs/repo_inventory.md` | `tests/test_hydra_component_switching.py` |

## Audit Rule

If a doc introduces a new durable claim, add one of:

- a test;
- a validation script;
- a documented manual command with expected output;
- an explicit note that the claim is a plan, not implemented behavior.
