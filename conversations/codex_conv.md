Final spec (concise)

1) CheckpointManager
- Keep minimal role: schema validation + save/load + metadata/symlinks/retention.
- API:
  - save_checkpoint(checkpoint: dict, step: int, name: Optional[str] = None, is_best: bool = False) -> Path
  - load_checkpoint(checkpoint_path: Optional[Union[str, Path]] = None, load_latest: bool = True) -> Optional[dict]

2) Checkpoint schema (top-level)
- schema_version: int
- step: int  (global_step)
- timestamp: str
- components: Dict[str, StateDict]  (WorldModelComponents.state_dict())
- optimizers: Dict[str, StateDict]
- controllers: Dict[str, StateDict]  (ControllerManager.state_dict())
- workflow_custom: Dict[str, Any]  (Workflow.get_custom_state())
- buffers: Dict[str, Dict[str, Any]]  (Buffer.save_checkpoint())
- scheduler_state: Dict[str, Any]  (PhaseScheduler.get_state())
- rng_state: Optional[Dict[str, Any]]
- metrics: Optional[Dict[str, Any]]
- metadata: Optional[Dict[str, Any]]

3) Save order (orchestrator)
1. components = context.components.state_dict()
2. optimizers = {name: opt.state_dict()}
3. controllers = controller_manager.state_dict()
4. workflow_custom = workflow.get_custom_state()
5. buffers = {name: buf.save_checkpoint()}
6. scheduler_state = scheduler.get_state()
7. assemble schema + step + optional rng_state/metrics/metadata
8. checkpoint_manager.save_checkpoint(schema, step, name, is_best)

4) Load order (orchestrator)
1. ckpt = checkpoint_manager.load_checkpoint(...)
2. self.global_step = ckpt["step"]
3. scheduler.set_state(ckpt["scheduler_state"])
4. context.components.load_state_dict(ckpt["components"])
5. for each optimizer: optimizer.load_state_dict(ckpt["optimizers"][name])
6. rebuild scheduler from config, set last_epoch=global_step, apply LR to optimizers
7. controller_manager.load_state_dict(ckpt["controllers"])
8. workflow.set_custom_state(ckpt["workflow_custom"])
9. for each buffer: buffer.load_checkpoint(ckpt["buffers"][name])
10. optional restore RNG state
11. update context global_step

5) Component responsibilities
- WorldModelComponents: strict state_dict/load_state_dict over modules only (no hasattr).
- Optimizers: orchestrator loops and saves/loads state.
- ControllerManager: state_dict/load_state_dict required.
- WorldModelWorkflow: get_custom_state/set_custom_state only.
- Buffers: save_checkpoint/load_checkpoint required.
- PhaseScheduler: get_state/set_state required.
- CheckpointManager: validation + disk I/O only.

6) File-level change list
- src/utils/checkpoint.py: remove get_state/set_state introspection; validate schema; keep I/O/metadata/retention.
- src/workflows/world_models/context.py: add strict state_dict/load_state_dict for modules.
- src/workflows/world_models/base.py: add get_custom_state/set_custom_state abstract API.
- src/workflows/world_models/dreamer.py: implement custom state (world_model_updates, episode tracking, total_episodes).
- src/orchestration/world_model_orchestrator.py: orchestrate save/load order; persist scheduler_state; restore LR via scheduler re-init at global_step.
- src/orchestration/phase_scheduler.py: add get_state/set_state.
- scripts/train.py: wire resume path into orchestrator restore.
