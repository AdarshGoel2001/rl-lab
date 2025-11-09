# Memory Growth Investigation – Next Steps

Context
- Run analyzed: `experiments/dreamer_ataripong-20251109_215746`
- Evidence from CUDA snapshots and TensorBoard shows linear growth of device small‑pool active memory with env steps during warmup (controller updates disabled).
- Growth characteristics:
  - Small pool increases from ~0.39 GB @1k → ~9.31 GB @45k; large pool flat at ~0.17 GB.
  - Active block count grows steadily (8,695 → 382,744), indicating many small tensors retained.
  - TB `gpu_allocated_gb` matches snapshots closely at the same steps.

Hypothesis to validate (warmup only)
- Autograd state created during online rollout/collection or world‑model update is being retained across iterations, leading to accumulation of many small device allocations.
- We must localize whether the anchor is in data collection (online RSSM updates + actor/critic forward) or in the world‑model loss path (encoder → RSSM → decoder/reward/kl) during update.

What has been changed already
- In `src/workflows/world_models/dreamer.py` (collect_step), the per‑step torch block (encoder → rssm.observe(detach_posteriors=True) → actor/critic forward) is wrapped with `torch.no_grad()` to ensure collection does not build/retain autograd graphs during warmup.
  - Rationale: If online rollout graphs were retained via cached state, this change should significantly reduce or flatten the growth slope.

Step 1 — Quick re‑run to measure slope after no‑grad collect
- Objective: Determine if collection was anchoring the graph.
- How:
  - Run a short warmup (e.g., 10–20k env steps) to generate snapshots every 1k steps.
  - Analyze with the enhanced script:
    - `python analyze_memory_snapshot.py --snapshots-dir <run>/experiments/memory_snapshots --events-dir <run>/runs/<run_name>/<timestamp> --out-dir <run>/experiments/memory_snapshots/analysis`
- Expected signals:
  - If slope collapses (near‑flat small‑pool growth), collection was the source.
  - If linear slope persists, the anchor is in the world‑model update path.
- Why this step: Minimal, decisive A/B to split “collect vs update” as root cause without refactors.

Latest result (after no‑grad collect)
- Observation: The warmup run progressed much further with `torch.no_grad()` applied in collect.
- Interpretation:
  - Strongly supports that autograd graphs built during online collection (encoder → RSSM.observe → actor/critic forward) were being retained across env steps, driving the linear small‑pool growth.
  - With no‑grad, collection no longer constructs graphs, removing the inter‑step anchor and allowing memory to remain bounded.
- Consequence:
  - We can adopt no‑grad collection as a default for warmup and, in practice, for all on‑environment data collection since training of controllers happens via imagined rollouts, not via backprop through environment steps.

Step 2 — Fine‑grained memory checkpoints (only if slope persists)
- Objective: Pinpoint where in the WM update the baseline ratchets up.
- How (temporary logging for a short run): Record `gpu_allocated_gb` at these points each iteration:
  - start of loop; after collect; immediately after sampling batch; after encoder; after RSSM; after decoder; after backward; after optimizer; start of next iteration.
- Expected signals:
  - The “start of next iteration” baseline keeps increasing, and the nearest step preceding it identifies the anchor stage.
- Why: High‑resolution telemetry narrows to a module/loss branch before making any code change.

Step 3 — Toggle WM loss branches to isolate the anchor
- Objective: Determine which loss path retains graphs.
- How (short runs, one change at a time):
  - Disable decoder reconstruction loss; then reward predictor loss; then KL term.
  - Keep everything else constant (batch size, sequence length, num_envs, total steps).
- Expected signals:
  - If the slope drops substantially when a specific branch is disabled, that branch anchors the graph.
- Why: Each branch (decoder/reward/kl) adds distinct compute graphs with many small saved tensors; identifying the dominant contributor targets the fix.

Step 4 — Scale with sequence_length (L)
- Objective: Verify that growth tracks the number of per‑step activations retained.
- How: Run with `sequence_length` halved (e.g., 25 vs 50) while holding env steps comparable.
- Expected signals:
  - Slope (GB per 1k env steps) should reduce proportionally if retention is per‑time‑step.
- Why: Confirms that accumulation corresponds to per‑step activations rather than batch‑level artifacts.

Step 5 — Confirm final fix direction (no code yet)
- If Step 1 flattens growth:
  - Adopt `torch.no_grad()` for the collection path permanently (at least during warmup), since we do not require gradients through real environment steps for Dreamer; policy learning uses imagined rollouts.
  - In addition, harden RSSM online state management: when `detach_posteriors=True` is passed to `observe`, ensure cached internal state (`self._state`, `self._prior_state`) stores detached tensors. This avoids accidental graph retention even if no‑grad is removed by mistake in future changes.
- If Steps 2–4 implicate a specific WM branch:
  - Target that path to detach intermediates promptly (e.g., ensure no retained references/hook objects keep graphs alive after backward), or refactor to reduce saved tensors.

Data collection checklist for each run
- Ensure snapshots are being saved every 1k timesteps (train.log contains “Saved memory snapshot … snapshot_step_X.pickle”).
- After run, execute:
  - `python analyze_memory_snapshot.py --snapshots-dir <run>/experiments/memory_snapshots --events-dir <run>/runs/<run_name>/<timestamp>`
  - Review:
    - Growth table (total active GB, small/large pool breakdown, active block counts).
    - TB scalars correlation (`train/memory/gpu_allocated_gb`, `gpu_reserved_gb`).
    - Top contexts if frames are present in snapshots.

Success criteria
- Slope of device small‑pool active GB vs env steps is flat or near‑flat during warmup.
- No monotonic increase in “start of next iteration” baseline after optimizer step.
- Training proceeds well beyond 50k steps without memory‑driven termination.

Why this plan
- Each step is incremental, isolates a single factor, and produces clear signals (slope change) that decisively narrow the root cause.
- It avoids premature fixes and focuses on measurement, then surgical remediation based on evidence.

Recommended next actions (based on latest result)
- Keep `torch.no_grad()` wrapping in `collect_step` as a permanent change (at least for warmup). This aligns with Dreamer’s design where controller learning uses imagined trajectories, not real env gradients.
- Add a follow‑up robustness change: in RSSM `observe`, when called with `detach_posteriors=True`, cache `posterior.detach()` and `prior.detach()` in internal state. Rationale: prevents future regressions where online state inadvertently holds computation graphs.
- Run a short joint‑training test (after warmup) to ensure memory remains stable when controller updates are active:
  - Monitor snapshots/TB up to ~20–30k env steps into the joint phase.
  - If any slope re‑appears, re‑examine `update_controller` (e.g., necessity of `retain_graph=True`, and lifetime of rollout tensors in imagination).
