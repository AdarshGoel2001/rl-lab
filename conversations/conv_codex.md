# Codex Conversation: Dreamer Atari Pong OOM

Purpose: Shared working note for multi‑LLM collaboration on the CUDA OOM during Dreamer warmup on Atari Pong. Add your observations, experiment results, and suggestions under the relevant sections. Keep entries concise and dated.

## Context (from conversations/problem.md)
- Failure: `torch.OutOfMemoryError: Tried to allocate ~174 MiB` around ~53k steps, during world‑model update backward.
- Setup highlights: `B=16`, `L=50`, 84×84×4 frames, NatureCNN→RSSM→Atari deconv decoder; 2 envs; warmup only.
- EpisodeReplayBuffer V2 moved sampling to O(B×L) CPU path and stores obs as uint8; not implicated in GPU OOM.

## Initial View (Codex)
- Likely cause: Peak activation memory from processing `B×L` image frames through encoder → RSSM → decoder with autograd, plus KL and reward heads. Decoder deconvs dominate memory; flattening to `(B*L, latent)` and decoding all at once keeps a large graph alive.
- Contributing factors:
  - Full float32 training; no AMP or gradient checkpointing in world‑model path.
  - Peaks + CUDA allocator fragmentation → “sudden” OOM after many steps.
- Relevant code points:
  - Update start: `src/workflows/world_models/dreamer.py:366`
  - Decoder forward: `src/workflows/world_models/dreamer.py:408`
  - Backward: `src/workflows/world_models/dreamer.py:429`

## Proposed Mitigation Order
1) Config‑only headroom (fastest to validate)
   - `buffer.batch_size: 16 → 8` and/or `buffer.sequence_length: 50 → 32`.
   - Decoder channel reductions: base 32→16, mid 16→8.
2) Enable AMP on world‑model update
   - Wrap encoder/RSSM/decoder/reward in `torch.autocast(device_type, dtype)` and use `GradScaler` for backward/step.
   - Keep KL in float32 for stability.
3) Micro‑batch decoder reconstruction
   - Split `latent_flat` `(B*L, D)` into N chunks (e.g., 4); compute recon loss per chunk, accumulate, then single backward.
4) Optional: Gradient checkpointing on encoder/decoder blocks.
5) Resilience: Catch CUDA OOM and retry current update with lower micro‑batch factor; log once.

## Quick Experiments To Run (please claim + record)
- E1 Config shrink: B=8, L=50 → expect no OOM. [ ] Pending
- E2 Config shrink: B=16, L=32 → expect no OOM. [ ] Pending
- E3 AMP: Same B=16, L=50 with autocast+scaler → expect no OOM, ~30–50% lower memory. [ ] Pending
- E4 Micro‑batch decoder: N=4 chunks at B=16, L=50 → expect no OOM, similar loss. [ ] Pending
- E5 Memory profiling: Log `torch.cuda.memory_summary()` every 500 steps to confirm peak site. [ ] Pending

Please include: GPU model, driver, CUDA, PyTorch version, cudnn settings.

## Implementation Notes (if we proceed)
- AMP insertion points:
  - Begin autocast before: encoder forward → `rssm.observe_sequence(...)` → decoder/reward heads.
  - Keep KL divergence calculation out of autocast or cast back to float32 for mean.
  - Use `scaler.scale(total_loss).backward()`, then `scaler.step(optimizer)` and `scaler.update()`.
- Micro‑batching sketch:
  - For `latent_flat` and `target` at `src/workflows/world_models/dreamer.py:408`, iterate over chunks, compute per‑chunk recon loss, sum, divide by `num_chunks` for mean.
  - Ensure gradients flow; call backward once on the aggregated loss.

## Data We’d Like From You (other LLMs / operators)
- Confirm exact experiment config used (paste resolved Hydra config deltas for buffer, encoder/decoder, training).
- PyTorch/CUDA versions and GPU free/total memory at start.
- If you can, run E1/E2 quickly; attach step count reached and whether OOM occurred.
- If you implement AMP or micro‑batch locally, note any numerical issues (loss NaNs, KL instability) and throughput impact.

## Decision Log
- 2025‑11‑09: Codex suggests prioritizing config shrink, then AMP, then micro‑batch decoder; agrees with `problem.md` analysis that this is a compute‑graph peak, not a buffer issue.

## Open Questions
- Target headroom: What margin (GB) do we want to maintain for long runs to avoid fragmentation failures?
- Are there concurrent GPU consumers (logging video, eval) near the 53k mark? The report says no; please reconfirm.
- Any preference between AMP vs. micro‑batch first, based on infra constraints?

## How To Contribute
- Append under “Quick Experiments” with your name/agent and results.
- If proposing code changes, reference files with exact lines (e.g., `src/workflows/world_models/dreamer.py:366`).
- Keep entries short; link to logs under `experiments/…` where applicable.

## Updates (2025‑11‑09)
Context: Read `conversations/conv_claude_analysis.md` and re‑checked code paths.

Truth we converge on:
- The OOM originates from activation memory during world‑model backward with image reconstructions; buffer V2 isn’t the GPU culprit.
- AMP is a high‑ROI mitigation that preserves B and L, typically stable for Dreamer if we keep KL accumulation in fp32.
- Decoder micro‑batching is a safe additional guard if AMP alone isn’t enough.

Refined recommendation order (Codex):
1) Enable AMP for the world‑model update first (encoder → RSSM → decoder/reward under autocast; KL reduce in fp32; use `GradScaler`).
2) If still tight on memory or AMP isn’t available, enable decoder micro‑batching for reconstruction (accumulate mean loss, single backward).
3) Use config shrink (B=8 or L=32) as immediate validation or fallback, but not as the preferred long‑term fix.

Clarifications:
- AMP reduces activation memory; params typically remain fp32. Saying “backward in float16” is shorthand—GradScaler manages scaling to keep numerics stable.
- We should gate AMP and micro‑batch via config flags to avoid surprising existing runs.

Actionable code pointers (unchanged):
- Insert AMP/micro‑batch in `src/workflows/world_models/dreamer.py:366–443` (decoder forward at :408, backward at :429).

## Updates (2025‑11‑09, addendum)
Read user’s allocator analysis (fragmentation vs leak) and reconciled with code.

Assessment:
- Most plausible: CUDA allocator fragmentation + variable peak, not a true leak. `update_world_model()` uses local tensors and doesn’t retain graphs across steps; nothing obvious is kept alive after the function returns.
- The “allocated vs reserved” explanation matches PyTorch behavior: small steady `memory_allocated()` between steps with growing `memory_reserved()` is classic fragmentation/caching.

Diagnostics to disambiguate (minimal, targeted):
- Log every N steps: `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`; also `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()` to track peaks.
- Periodically dump `torch.cuda.memory_summary()` (or `memory_stats()`), looking at `active_bytes.*` and `segment` stats; optionally take a `torch.cuda.memory_snapshot()` once before/after OOM for post‑mortem.
- A/B test `torch.cuda.empty_cache()` every K steps: leak won’t improve; fragmentation delays/crash moves.

Allocator knobs worth testing (env):
- `PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"` to reduce over‑splitting and trigger GC earlier.
- If on recent PyTorch, try `expandable_segments:True` to reduce fragmentation across streams.

Low‑risk code tweaks that can help allocator behavior:
- Use `optimizer.zero_grad(set_to_none=True)` to avoid writing into existing grad buffers and reduce realloc patterns.
- Keep cuDNN algorithm selection predictable (e.g., consider `torch.backends.cudnn.benchmark=False`) to limit workspace variability; trade‑off is throughput.

Bottom line:
- I align with Fragmentation as primary cause. Action priority stays: AMP → (optional) decoder micro‑batching → config shrink. Add allocator/config diagnostics as above to produce hard evidence.

Root-level retention fix applied:
- Cleared `last_batch` after each world‑model update in `src/orchestration/world_model_orchestrator.py` so a large GPU batch isn’t kept alive between steps when not needed. Controller updates already resample if `last_batch` is `None`, so behavior remains correct while reducing persistent GPU residency.
