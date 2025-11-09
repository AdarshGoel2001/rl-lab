# Dreamer Atari Pong CUDA OOM - Factual Report

## Observed Behavior

Training Dreamer on Atari Pong crashes with CUDA out-of-memory error at approximately 53,000-54,000 steps. Crash is reproducible across multiple runs with different configurations.

## Hardware/Environment

- GPU: NVIDIA GeForce RTX 3060 Laptop (6 GB total memory)
- Framework: PyTorch with CUDA
- OS: Linux 5.15.167.4-microsoft-standard-WSL2

## Crash Details

### Error Message
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 174.00 MiB.
GPU 0 has a total capacity of 6.00 GiB of which 0 bytes is free.
Including non-PyTorch memory, this process has 17179869184.00 GiB memory in use.
Of the allocated memory 12.15 GiB is allocated by PyTorch, and 319.03 MiB is
reserved by PyTorch but unallocated.
```

### Crash Location
- Occurs during backward pass in world model training
- Stack trace points to `torch.autograd.backward()` in `src/workflows/world_models/dreamer.py`

## Reproduction Data

### Run 1 (Initial)
- Config: 100k warmup steps, buffer capacity 50k
- Crash step: ~53,000
- Phase: warmup (world model only, no controller)

### Run 2 (Rerun same config)
- Config: 100k warmup steps, buffer capacity 50k
- Crash step: 53,800
- Phase: warmup (world model only, no controller)

### Run 3 (Reduced warmup)
- Config: 25k warmup steps, buffer capacity 50k
- Crash step: 53,800
- Phase: joint_training (world model + controller)
- Successfully transitioned from warmup to joint training at 25k
- Successfully passed buffer capacity threshold at 50k
- Crashed 3,800 steps after buffer reached capacity

## Configuration Constants Across All Runs

```yaml
buffer:
  capacity: 50000
  batch_size: 16
  sequence_length: 50
  num_envs: 2

environment:
  name: Atari Pong
  frame_stack: 4
  observation_shape: [84, 84, 4]
  action_space: Discrete(6)

model:
  encoder: NatureCNN (channels: [32, 64, 64]) → 512 features
  decoder: AtariConvDecoder (base_channels: 32, mid_channels: 16)
  rssm:
    deterministic_dim: 200
    stochastic_dim: 32
```

## Observed Memory Reporting Anomaly

PyTorch reports allocating **12.15 GiB** on a **6.00 GiB** GPU. This is physically impossible and indicates corrupted memory accounting or reporting.

## Tested Variables (No Effect on Crash)

- ✗ Warmup duration: Tested 25k and 100k steps - both crash at ~53.8k
- ✗ Training phase: Crashes in both warmup-only and joint_training phases
- ✗ Buffer capacity: Set at 50k, crash occurs at 53.8k (3.8k steps later)

## Consistent Patterns

1. **Crash step**: 53,000-53,800 steps (±800 step variance)
2. **Allocation size**: Always requests 174 MB at failure point
3. **Buffer relationship**: Crash occurs ~3,800 steps after buffer reaches capacity
4. **Deterministic**: Reproduced identically across 3 independent runs

## Memory Accumulation Evidence

Run 3 process memory usage over time:
- Start: Unknown
- Step 12,400 (warmup): 2.4 GB RAM (30.8%)
- Step 44,300 (joint training): 4.1 GB RAM (52.0%)
- Step 53,800 (crash): Process terminated

RAM usage increased ~1.7 GB between steps 12k-44k (~53 MB per 1000 steps).

## Code Locations

- World model training: `src/workflows/world_models/dreamer.py:366-433`
- Buffer implementation: `src/buffers/episode_replay.py`
- Orchestrator: `src/orchestration/world_model_orchestrator.py`

## Files

- Latest crash log: `/tmp/dreamer_25k.log`
- Config used: `configs/experiment/dreamer_ataripong.yaml`
- Previous run logs: `experiments/dreamer_ataripong-20251108_014609/`
