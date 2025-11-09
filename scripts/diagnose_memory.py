"""
Simple diagnostic to check if the OOM is from fragmentation or a memory leak.

This simulates the training loop and logs memory usage to show what's happening.
"""

import torch
import torch.nn as nn
import time

print("=" * 60)
print("MEMORY DIAGNOSTIC TEST")
print("=" * 60)
print()

# Check if CUDA is available
if not torch.cuda.is_available():
    print("❌ CUDA not available. This test requires a GPU.")
    exit(1)

device = torch.device("cuda")
print(f"✅ Using device: {torch.cuda.get_device_name(0)}")
print(f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print()

# Simulate the training workload
# B=16, L=50, 84x84x4 images
batch_size = 16
sequence_length = 50
img_size = 84

print(f"Simulating Dreamer workload:")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {sequence_length}")
print(f"  Image size: {img_size}x{img_size}x4")
print(f"  Total frames per batch: {batch_size * sequence_length}")
print()

# Simple model to simulate encoder -> decoder
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512)
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 64 * 7 * 7),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, 3, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2), nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 8, 4),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = SimpleModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Running simulated training steps...")
print()
print("Step | Allocated (MB) | Reserved (MB) | Peak (MB) | Fragmentation")
print("-" * 75)

def get_memory_stats():
    """Get current memory statistics"""
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    return allocated, reserved, max_allocated

# Run multiple steps
num_steps = 100
memory_log = []

for step in range(num_steps):
    # Create batch (B*L, C, H, W)
    batch = torch.randn(
        batch_size * sequence_length,
        4, img_size, img_size,
        device=device
    )

    # Forward pass
    output = model(batch)
    loss = torch.mean((output - batch) ** 2)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Clean up batch
    del batch, output, loss

    # Get memory stats AFTER cleanup
    allocated, reserved, peak = get_memory_stats()
    fragmentation = reserved - allocated
    memory_log.append((step, allocated, reserved, peak, fragmentation))

    # Print every 10 steps
    if step % 10 == 0 or step < 5:
        print(f"{step:4d} | {allocated:13.1f} | {reserved:12.1f} | {peak:8.1f} | {fragmentation:13.1f}")

print("-" * 75)
print()

# Analysis
print("=" * 60)
print("ANALYSIS")
print("=" * 60)
print()

# Check if allocated memory grows (leak)
first_5_allocated = [log[1] for log in memory_log[:5]]
last_5_allocated = [log[1] for log in memory_log[-5:]]

avg_first = sum(first_5_allocated) / len(first_5_allocated)
avg_last = sum(last_5_allocated) / len(last_5_allocated)

print(f"Average allocated memory:")
print(f"  First 5 steps:  {avg_first:.1f} MB")
print(f"  Last 5 steps:   {avg_last:.1f} MB")
print(f"  Change:         {avg_last - avg_first:+.1f} MB")
print()

if abs(avg_last - avg_first) > 10:
    print("⚠️  MEMORY LEAK DETECTED")
    print("   Allocated memory is growing between training steps!")
    print("   This suggests something isn't being freed properly.")
else:
    print("✅ NO MEMORY LEAK")
    print("   Allocated memory stays constant between steps.")
    print("   Memory IS being freed properly after each step.")

print()

# Check if reserved memory grows (fragmentation)
first_reserved = memory_log[0][2]
last_reserved = memory_log[-1][2]

print(f"Reserved memory (CUDA cache):")
print(f"  Start:  {first_reserved:.1f} MB")
print(f"  End:    {last_reserved:.1f} MB")
print(f"  Growth: {last_reserved - first_reserved:+.1f} MB")
print()

if last_reserved - first_reserved > 100:
    print("⚠️  CACHE GROWTH DETECTED")
    print("   PyTorch's CUDA cache is growing over time.")
    print("   This is FRAGMENTATION - the freed memory is being")
    print("   kept in increasingly fragmented blocks.")
else:
    print("✅ STABLE CACHE")
    print("   CUDA cache size is stable.")

print()

# Check fragmentation
avg_fragmentation_first = sum([log[4] for log in memory_log[:5]]) / 5
avg_fragmentation_last = sum([log[4] for log in memory_log[-5:]]) / 5

print(f"Fragmentation (Reserved - Allocated):")
print(f"  First 5 steps:  {avg_fragmentation_first:.1f} MB")
print(f"  Last 5 steps:   {avg_fragmentation_last:.1f} MB")
print(f"  Increase:       {avg_fragmentation_last - avg_fragmentation_first:+.1f} MB")
print()

# Final verdict
print("=" * 60)
print("CONCLUSION")
print("=" * 60)
print()

if abs(avg_last - avg_first) > 10:
    print("🔴 PRIMARY ISSUE: Memory leak")
    print("   Something in the code is not releasing memory.")
    print("   Need to investigate what's holding references.")
else:
    print("🟡 PRIMARY ISSUE: Memory fragmentation")
    print("   Memory IS being freed, but the CUDA allocator")
    print("   cache becomes fragmented over time.")
    print()
    print("   Why it fails after many steps:")
    print("   - Each step frees memory back to cache")
    print("   - Cache holds onto it in fragmented blocks")
    print("   - Eventually can't find contiguous block")
    print("   - Tries to allocate new memory → GPU full → CRASH")

print()
print("=" * 60)
print()

# Cleanup
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
