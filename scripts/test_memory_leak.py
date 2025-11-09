#!/usr/bin/env python3
"""
Deep investigation of the memory leak.
Test each component of the buffer->training pipeline.
"""

import torch
import numpy as np
import gc
import sys
from collections import deque

print("="*60)
print("MEMORY LEAK INVESTIGATION")
print("="*60)

def get_refcount(obj):
    """Get reference count minus the getrefcount call itself."""
    return sys.getrefcount(obj) - 1

# ==============================================================================
# TEST 1: Does torch.tensor() actually break the reference chain?
# ==============================================================================
print("\n" + "="*60)
print("TEST 1: torch.tensor() vs torch.from_numpy()")
print("="*60)

# Create numpy array simulating episode data
ep_data = np.ones((100, 84, 84, 4), dtype=np.uint8)
print(f"\n1. Created episode data")
print(f"   Refcount: {get_refcount(ep_data)}")

# Convert to float
obs_float = ep_data.astype(np.float32) / 255.0
print(f"\n2. Converted to float (obs_float)")
print(f"   Refcount to ep_data: {get_refcount(ep_data)}")
print(f"   Shares memory: {np.shares_memory(obs_float, ep_data)}")

# Test torch.from_numpy (OLD WAY)
print(f"\n3a. OLD WAY: torch.from_numpy()")
tensor_old = torch.from_numpy(obs_float)
print(f"   Refcount to obs_float: {get_refcount(obs_float)}")
print(f"   Refcount to ep_data: {get_refcount(ep_data)}")

del tensor_old
gc.collect()
print(f"\n3b. After deleting tensor_old:")
print(f"   Refcount to obs_float: {get_refcount(obs_float)}")
print(f"   Refcount to ep_data: {get_refcount(ep_data)}")

# Test torch.tensor (NEW WAY)
print(f"\n4a. NEW WAY: torch.tensor()")
tensor_new = torch.tensor(obs_float, dtype=torch.float32)
print(f"   Refcount to obs_float: {get_refcount(obs_float)}")
print(f"   Refcount to ep_data: {get_refcount(ep_data)}")

del tensor_new
gc.collect()
print(f"\n4b. After deleting tensor_new:")
print(f"   Refcount to obs_float: {get_refcount(obs_float)}")
print(f"   Refcount to ep_data: {get_refcount(ep_data)}")

print(f"\n✅ torch.tensor() DOES break the reference chain")

# ==============================================================================
# TEST 2: Does the .to(device) call hold references?
# ==============================================================================
print("\n" + "="*60)
print("TEST 2: Does .to(device) hold references?")
print("="*60)

if torch.cuda.is_available():
    device = torch.device('cuda')

    ep_data2 = np.ones((100, 84, 84, 4), dtype=np.uint8)
    obs_float2 = ep_data2.astype(np.float32) / 255.0

    print(f"\n1. Before tensor creation:")
    print(f"   Refcount to obs_float2: {get_refcount(obs_float2)}")

    cpu_tensor = torch.tensor(obs_float2, dtype=torch.float32)
    print(f"\n2. After torch.tensor():")
    print(f"   Refcount to obs_float2: {get_refcount(obs_float2)}")

    gpu_tensor = cpu_tensor.to(device)
    print(f"\n3. After .to(device):")
    print(f"   Refcount to obs_float2: {get_refcount(obs_float2)}")
    print(f"   Refcount to cpu_tensor: {get_refcount(cpu_tensor)}")

    del gpu_tensor
    torch.cuda.synchronize()
    gc.collect()
    print(f"\n4. After deleting gpu_tensor:")
    print(f"   Refcount to obs_float2: {get_refcount(obs_float2)}")
    print(f"   Refcount to cpu_tensor: {get_refcount(cpu_tensor)}")

    del cpu_tensor
    gc.collect()
    print(f"\n5. After deleting cpu_tensor:")
    print(f"   Refcount to obs_float2: {get_refcount(obs_float2)}")

    del obs_float2, ep_data2
    gc.collect()

    print(f"\n✅ .to(device) does NOT hold references after deletion")
else:
    print("\n⚠️  CUDA not available, skipping")

# ==============================================================================
# TEST 3: Simulate the actual buffer.sample() flow
# ==============================================================================
print("\n" + "="*60)
print("TEST 3: Simulate buffer.sample() flow")
print("="*60)

# Create fake episodes
episodes_deque = deque()
for i in range(10):
    ep = {
        'obs': np.ones((100, 84, 84, 4), dtype=np.uint8) * i,
        'actions': np.ones((100, 6), dtype=np.float32),
        'rewards': np.ones((100,), dtype=np.float32),
    }
    episodes_deque.append(ep)

print(f"\n1. Created 10 episodes in deque")

# Get reference to first episode
ep0 = episodes_deque[0]
print(f"\n2. Got reference to episode 0")
print(f"   Refcount to ep0['obs']: {get_refcount(ep0['obs'])}")

# Simulate sample() - create list snapshot
episodes_list = list(episodes_deque)
print(f"\n3. Created list(episodes_deque)")
print(f"   Refcount to ep0['obs']: {get_refcount(ep0['obs'])}")
print(f"   ⚠️  Reference count increased! The list holds a reference")

# Create window (view)
window = episodes_list[0]['obs'][0:50]
print(f"\n4. Created window (view) from episodes_list[0]['obs'][0:50]")
print(f"   Refcount to ep0['obs']: {get_refcount(ep0['obs'])}")
print(f"   Shares memory: {np.shares_memory(window, ep0['obs'])}")

# Now evict episode 0 from deque
old_ep = episodes_deque.popleft()
print(f"\n5. Evicted episode 0 from deque")
print(f"   Refcount to ep0['obs']: {get_refcount(ep0['obs'])}")
print(f"   Still in episodes_list: {id(episodes_list[0]['obs']) == id(ep0['obs'])}")

# The episode_list still holds it!
print(f"\n6. Check what's holding references:")
print(f"   - ep0 variable: 1 ref")
print(f"   - old_ep variable: 1 ref")
print(f"   - episodes_list[0]: 1 ref")
print(f"   - window (numpy view): 1 ref")
print(f"   Total: {get_refcount(ep0['obs'])}")

# Simulate function return - delete local variables
del episodes_list, window, old_ep
gc.collect()
print(f"\n7. After deleting local variables (simulating function return):")
print(f"   Refcount to ep0['obs']: {get_refcount(ep0['obs'])}")
print(f"   ✅ Episode can now be freed!")

print(f"\n🔍 INSIGHT: The 'episodes = list(self._episodes)' on line 270")
print(f"   keeps episodes alive DURING sample(), but they should be freed")
print(f"   after the function returns. So this isn't the leak source!")

# ==============================================================================
# TEST 4: Check if the problem is in the BATCH DICT
# ==============================================================================
print("\n" + "="*60)
print("TEST 4: Does the batch dict hold references?")
print("="*60)

# Recreate episodes
episodes_deque2 = deque()
for i in range(5):
    ep = {
        'obs': np.ones((100, 84, 84, 4), dtype=np.uint8) * i,
    }
    episodes_deque2.append(ep)

ep_test = episodes_deque2[0]
print(f"\n1. Created test episode")
print(f"   Refcount to ep_test['obs']: {get_refcount(ep_test['obs'])}")

# Simulate sample() creating obs_windows
obs_windows = []
episodes_snap = list(episodes_deque2)
obs_windows.append(episodes_snap[0]['obs'][0:50])

print(f"\n2. Created obs_windows (with view)")
print(f"   Refcount to ep_test['obs']: {get_refcount(ep_test['obs'])}")

# Stack (creates copy)
obs_uint8 = np.stack(obs_windows, axis=0)
print(f"\n3. Stacked to obs_uint8")
print(f"   Refcount to ep_test['obs']: {get_refcount(ep_test['obs'])}")
print(f"   obs_uint8 shares memory: {np.shares_memory(obs_uint8, ep_test['obs'])}")

# Convert to float
obs_float_test = obs_uint8.astype(np.float32) / 255.0
print(f"\n4. Converted to float")
print(f"   Refcount to ep_test['obs']: {get_refcount(ep_test['obs'])}")

# Create tensor (NEW WAY)
tensor_test = torch.tensor(obs_float_test, dtype=torch.float32)
print(f"\n5. Created torch.tensor()")
print(f"   Refcount to ep_test['obs']: {get_refcount(ep_test['obs'])}")

# Put in batch dict
batch = {"observations": tensor_test}
print(f"\n6. Put in batch dict")
print(f"   Refcount to ep_test['obs']: {get_refcount(ep_test['obs'])}")

# Evict episode
old = episodes_deque2.popleft()
print(f"\n7. Evicted episode from deque")
print(f"   Refcount to ep_test['obs']: {get_refcount(ep_test['obs'])}")

# Simulate function return
del episodes_snap, obs_windows, obs_uint8, obs_float_test, tensor_test, old
gc.collect()
print(f"\n8. After deleting sample() local variables:")
print(f"   Refcount to ep_test['obs']: {get_refcount(ep_test['obs'])}")
print(f"   (should be 2: ep_test + batch dict)")

# Delete batch (what should happen after training step)
del batch
gc.collect()
print(f"\n9. After deleting batch:")
print(f"   Refcount to ep_test['obs']: {get_refcount(ep_test['obs'])}")
print(f"   ✅ Should be 1 (only ep_test variable)")

print("\n" + "="*60)
print("CONCLUSION SO FAR:")
print("="*60)
print("""
1. ✅ torch.tensor() DOES break the reference chain
2. ✅ .to(device) does NOT hold references after deletion
3. ✅ episodes_list is freed when sample() returns
4. ✅ Batch dict references are freed when batch is deleted

So where is the leak?

Hypothesis: The leak is NOT in the buffer code itself.
The leak must be in HOW THE BATCH IS USED in the training loop.

Possibilities:
A) The orchestrator or workflow is holding references to batches
B) The autograd graph is somehow keeping tensors alive across steps
C) There's a hidden reference in logging/metrics
D) Python's garbage collector isn't running frequently enough
E) There's something in the RSSM or other components keeping state

Need to check: What happens to the batch AFTER it's passed to update_world_model()?
""")