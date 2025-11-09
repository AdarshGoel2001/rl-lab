"""
Run the Atari experiment with memory monitoring to detect leaks.
"""
import subprocess
import threading
import time
import sys

def monitor_gpu():
    """Poll GPU memory every 10 seconds and print to stderr"""
    try:
        import torch
        if not torch.cuda.is_available():
            return

        with open("/tmp/gpu_memory_log.txt", "w") as f:
            f.write("Step,Allocated_MB,Reserved_MB,MaxAllocated_MB\n")

        step = 0
        while True:
            time.sleep(10)
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            max_allocated = torch.cuda.max_memory_allocated() / 1024**2

            with open("/tmp/gpu_memory_log.txt", "a") as f:
                f.write(f"{step},{allocated:.1f},{reserved:.1f},{max_allocated:.1f}\n")

            print(f"[MEM {step*10}s] Alloc: {allocated:.1f}MB, Reserved: {reserved:.1f}MB, Peak: {max_allocated:.1f}MB",
                  file=sys.stderr, flush=True)
            step += 1
    except Exception as e:
        print(f"Monitor error: {e}", file=sys.stderr)

# Start monitor thread
monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
monitor_thread.start()

# Run training
print("Starting training with memory monitoring...")
print("Memory log will be written to: /tmp/gpu_memory_log.txt")
print()

result = subprocess.run([
    "python", "scripts/train.py",
    "experiment=dreamer_ataripong"
], cwd="/home/omkar/rl-lab")

sys.exit(result.returncode)
