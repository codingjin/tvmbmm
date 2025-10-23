import tvm
import os, sys
import numpy as np
import torch
import argparse
from zeus.monitor import ZeusMonitor

NUM_WARMUP = 3
NUM_RUN = 100
NUM_MEASURE = 1000
monitor = ZeusMonitor(approx_instant_energy=False)

np.random.seed(137)

def main():
    parser = argparse.ArgumentParser(description="Batch Matrix-multiplication")
    
    parser.add_argument(
        "--batchsize", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--M", type=int, default=4096,
        help="Matrix dimension M (default: 4096)"
    )
    parser.add_argument(
        "--N", type=int, default=4096,
        help="Matrix dimension N (default: 4096)"
    )
    parser.add_argument(
        "--K", type=int, default=4096,
        help="Matrix dimension K (default: 4096)"
    )

    args = parser.parse_args()
    batchsize, M, N, K = args.batchsize, args.M, args.N, args.K

    # warmup
    print("warmup")
    for i in range(NUM_WARMUP):
        a_np = np.random.uniform(size=(batchsize, M, K)).astype("float16")
        b_np = np.random.uniform(size=(batchsize, K, N)).astype("float16")
        a_torch = torch.tensor(a_np, device="cuda", dtype=torch.float16)
        b_torch = torch.tensor(b_np, device="cuda", dtype=torch.float16)
        torch.bmm(a_torch, b_torch)
        monitor.begin_window("run")
        for j in range(100):
            torch.bmm(a_torch, b_torch)
        energy = monitor.end_window("run")
        #print(energy)
        print(f"Energy: {energy.total_energy} J")
    print("warmup done")
    
    # run
    results = []
    for i in range(NUM_RUN):
        a_np = np.random.uniform(size=(batchsize, M, K)).astype("float16")
        b_np = np.random.uniform(size=(batchsize, K, N)).astype("float16")
        a_torch = torch.tensor(a_np, device="cuda", dtype=torch.float16)
        b_torch = torch.tensor(b_np, device="cuda", dtype=torch.float16)
        torch.bmm(a_torch, b_torch)
        monitor.begin_window("run")
        for j in range(NUM_MEASURE):
            torch.bmm(a_torch, b_torch)
        energy = monitor.end_window("run")
        #print(energy)
        results.append(energy.total_energy)
        
    mean, std = np.mean(results), np.std(results)
    print(f"BMM torch batchsize={batchsize} M={M} N={N} K={K}")
    print(f"Mean energy consumption(x{NUM_MEASURE}): {mean} J, std: {std}")

    
    a_np = np.random.uniform(size=(batchsize, M, K)).astype("float16")
    b_np = np.random.uniform(size=(batchsize, K, N)).astype("float16")
    a_torch = torch.tensor(a_np, device="cuda", dtype=torch.float16)
    b_torch = torch.tensor(b_np, device="cuda", dtype=torch.float16)
    # Warm-up
    torch.bmm(a_torch, b_torch)
    torch.cuda.synchronize()

    # CUDA event-based timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(NUM_MEASURE):
        torch.bmm(a_torch, b_torch)
    end_event.record()

    # Wait for all kernels to finish
    torch.cuda.synchronize()

    timems = start_event.elapsed_time(end_event) / (1.0 * NUM_MEASURE)

    
    flops = 2 * batchsize * M * N * K
    gflops = flops * 1.0e-6 / timems if timems != 0 else float("inf")
    print(f"{gflops} GFLOPs")
    print("Complete!")
    print("----------------------------------------------\n")

if __name__ == "__main__":
    main()