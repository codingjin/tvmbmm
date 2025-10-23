import tvm
import os, sys
import numpy as np
import torch
import argparse
import pynvml

NUM_WARMUP = 3
NUM_RUN = 100
NUM_MEASURE = 1000


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

    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    handles, start_energy, consumed_energy = [], [], []
    for i in range(deviceCount):
        handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
        start_energy.append(0)
        consumed_energy.append(0)

    # warmup
    print("warmup")
    for i in range(NUM_WARMUP):
        a_np = np.random.uniform(size=(batchsize, M, K)).astype("float16")
        b_np = np.random.uniform(size=(batchsize, K, N)).astype("float16")
        a_torch = torch.tensor(a_np, device="cuda", dtype=torch.float16)
        b_torch = torch.tensor(b_np, device="cuda", dtype=torch.float16)
        torch.bmm(a_torch, b_torch)

        for k in range(deviceCount):
            start_energy[k] = pynvml.nvmlDeviceGetTotalEnergyConsumption(handles[k])
        
        for j in range(NUM_MEASURE):
            torch.bmm(a_torch, b_torch)
        
        torch.cuda.synchronize()
        for k in range(deviceCount):
            consumed_energy[k] = pynvml.nvmlDeviceGetTotalEnergyConsumption(handles[k]) - start_energy[k]
        total_energy = sum(consumed_energy) * 0.001
        print(f"Energy: {total_energy} J")
    print("warmup done")
    
    # run
    results = []
    for i in range(NUM_RUN):
        a_np = np.random.uniform(size=(batchsize, M, K)).astype("float16")
        b_np = np.random.uniform(size=(batchsize, K, N)).astype("float16")
        a_torch = torch.tensor(a_np, device="cuda", dtype=torch.float16)
        b_torch = torch.tensor(b_np, device="cuda", dtype=torch.float16)
        torch.bmm(a_torch, b_torch)
        
        for k in range(deviceCount):
            start_energy[k] = pynvml.nvmlDeviceGetTotalEnergyConsumption(handles[k])

        for j in range(NUM_MEASURE):
            torch.bmm(a_torch, b_torch)
        
        torch.cuda.synchronize()
        for k in range(deviceCount):
            consumed_energy[k] = pynvml.nvmlDeviceGetTotalEnergyConsumption(handles[k]) - start_energy[k]
        total_energy = sum(consumed_energy) * 0.001
        results.append(total_energy)
        
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
    for _ in range(100):
        torch.bmm(a_torch, b_torch)
    end_event.record()

    # Wait for all kernels to finish
    torch.cuda.synchronize()

    timems = start_event.elapsed_time(end_event) / 100.0

    
    flops = 2 * batchsize * M * N * K
    gflops = flops * 1.0e-6 / timems if timems != 0 else float("inf")
    print(f"{gflops} GFLOPs")
    print("Complete!")
    print("----------------------------------------------\n")

if __name__ == "__main__":
    main()