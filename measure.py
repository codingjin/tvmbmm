import tvm
import os, sys
import numpy as np
import torch
import argparse
from zeus.monitor import ZeusMonitor

NUM_WARMUP = 3
NUM_RUN = 100
NUM_MEASURE = 100
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
    parser.add_argument(
        "--I", type=int, default=0,
        help="Index, top{index+1} {default: 0}"
    )

    args = parser.parse_args()
    batchsize, M, N, K, I = args.batchsize, args.M, args.N, args.K, args.I
    sopath = f"./sodir/top{I+1}.so"

    func = tvm.runtime.load_module(sopath)

    # warmup
    print("warmup")
    for i in range(NUM_WARMUP):
        a_np = np.random.uniform(size=(batchsize, M, K)).astype("float16")
        b_np = np.random.uniform(size=(batchsize, K, N)).astype("float16")

        a_nd = tvm.runtime.tensor(a_np, device=tvm.cuda())
        b_nd = tvm.runtime.tensor(b_np, device=tvm.cuda())
        c_nd = tvm.runtime.tensor(np.zeros((batchsize, M, N), dtype="float16"), device=tvm.cuda())
        monitor.begin_window("run")
        for j in range(NUM_MEASURE):
            func(a_nd, b_nd, c_nd)
        energy = monitor.end_window("run")
        #print(energy)
        print(f"Energy: {energy.total_energy} J")
    print("warmup done\n")
    
    # run
    results = []
    for i in range(NUM_RUN):
        a_np = np.random.uniform(size=(batchsize, M, K)).astype("float16")
        b_np = np.random.uniform(size=(batchsize, K, N)).astype("float16")

        a_nd = tvm.runtime.tensor(a_np, device=tvm.device('cuda', 0))
        b_nd = tvm.runtime.tensor(b_np, device=tvm.device('cuda', 0))
        c_nd = tvm.runtime.tensor(np.zeros((batchsize, M, N), dtype="float16"), device=tvm.device('cuda', 0))

        #print(f"run {i}")
        monitor.begin_window("run")
        for j in range(NUM_MEASURE):
            func(a_nd, b_nd, c_nd)
        energy = monitor.end_window("run")
        #print(energy)
        results.append(energy.total_energy)
        

        """
        monitor.begin_window("run")
        for j in range(NUM_MEASURE):
            func(a_nd, b_nd, c_nd)
        energy = monitor.end_window("run")
        print(energy)
        print(energy.gpu_energy[0])
        #print(f"Energy: {energy.total_energy} J")
        """
    mean, std = np.mean(results), np.std(results)
    print(f"BMM I={I} batchsize={batchsize} M={M} N={N} K={K}")
    print(f"Mean energy consumption: {mean} J, std: {std}")
    print("Complete!")

if __name__ == "__main__":
    main()