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
LAST_RUNSECS = "last_run_secs"
lastso = "./last.so"

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

    func = tvm.runtime.load_module(lastso)

    # warmup
    print("warmup")
    for i in range(NUM_WARMUP):
        a_np = np.random.uniform(size=(batchsize, M, K)).astype("float16")
        b_np = np.random.uniform(size=(batchsize, K, N)).astype("float16")

        a_nd = tvm.runtime.tensor(a_np, device=tvm.cuda())
        b_nd = tvm.runtime.tensor(b_np, device=tvm.cuda())
        c_nd = tvm.runtime.tensor(np.zeros((batchsize, M, N), dtype="float16"), device=tvm.cuda())
        func(a_nd, b_nd, c_nd)
        monitor.begin_window("run")
        for j in range(100):
            func(a_nd, b_nd, c_nd)
        energy = monitor.end_window("run")
        #print(energy)
        print(f"Energy: {energy.total_energy} J")
    print("warmup done")
    
    # run
    results = []
    for i in range(NUM_RUN):
        a_np = np.random.uniform(size=(batchsize, M, K)).astype("float16")
        b_np = np.random.uniform(size=(batchsize, K, N)).astype("float16")

        a_nd = tvm.runtime.tensor(a_np, device=tvm.cuda())
        b_nd = tvm.runtime.tensor(b_np, device=tvm.cuda())
        c_nd = tvm.runtime.tensor(np.zeros((batchsize, M, N), dtype="float16"), device=tvm.cuda())

        #print(f"run {i}")
        func(a_nd, b_nd, c_nd)
        monitor.begin_window("run")
        for j in range(NUM_MEASURE):
            func(a_nd, b_nd, c_nd)
        energy = monitor.end_window("run")
        #print(energy)
        results.append(energy.total_energy)
        
    mean, std = np.mean(results), np.std(results)
    print(f"BMM last batchsize={batchsize} M={M} N={N} K={K}")
    print(f"Mean energy consumption(x{NUM_MEASURE}): {mean} J, std: {std}")

    """
    if not os.path.exists(LAST_RUNSECS):
        print(f"Error: File '{LAST_RUNSECS}' does not exist.")
        sys.exit(1)

    with open(LAST_RUNSECS, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    timeline = lines[0]
    try:
        timens = float(timeline.split("\t")[-1])
    except ValueError:
        print(f"Cannot convert to float: {timens}")
        sys.exit(1)
    
    flops = 2 * batchsize * M * N * K
    gflops = flops / timens if timens != 0 else float("inf")
    print(f"{gflops} GFLOPs")
    print("Complete!")
    print("----------------------------------------------\n")
    """

if __name__ == "__main__":
    main()