import tvm
import os, sys
import numpy as np
import torch
import argparse
import torch



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

    a_np = np.random.uniform(size=(batchsize, M, K)).astype("float16")
    b_np = np.random.uniform(size=(batchsize, K, N)).astype("float16")

    # Pytorch
    a_torch = torch.tensor(a_np, device="cuda", dtype=torch.float16)
    b_torch = torch.tensor(b_np, device="cuda", dtype=torch.float16)
    c_torch = torch.bmm(a_torch, b_torch)
    #print(c_torch)
    print("pass0")

    # TVM
    a_nd = tvm.runtime.tensor(a_np, device=tvm.cuda())
    b_nd = tvm.runtime.tensor(b_np, device=tvm.cuda())
    c_nd = tvm.runtime.tensor(np.zeros((batchsize, M, N), dtype="float16"), device=tvm.cuda())

    f = tvm.runtime.load_module("./sodir/top10.so")
    f(a_nd, b_nd, c_nd)
    np.testing.assert_allclose(c_nd.numpy(), c_torch.cpu().numpy(), rtol=1e-1)
    #np.testing.assert_allclose(c_nd.numpy(), c_torch.cpu().numpy(), rtol=1e-10)
    print("pass10")


    f = tvm.runtime.load_module("./sodir/top1.so")
    f(a_nd, b_nd, c_nd)
    np.testing.assert_allclose(c_nd.numpy(), c_torch.cpu().numpy(), rtol=1e-1)
    #np.testing.assert_allclose(c_nd.numpy(), c_torch.cpu().numpy(), rtol=1e-10)
    print("pass1")

if __name__ == "__main__":
    main()



