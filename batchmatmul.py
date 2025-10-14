import tvm
import argparse
from tvm import meta_schedule as ms
from tvm import te
from tvm.meta_schedule.runner.config import EvaluatorConfig
from tvm.script import tir as T
from typing import Tuple
from tvm.meta_schedule.testing import te_workload
from tvm.te import create_prim_func
import os
from pathlib import Path


#target = tvm.target.Target(f"cuda -max_threads_per_block 1024 -max_shared_memory_per_block 49152") # 3090
target = tvm.target.Target({"kind": "cuda", "arch": "sm_86", "max_threads_per_block": 1024, "max_shared_memory_per_block": 49152}) # 3090
#target = tvm.target.Target({"kind": "cuda", "arch": "sm_70", "max_threads_per_block": 1024, "max_shared_memory_per_block": 49152}) # V100

FILE_RUNSECS = "run_secs"

def batch_matmul_mkkn(  # pylint: disable=invalid-name,missing-docstring
    B: int,
    M: int,
    N: int,
    K: int,
    in_dtype: str = "float32",
    out_dtype: str = "float32",
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    x = te.placeholder((B, M, K), name="X", dtype=in_dtype)
    y = te.placeholder((B, K, N), name="Y", dtype=in_dtype)
    k = te.reduce_axis((0, K), name="k")
    z = te.compute(  # pylint: disable=invalid-name
        (B, M, N),
        lambda b, i, j: te.sum(
            x[b][i][k].astype(out_dtype) * y[b][k][j].astype(out_dtype),
            axis=[k],
        ),
        name="Z",
    )
    return (x, y, z)


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

    print("Batch Matmul")
    print(f"Batch size: {args.batchsize}, M: {args.M}, N: {args.N}, K: {args.K}")

    bmm = create_prim_func(batch_matmul_mkkn(args.batchsize, args.M, args.K, args.N, in_dtype="float32", out_dtype="float32"))
    print(bmm)

    database = ms.tune_tir(
        mod=bmm,
        target=target,
        max_trials_global=1000, # 1000
        num_trials_per_iter=64, # 64
        work_dir="./",
        runner=ms.runner.LocalRunner(
            evaluator_config=EvaluatorConfig(
                number=10, # 10
                enable_cpu_cache_flush=False,
            )
        ),
        cost_model=ms.cost_model.XGBModel(
            extractor=ms.feature_extractor.PerStoreFeature(),
            adaptive_training=False,
        ),
        strategy=ms.search_strategy.EvolutionarySearch(),
    )
    
    tune_record_list = database.get_all_tuning_records()
    workload = tune_record_list[0].workload
    mod = workload.mod

    top10 = database.get_top_k(workload, 10)
    sodir = "./sodir"
    os.makedirs(sodir, exist_ok=True)
    Path(FILE_RUNSECS).write_text("")
    with open(FILE_RUNSECS, "a") as f:
        for i in range(10):
            db = ms.database.MemoryDatabase()
            db.commit_workload(mod)
            db.commit_tuning_record(top10[i])
            sch = ms.tir_integration.compile_tir(db, mod, target)
            print(f"top{i+1}")
            print(sch.mod)
            with tvm.transform.PassContext(config={"tir.disable_assert": True}):
                lib = tvm.tir.build(sch.mod, target)
            
            sopath = sodir + f"/top{i+1}.so"
            lib.export_library(sopath)
            seconds = top10[i].run_secs[0].value
            f.write(f"top{i+1}\t{seconds}\t{seconds*1e9}\n")


if __name__ == "__main__":
    main()

