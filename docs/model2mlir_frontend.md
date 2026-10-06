# Source-anchored model2MLIR frontend

The reproducible PyTorch frontend for PyTorch-expressible Radiance kernels is
model2MLIR `main` at `1224a05f66bee0e296f2cccc292e7d0a70b74bf6`
(verified as the latest upstream `main` on 2026-10-06). The active local
model2MLIR development worktree has unrelated edits; the captures use a clean
isolated checkout of that exact commit. Scripts require an explicit
`--model2mlir-root` and record a hash of every Python source in `m2m/`.
The reference is the local `radiance-kernels` revision
`a27f6abd24830fdc7999d872d170ab778f1e662e`.

| Frontend capture | Source check before capture | Captured result |
| --- | --- | --- |
| STREAM Copy, Scale, Add, Triad | Every FP32 result for 1,048,576 elements per operation matches `kernels/stream/run.py`; FNV digests match the source formulas | Four `linalg`/tensor modules, zero opaque calls, parsed by all three OOT tools |
| SIMT GEMM | All 4,096 BF16 output words match the `kernels/gemm_simt/gemm.py` generated golden | `linalg.matmul`, zero opaque calls, parsed by all three OOT tools |
| MX FP8 GEMM | 64×64×64 shape and operator selected from `kernels/gemm_mxgemmini/gen_mxgemm_data.py` | One `functional:matmul` site selected by the MX adapter; handoff parsed by MX and Radiance tools |

Run the two scripts in this repository and the MX script in `mx-gemmini-mlir`
with the same clean model2MLIR checkout, its Python environment, the pinned
source checkout, and built `muon-opt`, `mx-gemmini-opt`, and `radiance-opt`.
The scripts take explicit tool paths and write MLIR, parse logs, and hashed
receipts under `--out`. Small result snapshots are in `evidence/` here and
`docs/evidence/` in MX. The scripts fail if the source Git revision changes,
the PyTorch model differs from the source output, a frontend op stays opaque,
or a dialect parser rejects the captured IR.

For the MX case, the model2MLIR external quantization API uses the MX OOT
adapter and the `microscaling-quant` operand capture package. The PyTorch
graph establishes contraction identity and target selection. The handwritten
FP8 code and E8M0 scale blobs are the numerical oracle; ideal PyTorch matmul
is not a substitute for the source `mx_golden`. The current MX capture does
not yet bind source blobs or compare accelerator results.

These captures are the frontend inputs for target lowering, not target
programs. The next Muon compiler step is to bufferize the generated
`linalg`/tensor IR, distribute parallel loops onto Muon launches, and emit
native RV32 objects using a toolchain that supports the required stack
stride. The next MX step is to lower selected handoff sites into actual MX
operand loads, commands, waits, and readout with complete source-golden
comparison. Radiance then combines the two executable paths and checks
cross-engine synchronization under an artifact-bound SoC profile. The
existing handwritten LLVM-dialect import is only a reference baseline.
