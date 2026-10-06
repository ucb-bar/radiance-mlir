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
| STREAM Copy, Scale, Add, Triad | Every FP32 result for 1,048,576 elements per operation matches `kernels/stream/run.py`; FNV digests match the source formulas | Four tensor modules, zero opaque calls; upstream fusion and bufferization produce explicit parallel loops for Scale/Add/Triad; all stages parse in all three OOT tools |
| SIMT GEMM | All 4,096 BF16 output words match the `kernels/gemm_simt/gemm.py` generated golden | `linalg.matmul`, zero opaque calls, parsed by all three OOT tools; compiled host output matches all source golden words |
| MX FP8 GEMM | 64×64×64 shape and operator selected from `kernels/gemm_mxgemmini/gen_mxgemm_data.py` | One `functional:matmul` site selected by the MX adapter; handoff parsed by MX and Radiance tools |

Run the two scripts in this repository and the MX script in `mx-gemmini-mlir`
with the same clean model2MLIR checkout, its Python environment, the pinned
source checkout, and built `muon-opt`, `mx-gemmini-opt`, and `radiance-opt`.
The scripts take explicit tool paths and write MLIR, parse logs, and hashed
receipts under `--out`. Small result snapshots are in `evidence/` here and
`docs/evidence/` in MX. The scripts fail if the source Git revision changes,
the PyTorch model differs from the source output, a frontend op stays opaque,
or a dialect parser rejects the captured IR.

`tests/capture_model2mlir_stream.py` also requires upstream `mlir-opt`.
Its `linalg-fuse-elementwise-ops`, `canonicalize`, and `cse` pipeline turns
Triad into one `linalg.generic` loop with the source multiply and add. A
one-shot bufferization and `convert-linalg-to-parallel-loops` pipeline emits
the explicit `scf.parallel` loop in
`evidence/model2mlir_stream_triad_parallel_20261006.mlir`. This IR is the
current handoff point for Muon thread distribution; it has no Muon launch.
The `muon-mlir` driver accepts this parallel IR for native Muon LLVM IR
translation and records it as undistributed. It refuses target object and
ELF emission until a Muon launch is present.

`tests/run_model2mlir_stream_host.py` binds to the capture receipt and
compiles its parallel-loop MLIR through upstream LLVM dialect lowering and
the matching host Clang. It feeds source-generated input blobs to the four
compiled functions and compares all 1,048,576 output words per operation
against source-generated expected blobs. The four host executables passed;
`evidence/model2mlir_stream_host_20261006.json` records the IR and executable
hashes and complete output digests. Host execution proves the current typed
frontend and standard lowering semantics for these four kernels. It does not
qualify Muon scheduling, the RV32 stack ABI, or SoC execution.

`tests/run_model2mlir_gemm_host.py` similarly binds the GEMM capture receipt,
reads the handwritten generator's actual `A_raw`, `B_raw`, and
`expected_C_raw` arrays, lowers the captured `linalg.matmul` through upstream
MLIR to host LLVM IR, and compares all 4,096 BF16 result words. The compiled
host output matched every source golden word; the executable and IR hashes
are in `evidence/model2mlir_gemm_simt_host_20261006.json`. This checks the
generated typed MLIR computation, not the Muon target schedule.

For the MX case, the model2MLIR external quantization API uses the MX OOT
adapter and the `microscaling-quant` operand capture package. The PyTorch
graph establishes contraction identity and target selection. The handwritten
FP8 code and E8M0 scale blobs are the numerical oracle; ideal PyTorch matmul
is not a substitute for the source `mx_golden`. The current MX capture does
not yet bind source blobs or compare accelerator results.

These captures are the frontend inputs for target lowering, not target
programs. The next Muon compiler step is to distribute the generated
parallel loops onto Muon launches and emit
native RV32 objects using a toolchain that supports the required stack
stride. The next MX step is to lower selected handoff sites into actual MX
operand loads, commands, waits, and readout with complete source-golden
comparison. Radiance then combines the two executable paths and checks
cross-engine synchronization under an artifact-bound SoC profile. The
existing handwritten LLVM-dialect import is only a reference baseline.
