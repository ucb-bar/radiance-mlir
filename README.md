# Radiance MLIR composition

This out-of-tree package composes `muon-mlir` and `mx-gemmini-mlir`. It owns
the selected SoC profile and cross-engine ordering checks. It does **not**
define a third compute dialect. Merlin remains target agnostic; its existing
explicit OOT provider interface is the intended entry point once an
executable compiler package is qualified.

The source-anchored PyTorch → latest model2MLIR → typed MLIR capture path is
documented in [docs/model2mlir_frontend.md](docs/model2mlir_frontend.md).
STREAM Copy/Scale/Add/Triad and SIMT GEMM have complete source-formula or
source-golden output checks before capture. The separate MX FP8 GEMM capture
uses the MX package's quantization adapter and emits a verified MX handoff.
Standard MLIR fusion and bufferization expose STREAM parallel loops; host
executables produced from those loops match every source output word.
SIMT GEMM also compiles from captured `linalg.matmul` to a host executable
and matches all 4,096 source BF16 golden words. A source-derived Spatter
Gather trace compiles and matches all 262,144 reads; its repeated dense
destination writes still need lowering. Muon now outlines the four captured
STREAM functions into callbacks with source ABI storage symbols, distributes
their loops across Muon lanes, and matches every source output word in host
execution. It also outlines and runs the captured Spatter Gather read trace
through a Muon callback, matching all 262,144 source reads. The captured
SIMT GEMM also runs as a staged Muon callback with barriers and matches all
4,096 source BF16 golden words in host execution. MX commands and
mixed-engine execution remain to be implemented.

Use the `handwritten-implementation` branches of both `muon-mlir` and
`mx-gemmini-mlir` for the build below. The MX branch adds `readout_to_smem`
and `wait`, which the composition tests use.

```text
Merlin interface / upstream MLIR
  -> Muon SIMT ops + upstream scf/memref/arith
  -> MX Gemmini encode/contract/readout
  -> Radiance profile and cross-engine verification
  -> Muon runtime ABI and MX command lowering
  -> fused RV64 host + RV32 Muon ELF
```

The selected U250 profile identifies
`FireSimE4M3MxGemminiRadianceConfig` over
`RadianceE4M3MxGemminiSingleClusterConfig`: one cluster, two Muon cores,
16 lanes per warp, eight maximum warps per core, 128 KiB shared memory with
four banks, and an E4M3-only MX mesh without QuantLut. It binds the observed
bitstream archive, the embedded `.bit` payload, and device tree by SHA-256.
The source checkout is dirty
and the exact build-time diff is unavailable, so the listed source files are
a candidate closure, not proof of bitstream-to-source equivalence. The
`--chipyard` argument for this profile must point to Nico's checkout (here
`/scratch/nicorakela/chipyard`); Agustin's separate Chipyard checkout has a
different FireChip configuration file and correctly fails the hash check.
Merlin's existing Phase 0 Radiance descriptor names `RadianceMuonConfig`
and explicitly describes a one-core elaboration. It is a different hardware
scope from this two-core U250 profile. A Phase 0 result from that descriptor
must not be relabeled as evidence for Nico's FireSim configuration; a new
source-bound Phase 0 contract must select this exact SoC profile and artifact
identity before those results can be compared.
`single-cluster-full-mx.yaml` is a provisional simulator profile for FP6
and FP4; it cannot pass the executable artifact check until its own source
and simulator evidence are supplied.

## Build and verify

```sh
cmake -S . -B build -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DMUON_MLIR_SOURCE_DIR=/path/to/muon-mlir \
  -DMX_GEMMINI_MLIR_SOURCE_DIR=/path/to/mx-gemmini-mlir
cmake --build build -j
python3 tests/run_verify.py --radiance-opt build/tools/radiance-opt
python3 check_ir.py tests/composed.mlir \
  --profile profiles/u250-e4m3.yaml --chipyard /path/to/nico/chipyard \
  --device-tree /path/to/device-tree.dts \
  --bitstream-archive /path/to/bitstream.tar.gz \
  --radiance-opt build/tools/radiance-opt
```

The profile pass requires the selected profile name and file SHA-256 as
explicit options and matches both to module attributes. For MX accumulator
readout into shared memory, it requires a same-site MX wait, a Muon shared
memory fence, and a Muon barrier in that order before control leaves the
block or a Muon launch begins. A following MX use after a Muon launch requires
a Muon barrier and shared-memory fence in that order. These checks operate
within one block; control-flow and alias analysis are still needed before
general mixed-engine execution. It rejects FP4/FP6 and QuantLut use under
the pinned U250 profile.
`tests/mixed_attention.mlir` shows an FP8 QK contraction, MX-to-Muon
shared-memory handoff, Muon callback launch for softmax/quantization, and
FP8 PV contraction. Its callback and MX command lowering are external, so
the test checks structure and provenance rather than numerical execution.

The available implementation verifies IR composition. Muon-only original-size
STREAM and Spatter Gather comparisons, plus fused STREAM RV64/RV32 guest
image construction, live in `muon-mlir`; MX command lowering, quantized
attention dataflow, mixed-engine
performance parity, and FPGA guest execution remain open. No existing
handwritten kernel result is treated as a compiler result or as a universal
golden output.

## Handwritten source reference import

`tools/import_llvm_reference.py` imports native Muon LLVM IR into LLVM-dialect
MLIR, exports LLVM IR, and compiles it with the Muon LLVM 18 backend. Use an
LLVM 23 `mlir-translate` to preserve source inline assembly. The tool removes
LLVM 23's `captures(none)` parameter attribute and GEP no-wrap flags, which
LLVM 18 cannot parse, and lifetime hints whose signature changed between
releases. It also converts LLVM 23 float and half literal spelling and removes
`nocreateundeforpoison`. It rejects other capture attributes. Its JSON report
records source, MLIR, and object hashes and counts inline assembly ops. This path provides a
coverage baseline for the handwritten kernels; it does not use the typed
Muon or MX dialect lowering passes.

`tools/sweep_reference_imports.py` applies this path to every default
Radiance ELF target in the pinned `muon-mlir` v2 inventory. It queries each
Makefile with `make -Bn`, compiles the selected source to LLVM IR, imports and
exports it through LLVM-dialect MLIR, and emits an RV32 object. Its arguments
select the source worktree, Muon Clang, LLVM 23 `mlir-translate`, Muon libc++
headers, RISC-V C headers, generated `__config_site`, and MX submodule.
Compiler outputs stay under `--out`; each failed target retains its stage and
diagnostic. The manifest records source Git revision, tracked worktree diff
hash, and hashes of generated kernel inputs. The pinned result is
`evidence/reference-sweep-prepared-20261006.json`: **110/136** objects emitted
across **41 kernel families**. Twelve target names have no matching Muon
object recipe in their Makefiles; fourteen fail source compilation, including
six missing `data` files, four missing `ubench/lib.h` includes, and four
source/header errors. The 136-target inventory is a Makefile-declared target
count, not a claim that all 136 source targets currently build. Generated
inputs were prepared only in an isolated source worktree; the active
`spatter-workloads` checkout was not edited. Object emission confirms
importability and native code generation, not executable or numerical parity.
`tools/compare_reference_objects.py` separately compiles each saved source
LLVM file with the same Muon LLVM backend and compares its RV32 object with
the MLIR-round-trip object. The pinned
`evidence/reference-object-comparison-20261006.json` reports **110/110**
identical loadable section bytes, BSS sizes, relocation records, and link
symbols (excluding the source-file label, which the LLVM dialect exporter
renames). This is object-level parity under the available Muon LLVM 18 build.
That build predates the source Makefiles' stack-word-stride backend option, so
the sweep omits that option for both direct and imported paths; the result
does not qualify the final SoC stack ABI or prove linked execution.

The pinned original-size STREAM Copy case in
`evidence/stream-copy-llvm-import-20261006.json` linked the imported object
with the same data and runtime as the source ELF. Cyclotron checked all
1,048,576 FP32 outputs and both guards: both output digests were
`49fabae987622325`; source and imported paths took 2,104,119 and 2,102,449
timing-model cycles respectively. Other kernel families still need separate
import and semantic checks before this baseline establishes broad coverage.

`evidence/gemm-mxgemmini-ws-restream-reference-20261006.json` pins a mixed
MX/Muon source from the active radiance-kernels checkout. After importing its
native LLVM IR into MLIR and recompiling, the source and imported RV32 objects
had identical `.text` and `.rodata` bytes. The MLIR retained 151 inline
assembly ops. That is an instruction-level reference check; the mixed kernel
has not yet been linked, run, or expressed as typed MX and Muon ops.
