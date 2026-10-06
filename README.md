# Radiance MLIR composition

This out-of-tree package composes `muon-mlir` and `mx-gemmini-mlir`. It owns
the selected SoC profile and cross-engine ordering checks. It does **not**
define a third compute dialect. Merlin remains target agnostic; its existing
explicit OOT provider interface is the intended entry point once an
executable compiler package is qualified.

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
only LLVM 23's `captures(none)` parameter attribute, which LLVM 18 cannot
parse, and rejects other capture attributes. Its JSON report records source,
MLIR, and object hashes and counts inline assembly ops. This path provides a
coverage baseline for the handwritten kernels; it does not use the typed
Muon or MX dialect lowering passes.

The pinned original-size STREAM Copy case in
`evidence/stream-copy-llvm-import-20261006.json` linked the imported object
with the same data and runtime as the source ELF. Cyclotron checked all
1,048,576 FP32 outputs and both guards: both output digests were
`49fabae987622325`; source and imported paths took 2,104,119 and 2,102,449
timing-model cycles respectively. Other kernel families still need separate
import and semantic checks before this baseline establishes broad coverage.
