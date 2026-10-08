# Lowering routes and promotion contract

`contracts/lowering_routes.json` fixes the owner and boundary for each route
that Radiance composes.
`tools/check_lowering_contract.py` checks those choices so a route change needs
an intentional code and contract review. The routes are:

| Target | Accelerator path | LLVM role |
| --- | --- | --- |
| MX Gemmini commands | MX OOT contract to physical commands, with Rocket RoCC and Muon MMIO issuers | Issuer backends only: RV64 host or native Muon RV32 |
| Muon standalone | Muon OOT ops to its runtime ABI | Native Muon RV32 backend required by the current toolchain |
| Radiance | Compose Muon and MX, verify handoffs and selected SoC profile; MX commands use Muon MMIO | RV64 host and native Muon RV32 backend |

Each route also declares named qualification proofs. MX needs a physical
schedule, RoCC and Muon MMIO issuers, and source output parity through each
issuer. Muon needs stack
ABI qualification, an RV32 executable, and source output parity. Radiance
needs profile identity, a linked mixed artifact, handoff ordering, mixed
execution, and source output parity. A route cannot be marked `validated`
until hashed receipts for every proof are present and pass the gate.

Radiance owns composition and artifact linking, not a third compute dialect.
Merlin provides generic target-provider selection and artifact bundles. It
must not acquire target-name branches. The route contract is a review gate,
not proof that every route has an executable compiler today.

Atlas and regular Gemmini have distinct lowering boundaries. Atlas can emit
packed IMEM words directly from its OOT dialect; regular Gemmini uses LLVM for
RV64 host code and instruction encoding. Their qualification gates belong in
their own repositories. Radiance does not control their validation status.

The source roster in `contracts/source_roster.json` projects the pinned Muon
v2 inventory, mirrored in `evidence/upstream/`, from radiance-kernels revision
`a27f6abd24830fdc7999d872d170ab778f1e662e`: 64 families and 136
Makefile-declared Radiance ELF targets. The optional `--inventory` argument
checks the projection against the original inventory. A source revision or
Makefile change requires refreshing that inventory, roster, captures, and
affected numerical or execution evidence. Different targets may require
different compatible SoC profiles; the U250 E4M3 profile cannot qualify FP4
or FP6 kernels.

`contracts/coverage.json` lists only proven scopes. STREAM and SIMT GEMM have
source-output parity through Muon callbacks on a host. Spatter covers its
Gather read trace, not destination write behavior. The MX entries cover
frontend handoff shape and the FP8 source payload; they do not claim MX
execution or source-golden output parity. The MX receipts are mirrored under
`evidence/upstream/` with content hashes; their owner remains mx-gemmini-mlir.
The CI check verifies route ownership, exact inventory and receipt bytes,
schema, status, source revision, and claim scope. `--require-complete-claims`
also requires a target execution receipt and artifact bundle for every one of
the 136 declared ELF targets, plus route qualification receipts. It fails
today by design. This is a metadata completeness check; it does not rerun
device execution or establish release readiness by itself.

```sh
python3 tools/check_lowering_contract.py \
  --inventory /path/to/muon-mlir/evidence/kernel-inventory-20261006.json
python3 -m unittest discover -s tests -p 'test_lowering_contract.py'
python3 tools/check_lowering_contract.py --require-complete-claims
```

To promote a target, ingest the source kernel or its source-equivalent
PyTorch/model2MLIR capture, emit a typed OOT path, compile an artifact, execute
it under a compatible source-bound profile, and compare the source golden
behavior. Use output words, a memory trace, or a control trace as the oracle
according to the kernel; this also covers kernels without an output buffer.
Record its target name, profile digest, route, artifact bundle, execution
environment, oracle kind, nonzero observation count, and matching digests in a
`radiance.target_execution.v1` receipt. Add the receipt to coverage and rerun
the checks. Structural IR or host callback success remains a lower evidence
level. The linked source inventory and receipts make gaps explicit even when
implementation is staged over multiple releases.

This CI workflow must become a required branch check before it can prevent
merges on GitHub. The local gate works without a network connection. A release
gate must also rebuild and execute the compiled artifact under the selected
SoC profile, check the source oracle, and review the run logs. This metadata
checker validates receipt identity and declared fields only.
