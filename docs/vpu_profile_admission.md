# Admitting an MX VPU Radiance profile

The `gemmini-mx-cleanup` source checkout at commit
`266c593f2cb51d7e3fe83fc0317072b585ac3c52` contains the MX VPU and
`SPAD_REQUANT`. The MX package exports a structural target profile for every
named MX fragment and Chipyard wrapper. A profile records legal operand pairs,
PE modes, memory geometry, command ABI, and VPU/requantization capabilities.
Its `structural_unqualified` status does not establish numerical or FPGA
execution correctness.

The inspected `repack/xilinx_alveo_u250/firesim.bit`
has SHA-256 `a6485b39c673c8e3205f004d727ca6aed203c5b1b9b0c1d10911f71835996e28`.
The adjacent `metadata` names `chipyard.MxGemminiRocketConfig`, while the
public VPU first appears in Gemmini commit `737534f3b3580e25e407d5670378a0f4758c1ab8`
after this image's September 27 timestamp. The metadata does not establish
that this image contains Radiance or the VPU. Do not use this image as VPU
hardware evidence. The existing `u250-e4m3` Radiance profile also has no VPU
or scratchpad requantization capability.

For a new SoC build, record its exact FireSim configuration and artifact
hashes in a separate `radiance.soc_profile.v1` YAML file. Include the complete
SoC source closure in `source_files`. The `mx` mapping must declare
`formats`, `named_formats`, `lut`, `vpu`, `spad_requant`, `gemmini_config`, and
`target_profile_sha256`. The target digest is the canonical digest emitted by
`mx_gemmini_support.target_profile.profile_sha256` for the corresponding MX
profile JSON. The selected MX JSON must come from the exact Gemmini/MxGen
checkout used for the build.

Run the artifact and source check before any MLIR verification:

```sh
python3 check_ir.py input.mlir \
  --profile profiles/new-radiance-vpu.yaml \
  --chipyard /path/to/build/chipyard \
  --device-tree /path/to/build/device-tree.dts \
  --bitstream-archive /path/to/build/bitstream.tar.gz \
  --mx-mlir-source /path/to/mx-gemmini-mlir \
  --mx-target-profile /path/to/mx-target.json \
  --mx-rtl-root /path/to/build/gemmini \
  --radiance-opt build/tools/radiance-opt
```

`check_ir.py` regenerates the selected MX JSON from the RTL source closure,
checks its digest and Gemmini config against the SoC YAML, and compares the
declared VPU, requantization, LUT and named format capabilities. The pass then
requires matching `mx.profile_sha256` on the module and every MX operation,
and refuses VPU or requantization operations when the selected profile lacks
them. This is source and IR admission; a reproducible bitstream build manifest
and execution tests are still needed to qualify an FPGA image.
