"""Positive and negative checks for profile binding and MX/Muon handoff."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soc_profile import load_profile, mlir_pass_options  # noqa: E402
SOURCE = (ROOT / "tests/composed.mlir").read_text()
MIXED = (ROOT / "tests/mixed_attention.mlir").read_text()
CFG = (ROOT / "tests/cfg_handoff.mlir").read_text()
PROFILE_HASH = "2acc27561a647e54a91828d443e4bad8f0859527ad066b5613225803c6f343b6"
PROFILE = load_profile(ROOT / "profiles/u250-e4m3.yaml")
MX_DIGEST = "d" * 64


def vector_ir(source: str, *, profile_digest: str = MX_DIGEST) -> str:
    source = source.replace('  radiance.profile_sha256 =',
                            f'  mx.profile_sha256 = "{profile_digest}",\n  radiance.profile_sha256 =')
    source = source.replace('site_id = "qk", contract_sha256',
                            f'site_id = "qk", profile_sha256 = "{profile_digest}", contract_sha256')
    a, b, c = "a" * 64, "b" * 64, "c" * 64
    binding = (f'profile_sha256 = "{profile_digest}", contract_sha256 = "{a}", '
               f'policy_sha256 = "{b}", manifest_sha256 = "{c}"')
    commands = f'''    "mx_gemmini.vpu_execute"() {{site_id = "softmax", kind = "expsum",
      src1_row = 2048 : i32, src2_row = 4096 : i32, dst_row = 6144 : i32,
      rows = 128 : i32, reduction_length = 8 : i32, broadcast = true,
      immediate_bf16 = 0 : i32, second_dst_row = 7168 : i32, {binding}}} : () -> ()
    "mx_gemmini.spad_requant"() {{site_id = "p-feed", source_row = 4096 : i32,
      destination_row = 3072 : i32, m = 64 : i32, n = 128 : i32,
      output_format = "fp8_e4m3", tiled = true, resident = true,
      scale_dram_address = 268435456 : i64, {binding}}} : () -> ()
'''
    return source.replace('    "mx_gemmini.readout_to_smem"',
                          commands + '    "mx_gemmini.readout_to_smem"', 1)


def check(tool: Path, source: str, should_pass: bool, reason: str,
          *, selected_formats: str = "mxfp8", selected_lut: bool = False,
          selected_named_formats: str = "fp8_e4m3",
          profile: dict = PROFILE, expected_error: str | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="radiance-verify-") as directory:
        path = Path(directory) / "input.mlir"
        path.write_text(source)
        options = mlir_pass_options(profile, PROFILE_HASH)
        options = options.replace('selected-formats=mxfp8',
                                  f'selected-formats={selected_formats}')
        options = options.replace('selected-named-formats=fp8_e4m3',
                                  f'selected-named-formats={selected_named_formats}')
        options = options.replace('selected-lut=false',
                                  f'selected-lut={str(selected_lut).lower()}')
        command = [str(tool), options,
                   str(path), "-o", "/dev/null"]
        result = subprocess.run(command, capture_output=True, text=True)
        if (result.returncode == 0) != should_pass:
            raise AssertionError(f"{reason}: return={result.returncode}\n{result.stderr}")
        if expected_error and expected_error not in result.stderr:
            raise AssertionError(f"{reason}: expected {expected_error!r}\n{result.stderr}")
        print(f"{reason}: {'accepted' if should_pass else 'rejected'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiance-opt", type=Path, required=True)
    args = parser.parse_args()
    check(args.radiance_opt, SOURCE, True, "pinned U250 handoff")
    check(args.radiance_opt, SOURCE, False, "U250 refuses forged FP4 capability",
          selected_formats="mxfp8,mxfp4")
    check(args.radiance_opt, SOURCE, False, "U250 refuses forged named FP4 capability",
          selected_named_formats="fp8_e4m3,fp4_e2m1")
    check(args.radiance_opt, SOURCE, False, "U250 refuses forged QuantLut capability",
          selected_lut=True)
    vector = vector_ir(SOURCE)
    check(args.radiance_opt, vector, False, "U250 refuses MX VPU",
          expected_error="no MX VPU")
    check(args.radiance_opt, vector, False, "U250 refuses scratchpad requantization",
          expected_error="no SPAD_REQUANT")
    synthetic = dict(PROFILE, name="synthetic-vpu",
                     mx=dict(PROFILE["mx"], named_formats=["fp8_e4m3"],
                             vpu=True, spad_requant=True,
                             target_profile_sha256=MX_DIGEST))
    admitted = vector.replace('radiance.profile = "u250-e4m3",',
                              'radiance.profile = "synthetic-vpu",')
    check(args.radiance_opt, admitted, True, "synthetic VPU capability gate",
          profile=synthetic)
    check(args.radiance_opt, admitted.replace(MX_DIGEST, "0" * 64), False,
          "synthetic VPU digest mismatch", profile=synthetic,
          expected_error="selected source-bound MX profile digest")
    check(args.radiance_opt, MIXED,
          True, "QK to Muon to PV structural flow")
    check(args.radiance_opt, CFG, True, "MX handoff across CFG blocks")
    check(args.radiance_opt,
          CFG.replace('    "mx_gemmini.wait"()', '    "mx_gemmini.other_wait"()'),
          False, "missing wait across CFG blocks")
    divergent = (CFG.replace('%shared: memref<16x16xbf16>)',
                             '%shared: memref<16x16xbf16>, %take: i1)')
                 .replace('    cf.br ^bb1', '    cf.cond_br %take, ^bb1, ^bb3')
                 .replace('    cf.br ^bb2\n  ^bb2:',
                          '    cf.br ^bb2\n  ^bb3:\n    cf.br ^bb2\n  ^bb2:'))
    check(args.radiance_opt, divergent, False, "divergent completion at CFG join")
    check(args.radiance_opt,
          SOURCE.replace(PROFILE_HASH, "0" * 64), False, "wrong profile digest")
    check(args.radiance_opt,
          SOURCE.replace('site_id = "qk", contract_sha256',
                         'site_id = "qk", format = "mxfp4", contract_sha256', 1),
          False, "FP4 is absent from U250")
    check(args.radiance_opt,
          SOURCE.replace('"mx_gemmini.wait"', '"mx_gemmini.other_wait"'),
          False, "missing MX wait")
    check(args.radiance_opt,
          SOURCE.replace('site_id = "qk", contract_sha256', 'site_id = "pv", contract_sha256', 1),
          False, "readout/wait site mismatch")
    check(args.radiance_opt,
          SOURCE.replace('"muon.smem_fence"() : () -> ()', ''),
          False, "missing shared-memory fence")
    check(args.radiance_opt,
          SOURCE.replace('warps_per_core = 4 : i32', 'warps_per_core = 9 : i32'),
          False, "invalid Muon occupancy")
    check(args.radiance_opt,
          SOURCE.replace('  "muon.barrier"() {barrier_id = 1 : i32, num_warps = 8 : i32} : () -> ()', ''),
          False, "missing Muon barrier")
    launch = ('    "muon.launch"(%muon_args) '
              '{kernel = @softmax_quant_kernel, warps_per_core = 4 : i32} : (!llvm.ptr) -> ()')
    check(args.radiance_opt,
          MIXED.replace('    "mx_gemmini.wait"() {', launch + '\n    "mx_gemmini.wait"() {', 1),
          False, "Muon consumer launched before MX handoff")
    check(args.radiance_opt,
          MIXED.replace('    "muon.barrier"() {barrier_id = 2 : i32, num_warps = 8 : i32} : () -> ()', ''),
          False, "MX consumer after Muon launch without barrier")
    check(args.radiance_opt,
          MIXED.replace('    "muon.barrier"() {barrier_id = 2 : i32, num_warps = 8 : i32} : () -> ()\n'
                        '    "muon.smem_fence"() : () -> ()',
                        '    "muon.barrier"() {barrier_id = 2 : i32, num_warps = 8 : i32} : () -> ()'),
          False, "MX consumer after Muon launch without shared-memory fence")


if __name__ == "__main__":
    main()
