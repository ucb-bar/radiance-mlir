"""Positive and negative checks for profile binding and MX/Muon handoff."""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tests/composed.mlir").read_text()
MIXED = (ROOT / "tests/mixed_attention.mlir").read_text()
CFG = (ROOT / "tests/cfg_handoff.mlir").read_text()
PROFILE_HASH = "2acc27561a647e54a91828d443e4bad8f0859527ad066b5613225803c6f343b6"


def check(tool: Path, source: str, should_pass: bool, reason: str,
          *, selected_formats: str = "mxfp8", selected_lut: bool = False) -> None:
    with tempfile.TemporaryDirectory(prefix="radiance-verify-") as directory:
        path = Path(directory) / "input.mlir"
        path.write_text(source)
        command = [str(tool),
                   f"--radiance-verify-profile=selected-name=u250-e4m3 "
                   f"selected-sha256={PROFILE_HASH} selected-formats={selected_formats} "
                   f"selected-lut={'true' if selected_lut else 'false'}",
                   str(path), "-o", "/dev/null"]
        result = subprocess.run(command, capture_output=True, text=True)
        if (result.returncode == 0) != should_pass:
            raise AssertionError(f"{reason}: return={result.returncode}\n{result.stderr}")
        print(f"{reason}: {'accepted' if should_pass else 'rejected'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiance-opt", type=Path, required=True)
    args = parser.parse_args()
    check(args.radiance_opt, SOURCE, True, "pinned U250 handoff")
    check(args.radiance_opt, SOURCE, False, "U250 refuses forged FP4 capability",
          selected_formats="mxfp8,mxfp4")
    check(args.radiance_opt, SOURCE, False, "U250 refuses forged QuantLut capability",
          selected_lut=True)
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
