"""Negative profile admissions independent of the MLIR pass."""
from __future__ import annotations

import argparse
import sys
import hashlib
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soc_profile import (ProfileError, load_profile, require_format,
                         verify_inputs, verify_mx_target)  # noqa: E402


def refuses(action, label: str) -> None:
    try:
        action()
    except (ProfileError, FileNotFoundError):
        print(f"{label}: rejected")
        return
    raise AssertionError(f"{label}: unexpectedly accepted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mx-mlir-source", type=Path)
    parser.add_argument("--mx-rtl-root", type=Path)
    args = parser.parse_args()
    if (args.mx_mlir_source is None) != (args.mx_rtl_root is None):
        parser.error("both MX source paths are required for source-binding test")
    profile = load_profile(ROOT / "profiles/u250-e4m3.yaml")
    require_format(profile, "mxfp8")
    refuses(lambda: require_format(profile, "mxfp4"), "U250 FP4")
    refuses(lambda: require_format(profile, "mxfp8", lut=True), "U250 QuantLut")
    with tempfile.TemporaryDirectory(prefix="radiance-profile-") as directory:
        root = Path(directory)
        candidate = dict(profile, mx=dict(profile["mx"], vpu=True))
        candidate_path = root / "vpu.yaml"
        candidate_path.write_text(yaml.safe_dump(candidate))
        refuses(lambda: load_profile(candidate_path), "VPU without source-bound MX target")
        candidate["mx"].update(target_profile_sha256="d" * 64)
        candidate_path.write_text(yaml.safe_dump(candidate))
        refuses(lambda: load_profile(candidate_path), "VPU without Gemmini config and named formats")
        candidate["mx"].update(gemmini_config="e4m3SingleFp4", named_formats=["fp8_e4m3"])
        candidate_path.write_text(yaml.safe_dump(candidate))
        selected = load_profile(candidate_path)
        refuses(lambda: verify_mx_target(selected, None, None, None),
                "VPU target without MX source checkout")
        (root / "wrong.dts").write_text("wrong device tree")
        (root / "wrong.tar.gz").write_bytes(b"wrong bitstream")
        refuses(lambda: verify_inputs(profile, root, root / "wrong.dts",
                                      root / "wrong.tar.gz"),
                "unbound source and artifacts")
        artifact_profile = dict(profile, source_files={},
                                device_tree_sha256=hashlib.sha256(
                                    (root / "wrong.dts").read_bytes()).hexdigest())
        refuses(lambda: verify_inputs(artifact_profile, root, root / "wrong.dts",
                                      root / "wrong.tar.gz"), "wrong bitstream archive")
    if args.mx_mlir_source:
        mx_source = args.mx_mlir_source.resolve()
        sys.path.insert(0, str(mx_source))
        from mx_gemmini_support import target_profile as provider
        target_json = (mx_source / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")
        target = provider.load_profile(target_json)
        selected = dict(profile, mx=dict(profile["mx"],
                                        named_formats=["fp8_e4m3"],
                                        gemmini_config=target["gemmini_config"],
                                        vpu=True, spad_requant=True,
                                        target_profile_sha256=provider.profile_sha256(target)))
        verify_mx_target(selected, mx_source, target_json, args.mx_rtl_root)
        print("source-bound VPU target: accepted")
        corrupted = dict(selected, mx=dict(selected["mx"], target_profile_sha256="0" * 64))
        refuses(lambda: verify_mx_target(corrupted, mx_source, target_json,
                                         args.mx_rtl_root), "wrong MX target digest")


if __name__ == "__main__":
    main()
