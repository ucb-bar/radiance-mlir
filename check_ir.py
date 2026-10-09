"""Verify composed MLIR against a selected, byte-bound Radiance SoC profile."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from soc_profile import (_sha256, load_profile, mlir_pass_options, verify_inputs,
                         verify_mx_target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--chipyard", type=Path, required=True)
    parser.add_argument("--device-tree", type=Path, required=True)
    parser.add_argument("--bitstream-archive", type=Path, required=True)
    parser.add_argument("--radiance-opt", type=Path, required=True)
    parser.add_argument("--mx-mlir-source", type=Path)
    parser.add_argument("--mx-target-profile", type=Path)
    parser.add_argument("--mx-rtl-root", type=Path)
    args = parser.parse_args()
    selected = load_profile(args.profile)
    verify_inputs(selected, args.chipyard, args.device_tree, args.bitstream_archive)
    verify_mx_target(selected, args.mx_mlir_source, args.mx_target_profile, args.mx_rtl_root)
    digest = _sha256(args.profile)
    subprocess.run(
        [str(args.radiance_opt),
         mlir_pass_options(selected, digest),
         str(args.input), "-o", "/dev/null"], check=True)
    print(f"IR verified against {selected['name']} ({digest}); status={selected['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
