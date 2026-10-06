"""Verify composed MLIR against a selected, byte-bound Radiance SoC profile."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from profile import _sha256, load_profile, verify_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--chipyard", type=Path, required=True)
    parser.add_argument("--device-tree", type=Path, required=True)
    parser.add_argument("--bitstream-archive", type=Path, required=True)
    parser.add_argument("--radiance-opt", type=Path, required=True)
    args = parser.parse_args()
    selected = load_profile(args.profile)
    verify_inputs(selected, args.chipyard, args.device_tree, args.bitstream_archive)
    digest = _sha256(args.profile)
    subprocess.run(
        [str(args.radiance_opt),
         f"--radiance-verify-profile=selected-name={selected['name']} selected-sha256={digest}",
         str(args.input), "-o", "/dev/null"], check=True)
    print(f"IR verified against {selected['name']} ({digest}); status={selected['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
