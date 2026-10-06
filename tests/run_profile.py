"""Negative profile admissions independent of the MLIR pass."""
from __future__ import annotations

import sys
import hashlib
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profile import ProfileError, load_profile, require_format, verify_inputs  # noqa: E402


def refuses(action, label: str) -> None:
    try:
        action()
    except (ProfileError, FileNotFoundError):
        print(f"{label}: rejected")
        return
    raise AssertionError(f"{label}: unexpectedly accepted")


def main() -> None:
    profile = load_profile(ROOT / "profiles/u250-e4m3.yaml")
    require_format(profile, "mxfp8")
    refuses(lambda: require_format(profile, "mxfp4"), "U250 FP4")
    refuses(lambda: require_format(profile, "mxfp8", lut=True), "U250 QuantLut")
    with tempfile.TemporaryDirectory(prefix="radiance-profile-") as directory:
        root = Path(directory)
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


if __name__ == "__main__":
    main()
