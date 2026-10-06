"""Source and artifact identity checks for Radiance SoC compiler profiles."""
from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path

import yaml


class ProfileError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_profile(path: Path) -> dict:
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("schema") != "radiance.soc_profile.v1":
        raise ProfileError("expected radiance.soc_profile.v1")
    muon = profile.get("muon", {})
    mx = profile.get("mx", {})
    if not isinstance(muon, dict) or not isinstance(mx, dict):
        raise ProfileError("muon and mx must be mappings")
    if muon.get("clusters", 0) < 1 or muon.get("cores_per_cluster", 0) < 1:
        raise ProfileError("invalid Muon topology")
    if muon.get("lanes_per_warp") != 16:
        raise ProfileError("unsupported Muon warp width")
    if not mx.get("formats"):
        raise ProfileError("MX formats must be declared")
    return profile


def require_format(profile: dict, fmt: str, *, lut: bool = False) -> None:
    mx = profile["mx"]
    if fmt not in mx["formats"]:
        raise ProfileError(f"{profile['name']}: unsupported MX format {fmt}")
    if lut and not mx["lut"]:
        raise ProfileError(f"{profile['name']}: QuantLut is absent")


def verify_inputs(profile: dict, chipyard: Path, device_tree: Path, bitstream_archive: Path) -> None:
    for relative, expected in profile.get("source_files", {}).items():
        actual = _sha256(chipyard / relative)
        if actual != expected:
            raise ProfileError(f"source identity differs: {relative}")
    for path, key in ((device_tree, "device_tree_sha256"),
                      (bitstream_archive, "bitstream_archive_sha256")):
        expected = profile.get(key)
        if expected is None:
            raise ProfileError(f"profile lacks {key}")
        if _sha256(path) != expected:
            raise ProfileError(f"artifact identity differs: {key}")
    expected_bitstream = profile.get("bitstream_sha256")
    if expected_bitstream:
        try:
            with tarfile.open(bitstream_archive, "r:gz") as archive:
                members = [item for item in archive.getmembers()
                           if item.isfile() and item.name.endswith(".bit")]
                if len(members) != 1:
                    raise ProfileError("bitstream archive must contain one .bit payload")
                stream = archive.extractfile(members[0])
                if stream is None:
                    raise ProfileError("cannot read bitstream payload")
                actual = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    actual.update(block)
        except (OSError, tarfile.TarError) as exc:
            raise ProfileError(f"cannot inspect bitstream archive: {exc}") from exc
        if actual.hexdigest() != expected_bitstream:
            raise ProfileError("artifact identity differs: bitstream_sha256")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--chipyard", type=Path, required=True)
    parser.add_argument("--device-tree", type=Path, required=True)
    parser.add_argument("--bitstream-archive", type=Path, required=True)
    parser.add_argument("--format", choices=("mxfp8", "mxfp6", "mxfp4"))
    parser.add_argument("--uses-lut", action="store_true")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    verify_inputs(profile, args.chipyard, args.device_tree, args.bitstream_archive)
    if args.format:
        require_format(profile, args.format, lut=args.uses_lut)
    print(f"verified selected input hashes for {profile['name']}; status={profile['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
