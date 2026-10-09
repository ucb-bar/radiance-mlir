"""Source and artifact identity checks for Radiance SoC compiler profiles."""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
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
    if not isinstance(mx["formats"], list) or any(fmt not in {"mxfp4", "mxfp6", "mxfp8"}
                                                     for fmt in mx["formats"]):
        raise ProfileError("unknown legacy MX format")
    named = mx.get("named_formats")
    if named is not None and (not isinstance(named, list) or not named or
                              any(fmt not in {"fp4_e2m1", "fp6_e2m3", "fp6_e3m2",
                                               "fp8_e4m3", "fp8_e5m2"} for fmt in named)):
        raise ProfileError("unknown named MX format")
    for feature in ("lut", "vpu", "spad_requant"):
        if feature in mx and type(mx[feature]) is not bool:
            raise ProfileError(f"MX {feature} must be Boolean")
    target_digest = mx.get("target_profile_sha256")
    if target_digest is not None and (not isinstance(target_digest, str) or
                                      re.fullmatch(r"[0-9a-f]{64}", target_digest) is None):
        raise ProfileError("MX target profile digest must be lowercase SHA-256")
    if (mx.get("vpu", False) or mx.get("spad_requant", False)) and target_digest is None:
        raise ProfileError("MX VPU/SPAD_REQUANT needs a source-bound target profile digest")
    if target_digest is not None and (not isinstance(mx.get("gemmini_config"), str) or
                                      not mx["gemmini_config"] or named is None):
        raise ProfileError("source-bound MX target needs a Gemmini config and named formats")
    return profile


def mlir_pass_options(profile: dict, digest: str) -> str:
    """Pass only capabilities from a validated SoC profile to radiance-opt."""
    mx = profile["mx"]
    legacy_to_named = {"mxfp4": "fp4_e2m1", "mxfp6": "fp6_e3m2", "mxfp8": "fp8_e4m3"}
    named = mx.get("named_formats") or [legacy_to_named[fmt] for fmt in mx["formats"]]
    fields = [f"selected-name={profile['name']}", f"selected-sha256={digest}",
              f"selected-formats={','.join(mx['formats'])}",
              f"selected-named-formats={','.join(named)}",
              f"selected-lut={str(mx.get('lut', False)).lower()}",
              f"selected-vpu={str(mx.get('vpu', False)).lower()}",
              f"selected-spad-requant={str(mx.get('spad_requant', False)).lower()}"]
    if mx.get("target_profile_sha256"):
        fields.append(f"selected-mx-profile-sha256={mx['target_profile_sha256']}")
    return "--radiance-verify-profile=" + " ".join(fields)


def verify_mx_target(profile: dict, mx_source: Path | None,
                     target_json: Path | None, rtl_root: Path | None) -> None:
    """Recompute the selected MX profile from the current Gemmini RTL sources."""
    mx = profile["mx"]
    expected = mx.get("target_profile_sha256")
    if expected is None:
        return
    if mx_source is None or target_json is None or rtl_root is None:
        raise ProfileError("source-bound MX target requires --mx-mlir-source, "
                           "--mx-target-profile and --mx-rtl-root")
    if not (mx_source / "mx_gemmini_support/target_profile.py").is_file():
        raise ProfileError("MX MLIR source path lacks the target profile provider")
    sys.path.insert(0, str(mx_source.resolve()))
    try:
        from mx_gemmini_support import target_profile as provider
        if Path(provider.__file__).resolve() != (mx_source / "mx_gemmini_support/target_profile.py").resolve():
            raise ProfileError("loaded MX profile provider differs from --mx-mlir-source")
        target = provider.load_profile(target_json, rtl_root=rtl_root)
    except (OSError, ValueError) as exc:
        raise ProfileError(f"MX target differs from selected RTL sources: {exc}") from exc
    finally:
        sys.path.pop(0)
    if provider.profile_sha256(target) != expected:
        raise ProfileError("MX target profile digest differs from selected SoC profile")
    if target["gemmini_config"] != mx["gemmini_config"]:
        raise ProfileError("MX Gemmini config differs from selected SoC profile")
    for feature in ("lut", "vpu", "spad_requant"):
        if bool(target["resources"][feature]) != mx.get(feature, False):
            raise ProfileError(f"MX {feature} differs from selected RTL target profile")
    legal_formats = {item[key] for item in target["legal_compute"]
                     for key in ("activation_format", "weight_format")}
    legal_formats.update(target["candidate_output_modes"])
    if not set(mx.get("named_formats", [])).issubset(legal_formats):
        raise ProfileError("named MX formats exceed selected RTL target profile")
    precision = {"mxfp4": "fp4_", "mxfp6": "fp6_", "mxfp8": "fp8_"}
    if any(not any(fmt.startswith(precision[legacy]) for fmt in legal_formats)
           for legacy in mx["formats"]):
        raise ProfileError("legacy MX formats exceed selected RTL target profile")


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
