"""Compare linkable RV32 contents from direct and MLIR-round-trip objects.

This checks loadable section bytes, BSS sizes, relocations, and link symbols.
It does not check final linking, execution, or numerical results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_facts(path: Path, readobj: Path) -> tuple[list[dict], list[tuple], list[tuple]]:
    result = subprocess.run([str(readobj), "--elf-output-style=JSON", "--sections",
                             "--relocations", "--symbols", str(path)],
                            capture_output=True, text=True, check=True)
    facts = json.loads(result.stdout)[0]
    entries = facts["Sections"]
    section_names = {entry["Section"]["Index"]: entry["Section"]["Name"]["Name"]
                     for entry in entries}
    blob = path.read_bytes()
    output = []
    for entry in entries:
        section = entry["Section"]
        if not section["Flags"]["Value"] & 2:  # SHF_ALLOC
            continue
        name = section["Name"]["Name"]
        kind = section["Type"]["Name"]
        size = section["Size"]
        row = {"name": name, "type": kind, "flags": section["Flags"]["Value"],
               "size": size,
               "alignment": section["AddressAlignment"]}
        if kind != "SHT_NOBITS":
            offset = section["Offset"]
            if offset + size > len(blob):
                raise RuntimeError(f"invalid section bounds: {path}:{name}")
            row["sha256"] = hashlib.sha256(blob[offset:offset + size]).hexdigest()
        output.append(row)
    relocations = []
    for relocation_section in facts["Relocations"]:
        target_section = section_names[relocation_section["SectionIndex"]]
        for entry in relocation_section["Relocs"]:
            reloc = entry["Relocation"]
            relocations.append((target_section, reloc["Offset"], reloc["Type"]["Name"],
                                reloc["Symbol"]["Name"], reloc["Addend"]))
    symbols = []
    for entry in facts["Symbols"]:
        symbol = entry["Symbol"]
        if symbol["Type"]["Name"] == "File":
            continue  # LLVM dialect export changes the source file label only.
        symbols.append((symbol["Name"]["Name"], symbol["Value"], symbol["Size"],
                        symbol["Binding"]["Name"], symbol["Type"]["Name"],
                        symbol["Section"]["Name"], symbol["Other"]["Value"]))
    return output, sorted(relocations), sorted(symbols)


def compare(row: dict, args: argparse.Namespace) -> dict:
    family, target = row["family"], row["target"]
    case = args.sweep_root / family / target
    source = case / "source.ll"
    imported = case / "import/reference.mu.o"
    record = {"family": family, "target": target, "status": "input_missing"}
    try:
        if sha256(source) != row["llvm_sha256"] or sha256(imported) != row["object_sha256"]:
            raise RuntimeError("sweep input hash mismatch")
        direct = args.out / family / target / "direct.mu.o"
        direct.parent.mkdir(parents=True, exist_ok=True)
        command = [str(args.muon_clang), "-target", "riscv32-unknown-elf",
                   "-march=rv32im_zfinx_zhinx", "-mabi=ilp32", "-Xclang",
                   "-target-feature", "-Xclang", "+vortex", "-O3",
                   "-mcmodel=medany", "-x", "ir", "-c", str(source), "-o", str(direct)]
        result = subprocess.run(command, capture_output=True, text=True)
        (direct.parent / "direct_compile.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            record["status"] = "direct_compile_failed"
            record["error"] = f"clang exited {result.returncode}"
            return record
        direct_sections, direct_relocs, direct_symbols = object_facts(direct, args.llvm_readobj)
        imported_sections, imported_relocs, imported_symbols = object_facts(imported, args.llvm_readobj)
        record["direct_object_sha256"] = sha256(direct)
        record["imported_object_sha256"] = row["object_sha256"]
        record["direct_alloc_sections"] = direct_sections
        record["imported_alloc_sections"] = imported_sections
        record["direct_relocations_sha256"] = hashlib.sha256(
            json.dumps(direct_relocs).encode()).hexdigest()
        record["imported_relocations_sha256"] = hashlib.sha256(
            json.dumps(imported_relocs).encode()).hexdigest()
        record["relocation_count"] = len(direct_relocs)
        record["direct_symbols_sha256"] = hashlib.sha256(
            json.dumps(direct_symbols).encode()).hexdigest()
        record["imported_symbols_sha256"] = hashlib.sha256(
            json.dumps(imported_symbols).encode()).hexdigest()
        record["link_symbol_count"] = len(direct_symbols)
        if direct_sections != imported_sections:
            record["status"] = "alloc_sections_differ"
            left = Counter((item["name"], item["type"], item["size"],
                            item.get("sha256")) for item in direct_sections)
            right = Counter((item["name"], item["type"], item["size"],
                             item.get("sha256")) for item in imported_sections)
            record["direct_only"] = [name for name, *_ in (left - right).elements()]
            record["imported_only"] = [name for name, *_ in (right - left).elements()]
        elif direct_relocs != imported_relocs:
            record["status"] = "relocation_records_differ"
            record["direct_only"] = [list(item) for item in (Counter(direct_relocs)
                                                              - Counter(imported_relocs)).elements()]
            record["imported_only"] = [list(item) for item in (Counter(imported_relocs)
                                                                - Counter(direct_relocs)).elements()]
        elif direct_symbols != imported_symbols:
            record["status"] = "link_symbols_differ"
            record["direct_only"] = [list(item) for item in (Counter(direct_symbols)
                                                              - Counter(imported_symbols)).elements()]
            record["imported_only"] = [list(item) for item in (Counter(imported_symbols)
                                                                - Counter(direct_symbols)).elements()]
        else:
            record["status"] = "linkable_contents_identical"
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        record["error"] = str(error)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--muon-clang", type=Path, required=True)
    parser.add_argument("--llvm-readobj", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    for field in ("manifest", "sweep_root", "out", "muon_clang", "llvm_readobj"):
        setattr(args, field, getattr(args, field).resolve())
    source = json.loads(args.manifest.read_text())
    if source["schema"] != "radiance_reference_sweep.v1":
        parser.error("manifest must be a reference sweep")
    rows = [row for row in source["results"] if row["status"] == "native_object_emitted"]
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda row: compare(row, args), rows))
    document = {"schema": "radiance_reference_object_comparison.v1",
                "source_git_revision": source["source_git_revision"],
                "source_sweep_manifest_sha256": sha256(args.manifest),
                "scope": "loadable sections, relocations, and link symbols; execution unchecked",
                "attempted_target_count": len(rows), "results": results}
    (args.out / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"attempted": len(rows),
                      "status_counts": dict(Counter(row["status"] for row in results)),
                      "manifest": str(args.out / "manifest.json")}))


if __name__ == "__main__":
    main()
