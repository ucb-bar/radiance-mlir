"""Import native Muon LLVM IR into MLIR and emit a native Muon object.

This is a source-equivalence reference path. It retains LLVM dialect ops,
including source inline assembly; it does not claim that Muon or MX ops were
recognized or lowered by their out-of-tree dialects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], log: Path) -> None:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command[0]}; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-llvm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mlir-translate", type=Path, required=True,
                        help="LLVM 23 translator with LLVM inline-asm import")
    parser.add_argument("--muon-clang", type=Path, required=True,
                        help="Muon LLVM 18 clang with the Vortex target")
    parser.add_argument("--native-flag", action="append", default=[],
                        help="additional native backend flag; repeat as needed")
    args = parser.parse_args()
    source = args.input_llvm.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mlir = output / "reference.llvm.mlir"
    exported = output / "reference.exported.ll"
    compatible = output / "reference.native.ll"
    native = output / "reference.mu.o"
    translator = str(args.mlir_translate.resolve())
    clang = str(args.muon_clang.resolve())

    run([translator, "--import-llvm", str(source), "-o", str(mlir)],
        output / "import.log")
    run([translator, "--mlir-to-llvmir", str(mlir), "-o", str(exported)],
        output / "export.log")
    text = exported.read_text()
    # LLVM 23 prints this conservative parameter attribute; LLVM 18 rejects
    # its syntax. Removing it cannot strengthen an optimization assumption.
    other_capture = re.search(r"\bcaptures\((?!none\))", text)
    if other_capture:
        raise RuntimeError("unsupported LLVM 23 capture attribute; see reference.exported.ll")
    removed = text.count(" captures(none)")
    compatible.write_text(text.replace(" captures(none)", ""))
    run([clang, "-target", "riscv32-unknown-elf", "-march=rv32im_zfinx_zhinx",
         "-mabi=ilp32", "-Xclang", "-target-feature", "-Xclang", "+vortex",
         "-O3", "-mcmodel=medany", *args.native_flag,
         "-x", "ir", "-c", str(compatible), "-o", str(native)],
        output / "native_compile.log")
    report = {
        "schema": "radiance_llvm_reference_import.v1",
        "status": "native_object_emitted",
        "scope": "LLVM dialect source-equivalence reference; no typed Muon/MX lowering",
        "input_llvm": str(source), "input_sha256": digest(source),
        "mlir_sha256": digest(mlir), "exported_llvm_sha256": digest(exported),
        "native_llvm_sha256": digest(compatible), "native_object_sha256": digest(native),
        "removed_captures_none_attributes": removed,
        "inline_asm_ops": mlir.read_text().count("llvm.inline_asm"),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
