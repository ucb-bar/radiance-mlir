"""Import native Muon LLVM IR into MLIR and emit a native Muon object.

This is a source-equivalence reference path. It retains LLVM dialect ops,
including source inline assembly; it does not claim that Muon or MX ops were
recognized or lowered by their out-of-tree dialects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
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


def legacy_float(match: re.Match[str]) -> str:
    bits = int(match.group(1), 16)
    value = struct.unpack(">f", bits.to_bytes(4, "big"))[0]
    if not math.isfinite(value):
        raise RuntimeError("LLVM 23 f0x nonfinite constant needs explicit payload conversion")
    return "float " + float_hex(value)


def float_hex(value: float) -> str:
    if not math.isfinite(value):
        raise RuntimeError("nonfinite float literal needs explicit payload conversion")
    rounded = struct.unpack(">f", struct.pack(">f", value))[0]
    double_bits = int.from_bytes(struct.pack(">d", rounded), "big")
    return f"0x{double_bits:016X}"


def legacy_float_decimal(match: re.Match[str]) -> str:
    return "float " + float_hex(float(match.group(1)))


def legacy_half(match: re.Match[str]) -> str:
    return "half " + half_hex(float(match.group(1)))


def half_hex(value: float) -> str:
    if not math.isfinite(value):
        raise RuntimeError("LLVM 23 nonfinite half constant needs explicit payload conversion")
    bits = int.from_bytes(struct.pack(">e", value), "big")
    return f"0xH{bits:04X}"


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
    # LLVM 23 prints these conservative attributes; LLVM 18 rejects their
    # syntax. Removing them cannot strengthen an optimization assumption.
    other_capture = re.search(r"\bcaptures\((?!none\))", text)
    if other_capture:
        raise RuntimeError("unsupported LLVM 23 capture attribute; see reference.exported.ll")
    removed = text.count(" captures(none)")
    text = text.replace(" captures(none)", "")
    removed_no_undef = text.count("nocreateundeforpoison")
    text = text.replace("nocreateundeforpoison ", "")
    text, converted_floats = re.subn(r"\bfloat f0x([0-9A-Fa-f]{8})\b",
                                       legacy_float, text)
    sci = r"([+-]?(?:\d+\.\d+|\d+)(?:e[+-]?\d+)?)"
    text, converted_float_decimals = re.subn(r"\bfloat " + sci + r"\b",
                                               legacy_float_decimal, text)
    text, converted_halves = re.subn(
        r"\bhalf " + sci + r"\b",
        legacy_half, text)
    # LLVM arithmetic keeps its type before two operands, so the second
    # literal is separated from `half` by the first SSA operand.
    half_operand = re.compile(r"(,\s*)([+-]?\d+\.\d+e[+-]?\d+)(?![\w.])")
    half_nan = re.compile(r"(?<![\w.])(-?)nan\(0x([0-9A-Fa-f]+)\)")
    float_operand = re.compile(
        r"(,\s*)(f0x[0-9A-Fa-f]{8}|[+-]?\d+\.\d+e[+-]?\d+)(?![\w.])")
    lines = []
    converted_half_operands = 0
    converted_half_nans = 0
    converted_float_operands = 0
    for line in text.splitlines(keepends=True):
        if re.search(r"\b(?:fadd|fsub|fmul|fdiv|frem|fcmp)\b[^\n]*\bhalf\b", line):
            line, count = half_operand.subn(
                lambda match: match.group(1) + half_hex(float(match.group(2))), line)
            converted_half_operands += count
        elif re.search(r"\b(?:fadd|fsub|fmul|fdiv|frem|fcmp)\b[^\n]*\bfloat\b", line):
            def convert_operand(match: re.Match[str]) -> str:
                literal = match.group(2)
                if literal.startswith("f0x"):
                    bits = int(literal[3:], 16)
                    value = struct.unpack(">f", bits.to_bytes(4, "big"))[0]
                else:
                    value = float(literal)
                return match.group(1) + float_hex(value)
            line, count = float_operand.subn(convert_operand, line)
            converted_float_operands += count
        elif " phi float " in line:
            phi_literal = re.compile(r"(\[\s*)([+-]?\d+\.\d+e[+-]?\d+)(\s*,)")
            line, count = phi_literal.subn(
                lambda match: match.group(1) + float_hex(float(match.group(2)))
                + match.group(3), line)
            converted_float_operands += count
        if re.search(r"\bhalf\b", line) and half_nan.search(line):
            def convert_nan(match: re.Match[str]) -> str:
                payload = int(match.group(2), 16)
                if payload > 0x1FF:
                    raise RuntimeError("half NaN payload does not fit LLVM 18")
                bits = (0x8000 if match.group(1) else 0) | 0x7E00 | payload
                return f"0xH{bits:04X}"
            line, count = half_nan.subn(convert_nan, line)
            converted_half_nans += count
        lines.append(line)
    text = "".join(lines)
    gep_flags = re.compile(r"(getelementptr(?: inbounds)?) (?:nuw |nusw )+(\()")
    text, removed_gep_flags = gep_flags.subn(r"\1 \2", text)
    # LLVM 23 changed the lifetime intrinsic from (i64 size, ptr) to (ptr).
    # These are optimizer hints with no runtime effect; LLVM 18 must not see
    # the new signature, so discard its calls and declarations.
    lifetime = re.compile(r"@llvm\.lifetime\.(?:start|end)\.p0\(")
    kept_lines = []
    removed_lifetime_lines = 0
    for line in text.splitlines(keepends=True):
        if lifetime.search(line) and re.match(r"\s*(?:(?:tail|musttail|notail) )?call void |declare void ", line):
            removed_lifetime_lines += 1
            continue
        kept_lines.append(line)
    text = "".join(kept_lines)
    if lifetime.search(text):
        raise RuntimeError("unsupported LLVM 23 lifetime intrinsic use")
    compatible.write_text(text)
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
        "removed_no_undef_attributes": removed_no_undef,
        "converted_float_literals": converted_floats,
        "converted_float_decimal_literals": converted_float_decimals,
        "converted_half_literals": converted_halves,
        "converted_half_operand_literals": converted_half_operands,
        "converted_float_operand_literals": converted_float_operands,
        "converted_half_nan_literals": converted_half_nans,
        "removed_gep_no_wrap_flags": removed_gep_flags,
        "removed_lifetime_hint_lines": removed_lifetime_lines,
        "inline_asm_ops": mlir.read_text().count("llvm.inline_asm"),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
