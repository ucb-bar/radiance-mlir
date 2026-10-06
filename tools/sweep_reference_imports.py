"""Attempt source-to-MLIR reference imports for every default Radiance target.

Make is queried with -n; no recipe is run in radiance-kernels. All compiler
outputs and logs are written beneath --out. Failed targets remain in the
manifest with the stage and diagnostic, rather than disappearing from counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_provenance(root: Path) -> dict:
    """Record local generator products as well as the pinned Git revision."""
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                            check=True, capture_output=True, text=True).stdout
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=root,
                          check=True, capture_output=True).stdout
    patterns = ("data", "data.S", "a_data", "b_data",
                "generated/config.h", "include/fa_data.h", "mxgemm.data.*.h")
    generated = sorted({path for directory in (root / "kernels").iterdir()
                        if directory.is_dir() for pattern in patterns
                        for path in directory.glob(pattern) if path.is_file()})
    return {
        "git_status_porcelain": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "generated_input_sha256": {
            str(path.relative_to(root)): sha256(path) for path in generated
        },
    }


def run(command: list[str], cwd: Path, log: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(command, cwd=cwd, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        log.write_text(output + f"\nTimed out after {timeout} seconds\n")
        raise RuntimeError("timeout") from error
    log.write_text(result.stdout)
    return result


def compile_recipe(directory: Path, target: str, args: argparse.Namespace,
                   log: Path) -> list[str]:
    result = run(["make", "-Bn", target + ".mu.o",
                  f"LLVM_MUON={args.muon_clang.parent.parent}",
                  f"RADIANCE_LIB_PATH={args.source_root / 'lib'}",
                  f"MU_LIBC_INCLUDE={args.libc_include}"], directory, log)
    if result.returncode:
        raise RuntimeError(f"make dry run failed ({result.returncode})")
    for line in reversed(result.stdout.splitlines()):
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if (tokens and tokens[0].endswith(("/clang", "/clang++"))
                and "-c" in tokens and "-o" in tokens
                and tokens[tokens.index("-o") + 1] == target + ".mu.o"):
            return tokens
    raise RuntimeError("no native compile recipe for target")


def attempt(item: tuple[str, str], args: argparse.Namespace) -> dict:
    family, target = item
    directory = args.source_root / "kernels" / family
    output = args.out / family / target
    output.mkdir(parents=True, exist_ok=True)
    record = {"family": family, "target": target, "status": "make_recipe_failed"}
    try:
        tokens = compile_recipe(directory, target, args, output / "make_dry_run.log")
        record["status"] = "source_compile_failed"
        # The original command is retained in make_dry_run.log. The added
        # paths supply libc++'s generated header and the pinned MX submodule.
        old_output = tokens.index("-o")
        del tokens[old_output:old_output + 2]
        old_compile = tokens.index("-c")
        tokens[old_compile:old_compile + 1] = ["-S", "-emit-llvm"]
        tokens[0] = str(args.muon_clang)
        filtered = []
        skip = False
        for index, token in enumerate(tokens):
            if skip:
                skip = False
                continue
            if (token == "-mllvm" and index + 1 < len(tokens)
                    and tokens[index + 1].startswith("-riscv-stack-word-stride=")):
                # This Muon LLVM 18 build predates the target stack option;
                # it does not affect LLVM IR emission. Native object checks
                # must compare both paths using this same compiler build.
                skip = True
                continue
            if token.startswith(("--sysroot=", "--gcc-toolchain=")) or token == "-nodefaultlibs":
                continue
            filtered.append(token)
        tokens = filtered
        tokens += ["-target", "riscv32-unknown-elf",
                   "-isystem", str(args.assertion_include),
                   "-isystem", str(args.libcxx_include),
                   "-isystem", str(args.config_site_dir),
                   "-I", str(args.mx_submodule),
                   "-o", str(output / "source.ll")]
        result = run(tokens, directory, output / "source_compile.log")
        if result.returncode:
            raise RuntimeError(f"source compile failed ({result.returncode})")
        record["status"] = "reference_import_failed"
        importer = Path(__file__).with_name("import_llvm_reference.py")
        result = run([sys.executable, str(importer),
                      "--input-llvm", str(output / "source.ll"),
                      "--out", str(output / "import"),
                      "--mlir-translate", str(args.mlir_translate),
                      "--muon-clang", str(args.muon_clang)],
                     directory, output / "import_driver.log")
        if result.returncode:
            raise RuntimeError(f"LLVM dialect import or native compile failed ({result.returncode})")
        record["status"] = "native_object_emitted"
        report = json.loads((output / "import/report.json").read_text())
        record["llvm_sha256"] = report["input_sha256"]
        record["mlir_sha256"] = report["mlir_sha256"]
        record["object_sha256"] = report["native_object_sha256"]
        record["inline_asm_ops"] = report["inline_asm_ops"]
    except RuntimeError as error:
        record["error"] = str(error)
        log_name = {"make_recipe_failed": "make_dry_run.log",
                    "source_compile_failed": "source_compile.log",
                    "reference_import_failed": "import_driver.log"}.get(record["status"])
        if log_name and (output / log_name).exists():
            lines = (output / log_name).read_text(errors="replace").splitlines()
            diagnostic = next((line.strip() for line in lines
                               if "fatal error:" in line or "error:" in line
                               or "No rule to make target" in line), None)
            if diagnostic:
                record["diagnostic"] = diagnostic[:500]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--muon-clang", type=Path, required=True)
    parser.add_argument("--mlir-translate", type=Path, required=True)
    parser.add_argument("--libcxx-include", type=Path, required=True)
    parser.add_argument("--libc-include", type=Path, required=True)
    parser.add_argument("--config-site-dir", type=Path, required=True)
    parser.add_argument("--mx-submodule", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    for field in ("inventory", "source_root", "out", "muon_clang",
                  "mlir_translate", "libcxx_include", "libc_include",
                  "config_site_dir", "mx_submodule"):
        setattr(args, field, getattr(args, field).resolve())
    inventory = json.loads(args.inventory.read_text())
    if inventory["schema"] != "muon_mlir_kernel_inventory.v2":
        parser.error("inventory must use v2 schema")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.source_root,
                              check=True, capture_output=True, text=True).stdout.strip()
    if revision != inventory["source_git_revision"]:
        parser.error("source revision differs from inventory")
    args.out.mkdir(parents=True, exist_ok=True)
    args.assertion_include = args.out / "include"
    args.assertion_include.mkdir(exist_ok=True)
    handler = args.libcxx_include.parent / "vendor/llvm/default_assertion_handler.in"
    (args.assertion_include / "__assertion_handler").write_bytes(handler.read_bytes())
    targets = [(family["name"], name.removesuffix(".radiance.elf"))
               for family in inventory["families"]
               for name in family["build_targets"].get("radiance_elfs", [])]
    if args.limit is not None:
        targets = targets[:args.limit]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda item: attempt(item, args), targets))
    document = {"schema": "radiance_reference_sweep.v1",
                "source_git_revision": revision,
                "source_provenance": source_provenance(args.source_root),
                "scope": "LLVM dialect reference import; no typed Muon/MX lowering",
                "inventory_target_count": inventory["counts"]["default_radiance_elfs"],
                "attempted_target_count": len(results), "results": results}
    (args.out / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    counts = {status: sum(row["status"] == status for row in results)
              for status in sorted({row["status"] for row in results})}
    print(json.dumps({"attempted": len(results), "status_counts": counts,
                      "manifest": str(args.out / "manifest.json")}))


if __name__ == "__main__":
    main()
