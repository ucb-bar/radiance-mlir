"""Capture all four source-defined STREAM operations with current model2MLIR.

This checks full PyTorch output against the handwritten source equations and
ingests the resulting typed MLIR. It does not claim native Muon execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SOURCE_REVISION = "a27f6abd24830fdc7999d872d170ab778f1e662e"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_generated_ir(path: Path) -> str:
    normalized = "\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n"
    path.write_text(normalized)
    return normalized


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=1048576)
    parser.add_argument("--muon-opt", type=Path, required=True)
    parser.add_argument("--mx-opt", type=Path, required=True)
    parser.add_argument("--radiance-opt", type=Path, required=True)
    parser.add_argument("--mlir-opt", type=Path, required=True,
                        help="upstream MLIR tool used to fuse captured elementwise ops")
    args = parser.parse_args()
    if args.elements < 2 or args.elements & 1:
        parser.error("--elements must be an even integer of at least two")
    m2m_root = args.model2mlir_root.resolve()
    source_root = args.source_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if git(source_root, "rev-parse", "HEAD") != SOURCE_REVISION:
        parser.error("radiance-kernels source differs from the pinned baseline")
    sys.path.insert(0, str(m2m_root))
    import m2m
    import torch
    from m2m.coverage import opaque_report

    if Path(m2m.__file__).resolve().parents[1] != m2m_root:
        raise RuntimeError("model2MLIR resolved to a different checkout")
    source = source_root / "kernels/stream/run.py"
    spec = importlib.util.spec_from_file_location("radiance_stream_reference", source)
    assert spec and spec.loader
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    inputs = tuple(torch.tensor(
        [reference.input_value(name, index) for index in range(args.elements)],
        dtype=torch.float32) for name in "abc")

    class Stream(torch.nn.Module):
        def __init__(self, kind: str) -> None:
            super().__init__()
            self.kind = kind

        def forward(self, a: torch.Tensor, b: torch.Tensor,
                    c: torch.Tensor) -> torch.Tensor:
            if self.kind == "copy":
                return a.clone()
            if self.kind == "scale":
                return 2.0 * c
            if self.kind == "add":
                return a + b
            return b + 2.0 * c

    cases = []
    for kind in reference.KINDS:
        model = Stream(kind).eval()
        actual = model(*inputs).contiguous().view(torch.int32)
        expected = torch.tensor(
            [reference.float_word(reference.result_value(kind, index))
             for index in range(args.elements)], dtype=torch.int32)
        if not torch.equal(actual, expected):
            mismatch = torch.nonzero(actual != expected).flatten()[0].item()
            raise RuntimeError(f"{kind} differs from source at element {mismatch}")
        result = m2m.convert(model, inputs, backend="fx_importer")
        if not result.ok:
            raise RuntimeError(f"model2MLIR {kind} failed: {result.diagnostics}")
        opaque = opaque_report(result.mlir_text)
        if opaque:
            raise RuntimeError(f"model2MLIR {kind} left opaque calls: {opaque}")
        mlir = out / f"stream_{kind}.model2mlir.mlir"
        mlir.write_text(result.mlir_text)
        fused = out / f"stream_{kind}.fused.mlir"
        optimized = subprocess.run([
            str(args.mlir_opt.resolve()), str(mlir),
            "--linalg-fuse-elementwise-ops", "--canonicalize", "--cse",
            "-o", str(fused)], capture_output=True, text=True)
        (out / f"stream_{kind}.fusion.log").write_text(
            optimized.stdout + optimized.stderr)
        if optimized.returncode:
            raise RuntimeError(f"upstream MLIR fusion failed for {kind}")
        fused_text = normalize_generated_ir(fused)
        if kind == "triad" and (fused_text.count("linalg.generic") != 1 or
                                "arith.mulf" not in fused_text or
                                "arith.addf" not in fused_text):
            raise RuntimeError("Triad did not fuse into one multiply-add loop")
        parallel = out / f"stream_{kind}.parallel.mlir"
        bufferized = subprocess.run([
            str(args.mlir_opt.resolve()), str(fused),
            "--one-shot-bufferize=bufferize-function-boundaries",
            "--convert-linalg-to-parallel-loops", "-o", str(parallel)],
            capture_output=True, text=True)
        (out / f"stream_{kind}.bufferization.log").write_text(
            bufferized.stdout + bufferized.stderr)
        if bufferized.returncode:
            raise RuntimeError(f"upstream MLIR bufferization failed for {kind}")
        parallel_text = normalize_generated_ir(parallel)
        if kind != "copy" and "scf.parallel" not in parallel_text:
            raise RuntimeError(f"{kind} did not lower to an explicit parallel loop")
        for stage, artifact in (("captured", mlir), ("fused", fused),
                                ("parallel", parallel)):
            for name, tool in (("muon", args.muon_opt), ("mx", args.mx_opt),
                               ("radiance", args.radiance_opt)):
                parsed = subprocess.run([str(tool.resolve()), str(artifact),
                                         "-o", "/dev/null"],
                                        capture_output=True, text=True)
                (out / f"stream_{kind}.{stage}.{name}.log").write_text(
                    parsed.stdout + parsed.stderr)
                if parsed.returncode:
                    raise RuntimeError(f"{name} rejected {kind} {stage}; see parse log")
        cases.append({"kind": kind, "elements_compared": args.elements,
                      "expected_digest": f"{reference.digest(expected.tolist()):016x}",
                      "captured_mlir_sha256": sha(mlir), "opaque_calls": opaque,
                      "fused_mlir_sha256": sha(fused),
                      "parallel_mlir_sha256": sha(parallel),
                      "model2mlir_path": result.path_taken})
    tree = {str(path.relative_to(m2m_root)): sha(path)
            for path in sorted((m2m_root / "m2m").rglob("*.py"))}
    receipt = {"schema": "radiance_model2mlir_stream_capture.v1",
               "status": "captured_and_parsed_not_target_lowered",
               "source_revision": SOURCE_REVISION,
               "source_run_sha256": sha(source),
               "model2mlir_revision": git(m2m_root, "rev-parse", "HEAD"),
               "model2mlir_source_tree_sha256": hashlib.sha256(
                   json.dumps(tree, sort_keys=True).encode()).hexdigest(),
               "model2mlir_worktree_status": git(m2m_root, "status", "--short").splitlines(),
               "mlir_opt_sha256": sha(args.mlir_opt.resolve()),
               "fusion_pipeline": ["linalg-fuse-elementwise-ops", "canonicalize", "cse"],
               "loop_pipeline": ["one-shot-bufferize=bufferize-function-boundaries",
                                 "convert-linalg-to-parallel-loops"],
               "cases": cases}
    destination = out / "receipt.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "elements_per_case": args.elements,
                      "receipt": str(destination)}))


if __name__ == "__main__":
    main()
