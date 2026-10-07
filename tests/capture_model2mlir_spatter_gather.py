"""Capture the original Spatter Gather transfer trace with current model2MLIR.

The PyTorch model performs every source read. It materializes a trace tensor
rather than reproducing repeated destination writes, so it is a frontend
coverage check and not an executable replacement for the Muon kernel.
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
SUITE = "kernels/spatter/inputs/standard-suite/basic-tests/gpu-stream.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_generated_ir(path: Path) -> str:
    normalized = "\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n"
    path.write_text(normalized)
    return normalized


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def signed(value: int) -> int:
    return value if value < (1 << 63) else value - (1 << 64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--muon-opt", type=Path, required=True)
    parser.add_argument("--mx-opt", type=Path, required=True)
    parser.add_argument("--radiance-opt", type=Path, required=True)
    parser.add_argument("--mlir-opt", type=Path, required=True)
    args = parser.parse_args()
    m2m_root = args.model2mlir_root.resolve()
    source_root = args.source_root.resolve()
    if git(source_root, "rev-parse", "HEAD") != SOURCE_REVISION:
        parser.error("radiance-kernels source differs from the pinned baseline")
    spatter = source_root / "kernels/spatter"
    source_run = spatter / "run.py"
    source_plan = spatter / "plan.py"
    suite_path = source_root / SUITE
    suite = json.loads(suite_path.read_text())
    sys.path.insert(0, str(spatter))
    try:
        spec = importlib.util.spec_from_file_location("radiance_spatter_source", source_run)
        assert spec and spec.loader
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
    finally:
        sys.path.pop(0)
    case = reference.normalize(suite[0])
    if case["kind"] != "gather" or case["wrap"] != 1:
        raise RuntimeError("selected source case is no longer single-wrap Gather")
    sys.path.insert(0, str(m2m_root))
    import m2m
    import torch
    from m2m.coverage import opaque_report

    if Path(m2m.__file__).resolve().parents[1] != m2m_root:
        raise RuntimeError("model2MLIR resolved to a different checkout")
    tag = case["payload_tag"]
    source_unsigned = [reference.payload(tag, i) for i in range(case["src_length"])]
    source = torch.tensor([signed(value) for value in source_unsigned],
                          dtype=torch.int64).reshape(-1, 1)
    pattern = torch.tensor(case["pattern"], dtype=torch.int64)

    class GatherTrace(torch.nn.Module):
        def __init__(self, count: int, delta: int) -> None:
            super().__init__()
            self.count = count
            self.delta = delta

        def forward(self, sparse: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            iteration = torch.arange(self.count, dtype=torch.int64)
            reads = indices.unsqueeze(0) + self.delta * iteration.unsqueeze(1)
            return torch.nn.functional.embedding(reads, sparse)

    model = GatherTrace(case["count"], case["delta"]).eval()
    observed = model(source, pattern).reshape(case["count"], case["length"])
    if observed.numel() != case["count"] * case["length"]:
        raise RuntimeError("PyTorch model did not represent every source transfer")
    for iteration in range(case["count"]):
        for j in range(case["length"]):
            index = reference.source_index(case, iteration, j)
            if observed[iteration, j].item() != signed(source_unsigned[index]):
                raise RuntimeError(f"source transfer differs at ({iteration}, {j})")
    final = [value & ((1 << 64) - 1) for value in observed[-1].tolist()]
    digest, overlap = reference.reference(case)
    if overlap or reference.fnv(final) != digest:
        raise RuntimeError("final dense output differs from source reference")
    result = m2m.convert(model, (source, pattern), backend="fx_importer")
    if not result.ok:
        raise RuntimeError(f"model2MLIR failed: {result.diagnostics}")
    opaque = opaque_report(result.mlir_text)
    if opaque or "tensor.extract" not in result.mlir_text:
        raise RuntimeError(f"Spatter Gather capture is incomplete: {opaque}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mlir = out / "spatter_gather_trace.model2mlir.mlir"
    mlir.write_text(result.mlir_text)
    fused = out / "spatter_gather_trace.fused.mlir"
    optimized = subprocess.run([
        str(args.mlir_opt.resolve()), str(mlir),
        "--linalg-fuse-elementwise-ops", "--canonicalize", "--cse",
        "-o", str(fused)], capture_output=True, text=True)
    (out / "fusion.log").write_text(optimized.stdout + optimized.stderr)
    if optimized.returncode:
        raise RuntimeError("upstream MLIR fusion failed for Spatter Gather")
    fused_text = normalize_generated_ir(fused)
    if (fused_text.count("linalg.generic") != 1 or "tensor.extract" not in fused_text or
            "arith.muli" not in fused_text or "arith.addi" not in fused_text):
        raise RuntimeError("source Gather address calculation did not fuse")
    parallel = out / "spatter_gather_trace.parallel.mlir"
    bufferized = subprocess.run([
        str(args.mlir_opt.resolve()), str(fused),
        "--one-shot-bufferize=bufferize-function-boundaries",
        "--convert-linalg-to-parallel-loops", "-o", str(parallel)],
        capture_output=True, text=True)
    (out / "bufferization.log").write_text(bufferized.stdout + bufferized.stderr)
    if bufferized.returncode:
        raise RuntimeError("upstream MLIR bufferization failed for Spatter Gather")
    parallel_text = normalize_generated_ir(parallel)
    if parallel_text.count("scf.parallel") != 1:
        raise RuntimeError("Spatter Gather did not lower to one parallel trace loop")
    for stage, artifact in (("captured", mlir), ("fused", fused),
                            ("parallel", parallel)):
        for name, tool in (("muon", args.muon_opt), ("mx", args.mx_opt),
                           ("radiance", args.radiance_opt)):
            parsed = subprocess.run([str(tool.resolve()), str(artifact),
                                     "-o", "/dev/null"],
                                    capture_output=True, text=True)
            (out / f"{stage}_{name}_parse.log").write_text(
                parsed.stdout + parsed.stderr)
            if parsed.returncode:
                raise RuntimeError(f"{name} rejected {stage} Spatter trace")
    tree = {str(path.relative_to(m2m_root)): sha(path)
            for path in sorted((m2m_root / "m2m").rglob("*.py"))}
    receipt = {"schema": "radiance_model2mlir_spatter_gather_capture.v1",
               "status": "full_transfer_trace_frontend_not_destination_write_lowering",
               "source_revision": SOURCE_REVISION, "source_suite_sha256": sha(suite_path),
               "source_run_sha256": sha(source_run), "source_plan_sha256": sha(source_plan),
               "source_case": 0, "count": case["count"], "length": case["length"],
               "wrap": case["wrap"], "delta": case["delta"],
               "transfer_words_compared": observed.numel(),
               "final_output_words_compared": len(final),
               "source_final_digest": f"{digest:016x}",
               "model2mlir_revision": git(m2m_root, "rev-parse", "HEAD"),
               "model2mlir_source_tree_sha256": hashlib.sha256(
                   json.dumps(tree, sort_keys=True).encode()).hexdigest(),
               "model2mlir_worktree_status": git(m2m_root, "status", "--short").splitlines(),
               "captured_mlir_sha256": sha(mlir), "opaque_calls": opaque,
               "fused_mlir_sha256": sha(fused), "parallel_mlir_sha256": sha(parallel),
               "mlir_opt_sha256": sha(args.mlir_opt.resolve()),
               "model2mlir_path": result.path_taken}
    destination = out / "receipt.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"transfers": observed.numel(), "final_digest": receipt["source_final_digest"],
                      "receipt": str(destination)}))


if __name__ == "__main__":
    main()
