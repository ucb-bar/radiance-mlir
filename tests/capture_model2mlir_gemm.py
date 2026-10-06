"""Capture the pinned handwritten SIMT GEMM golden through current model2MLIR.

This checks a source-anchored PyTorch frontend and dialect ingestion. It does
not yet claim a Muon scheduling or native lowering for the captured linalg IR.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


SOURCE_REVISION = "a27f6abd24830fdc7999d872d170ab778f1e662e"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def check_parse(tool: Path, source: Path, log: Path) -> None:
    result = subprocess.run([str(tool), str(source), "-o", "/dev/null"],
                            text=True, capture_output=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{tool.name} rejected captured MLIR; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--muon-opt", type=Path, required=True)
    parser.add_argument("--mx-opt", type=Path, required=True)
    parser.add_argument("--radiance-opt", type=Path, required=True)
    args = parser.parse_args()
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
        raise RuntimeError("imported model2MLIR is not the requested local checkout")
    generator = source_root / "kernels/gemm_simt/gemm.py"
    spec = importlib.util.spec_from_file_location("radiance_gemm_generator", generator)
    assert spec and spec.loader
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(reference.SEED)
        a = torch.randn((reference.M, reference.K), dtype=torch.bfloat16)
        b = torch.randn((reference.K, reference.N), dtype=torch.bfloat16)

    class GemmSIMT(torch.nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.matmul(lhs.float(), rhs.float()).to(torch.bfloat16)

    model = GemmSIMT().eval()
    reference_dir = out / "reference_data"
    reference_dir.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(generator)], cwd=reference_dir, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    expected_text = (reference_dir / "expected").read_text()
    match = re.search(r"const uint16_t expected_C_raw\[\] = \{(.*?)\};",
                      expected_text, re.S)
    if not match:
        raise RuntimeError("handwritten generator did not emit expected_C_raw")
    expected = [int(word, 16) for word in re.findall(r"0x([0-9a-fA-F]+)", match.group(1))]
    observed = model(a, b).contiguous().view(torch.uint16).reshape(-1).tolist()
    if observed != expected:
        raise RuntimeError("PyTorch capture model differs from the source-generated full output")
    result = m2m.convert(model, (a, b), backend="fx_importer")
    if not result.ok:
        raise RuntimeError(f"model2MLIR failed: {result.diagnostics}")
    if "linalg.matmul" not in result.mlir_text:
        raise RuntimeError("model2MLIR output lacks the source GEMM contraction")
    opaque = opaque_report(result.mlir_text)
    if opaque:
        raise RuntimeError(f"model2MLIR left opaque calls: {opaque}")
    mlir = out / "gemm_simt.model2mlir.mlir"
    mlir.write_text(result.mlir_text)
    for name, tool in (("muon", args.muon_opt), ("mx", args.mx_opt),
                       ("radiance", args.radiance_opt)):
        check_parse(tool.resolve(), mlir, out / f"{name}_parse.log")
    python_sources = sorted((m2m_root / "m2m").rglob("*.py"))
    tree = {str(path.relative_to(m2m_root)): sha(path) for path in python_sources}
    receipt = {
        "schema": "radiance_model2mlir_gemm_capture.v1",
        "status": "captured_and_parsed_not_target_lowered",
        "source_revision": SOURCE_REVISION,
        "source_generator_sha256": sha(generator),
        "source_expected_sha256": sha(reference_dir / "expected"),
        "model2mlir_revision": git(m2m_root, "rev-parse", "HEAD"),
        "model2mlir_branch": git(m2m_root, "branch", "--show-current"),
        "model2mlir_source_tree_sha256": hashlib.sha256(
            json.dumps(tree, sort_keys=True).encode()).hexdigest(),
        "model2mlir_worktree_status": git(m2m_root, "status", "--short").splitlines(),
        "captured_mlir_sha256": sha(mlir),
        "input_a_sha256": hashlib.sha256(a.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        "input_b_sha256": hashlib.sha256(b.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        "output_words_compared": len(expected),
        "model2mlir_path": result.path_taken,
        "opaque_calls": opaque,
        "parsers": ["muon-opt", "mx-gemmini-opt", "radiance-opt"],
    }
    destination = out / "receipt.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output_words_compared": len(expected),
                      "captured_mlir": str(mlir), "receipt": str(destination)}))


if __name__ == "__main__":
    main()
