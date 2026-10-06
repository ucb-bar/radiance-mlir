"""Execute model2MLIR-derived SIMT GEMM on a host against source golden bits."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

SOURCE_REVISION = "a27f6abd24830fdc7999d872d170ab778f1e662e"
M = K = N = 64
HARNESS = r'''
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  void *allocated;
  uint16_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2D;
extern void _mlir_ciface_forward(MemRef2D *result, const MemRef2D *a,
                                 const MemRef2D *b);
static uint16_t *read_words(const char *name, int count) {
  FILE *file = fopen(name, "rb");
  if (!file) { perror(name); exit(2); }
  uint16_t *words = malloc((size_t)count * sizeof(uint16_t));
  if (!words || fread(words, sizeof(uint16_t), (size_t)count, file) != (size_t)count ||
      fgetc(file) != EOF) { fprintf(stderr, "bad input: %s\n", name); exit(2); }
  fclose(file);
  return words;
}
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  uint16_t *a = read_words(argv[1], 4096), *b = read_words(argv[2], 4096);
  uint16_t *expected = read_words(argv[3], 4096);
  MemRef2D aa = {a, a, 0, {64, 64}, {64, 1}};
  MemRef2D bb = {b, b, 0, {64, 64}, {64, 1}};
  MemRef2D result = {0};
  _mlir_ciface_forward(&result, &aa, &bb);
  if (result.sizes[0] != 64 || result.sizes[1] != 64 || !result.aligned) {
    fprintf(stderr, "unexpected result descriptor\n"); return 1;
  }
  int mismatches = 0;
  for (int i = 0; i < 64; ++i)
    for (int j = 0; j < 64; ++j) {
      uint16_t observed = result.aligned[result.offset + i * result.strides[0] +
                                          j * result.strides[1]];
      if (observed != expected[i * 64 + j]) {
        if (mismatches < 8)
          fprintf(stderr, "[%d,%d] %04x expected %04x\n", i, j,
                  observed, expected[i * 64 + j]);
        ++mismatches;
      }
    }
  printf("{\"words\":4096,\"mismatches\":%d}\n", mismatches);
  free(result.allocated); free(a); free(b); free(expected);
  return mismatches ? 1 : 0;
}
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], log: Path) -> str:
    result = subprocess.run(argv, capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed; see {log}")
    return result.stdout


def words(path: Path, name: str) -> list[int]:
    match = re.search(rf"(?:const|__global) uint16_t {name}\[\] = \{{(.*?)\}};",
                      path.read_text(), re.S)
    if not match:
        raise RuntimeError(f"{name} is absent from {path}")
    result = [int(word, 16) for word in re.findall(r"0x([0-9a-fA-F]+)", match.group(1))]
    if len(result) != M * N:
        raise RuntimeError(f"{name} has {len(result)} words, expected {M * N}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    if subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                               text=True).strip() != SOURCE_REVISION:
        parser.error("radiance-kernels source differs from the pinned baseline")
    capture = args.capture_dir.resolve()
    capture_receipt = capture / "receipt.json"
    captured = json.loads(capture_receipt.read_text())
    source_ir = capture / "gemm_simt.model2mlir.mlir"
    if (captured.get("schema") != "radiance_model2mlir_gemm_capture.v1" or
            captured.get("source_revision") != SOURCE_REVISION or
            sha(source_ir) != captured.get("captured_mlir_sha256")):
        raise RuntimeError("MLIR differs from pinned model2MLIR capture")
    reference = capture / "reference_data"
    if sha(reference / "expected") != captured["source_expected_sha256"]:
        raise RuntimeError("source golden differs from capture receipt")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for name, reference_path in (("A_raw", reference / "data"),
                                 ("B_raw", reference / "data"),
                                 ("expected_C_raw", reference / "expected")):
        path = out / f"{name}.bin"
        path.write_bytes(struct.pack("<4096H", *words(reference_path, name)))
        inputs[name] = path
    harness = out / "gemm_host_check.c"
    harness.write_text(HARNESS)
    llvm_bin = args.llvm_bin.resolve()
    llvm_mlir = out / "gemm.llvm.mlir"
    llvm_ir = out / "gemm.ll"
    executable = out / "gemm.exe"
    run([str(llvm_bin / "mlir-opt"), str(source_ir),
         "--one-shot-bufferize=bufferize-function-boundaries",
         "--convert-linalg-to-parallel-loops", "--llvm-request-c-wrappers",
         "--convert-scf-to-cf", "--convert-arith-to-llvm",
         "--finalize-memref-to-llvm", "--convert-func-to-llvm",
         "--convert-cf-to-llvm", "--reconcile-unrealized-casts",
         "-o", str(llvm_mlir)], out / "convert.log")
    run([str(llvm_bin / "mlir-translate"), "--mlir-to-llvmir",
         str(llvm_mlir), "-o", str(llvm_ir)], out / "translate.log")
    run([str(llvm_bin / "clang"), "-O2", str(llvm_ir), str(harness),
         "-o", str(executable)], out / "clang.log")
    result = json.loads(run([str(executable), str(inputs["A_raw"]),
                             str(inputs["B_raw"]), str(inputs["expected_C_raw"])],
                            out / "run.log"))
    if result != {"words": 4096, "mismatches": 0}:
        raise RuntimeError(f"host GEMM differs from source golden: {result}")
    receipt = {"schema": "radiance_model2mlir_gemm_host_execution.v1",
               "status": "full_output_host_execution_not_muon_device",
               "source_revision": SOURCE_REVISION,
               "capture_receipt_sha256": sha(capture_receipt),
               "model2mlir_revision": captured["model2mlir_revision"],
               "source_mlir_sha256": sha(source_ir), "llvm_ir_sha256": sha(llvm_ir),
               "host_executable_sha256": sha(executable),
               "source_golden_sha256": sha(reference / "expected"),
               **result}
    destination = out / "receipt.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(destination), **result}))


if __name__ == "__main__":
    main()
