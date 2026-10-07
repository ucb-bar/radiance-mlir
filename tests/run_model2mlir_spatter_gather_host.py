"""Execute the captured Spatter Gather read trace on a host.

This checks every source read and the final dense digest. The generated trace
stores each read separately; it does not implement the Muon destination writes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

SOURCE_REVISION = "a27f6abd24830fdc7999d872d170ab778f1e662e"
SUITE = "kernels/spatter/inputs/standard-suite/basic-tests/gpu-stream.json"
HARNESS = r'''
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { void *allocated; int64_t *aligned; int64_t offset;
                 int64_t sizes[1], strides[1]; } MemRef1D;
typedef struct { void *allocated; int64_t *aligned; int64_t offset;
                 int64_t sizes[2], strides[2]; } MemRef2D;
typedef struct { void *allocated; int64_t *aligned; int64_t offset;
                 int64_t sizes[3], strides[3]; } MemRef3D;
extern void _mlir_ciface_forward(MemRef3D *out, const MemRef2D *source,
                                 const MemRef1D *pattern);
static int64_t *read_words(const char *path, size_t count) {
  FILE *file = fopen(path, "rb");
  if (!file) { perror(path); exit(2); }
  int64_t *array = malloc(count * sizeof(int64_t));
  if (!array || fread(array, sizeof(int64_t), count, file) != count ||
      fgetc(file) != EOF) { fprintf(stderr, "bad input: %s\n", path); exit(2); }
  fclose(file); return array;
}
int main(int argc, char **argv) {
  if (argc != 7) return 2;
  int64_t count = strtoll(argv[1], NULL, 10);
  int64_t length = strtoll(argv[2], NULL, 10);
  int64_t source_length = strtoll(argv[3], NULL, 10);
  int64_t *source = read_words(argv[4], (size_t)source_length);
  int64_t *pattern = read_words(argv[5], (size_t)length);
  int64_t *expected = read_words(argv[6], (size_t)(count * length));
  MemRef2D src = {source, source, 0, {source_length, 1}, {1, 1}};
  MemRef1D pat = {pattern, pattern, 0, {length}, {1}};
  MemRef3D result = {0};
  _mlir_ciface_forward(&result, &src, &pat);
  if (!result.aligned || result.sizes[0] != count ||
      result.sizes[1] != length || result.sizes[2] != 1) {
    fprintf(stderr, "unexpected result descriptor\n"); return 1;
  }
  uint64_t digest = UINT64_C(0xcbf29ce484222325);
  for (int64_t i = 0; i < count; ++i)
    for (int64_t j = 0; j < length; ++j) {
      uint64_t observed = (uint64_t)result.aligned[result.offset +
          i * result.strides[0] + j * result.strides[1]];
      uint64_t wanted = (uint64_t)expected[i * length + j];
      if (observed != wanted) {
        fprintf(stderr, "transfer (%" PRId64 ",%" PRId64 ") differs\n", i, j);
        return 1;
      }
      if (i == count - 1) {
        digest = (digest ^ (uint32_t)observed) * UINT64_C(0x100000001b3);
        digest = (digest ^ (observed >> 32)) * UINT64_C(0x100000001b3);
      }
    }
  printf("{\"transfers\":%" PRId64 ",\"final_digest\":\"%016" PRIx64 "\"}\n",
         count * length, digest);
  free(result.allocated); free(source); free(pattern); free(expected);
  return 0;
}
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed(value: int) -> int:
    return value if value < (1 << 63) else value - (1 << 64)


def run(argv: list[str], log: Path) -> str:
    result = subprocess.run(argv, capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed; see {log}")
    return result.stdout


def write_words(path: Path, values) -> None:
    with path.open("wb") as stream:
        chunk = []
        for value in values:
            chunk.append(signed(value))
            if len(chunk) == 32768:
                stream.write(struct.pack(f"<{len(chunk)}q", *chunk))
                chunk.clear()
        if chunk:
            stream.write(struct.pack(f"<{len(chunk)}q", *chunk))


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
    parallel = capture / "spatter_gather_trace.parallel.mlir"
    if (captured.get("schema") != "radiance_model2mlir_spatter_gather_capture.v1" or
            captured.get("source_revision") != SOURCE_REVISION or
            sha(parallel) != captured.get("parallel_mlir_sha256")):
        raise RuntimeError("Spatter parallel IR differs from pinned capture")
    spatter = source_root / "kernels/spatter"
    sys.path.insert(0, str(spatter))
    try:
        spec = importlib.util.spec_from_file_location("radiance_spatter_source", spatter / "run.py")
        assert spec and spec.loader
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
    finally:
        sys.path.pop(0)
    case = reference.normalize(json.loads((source_root / SUITE).read_text())[0])
    if (case["count"], case["length"], case["wrap"], case["delta"]) != (
            captured["count"], captured["length"], captured["wrap"], captured["delta"]):
        raise RuntimeError("source case differs from capture")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_data = out / "source.bin"
    pattern = out / "pattern.bin"
    expected = out / "expected_trace.bin"
    tag = case["payload_tag"]
    write_words(source_data, (reference.payload(tag, i) for i in range(case["src_length"])))
    write_words(pattern, case["pattern"])
    write_words(expected, (reference.payload(tag, reference.source_index(case, i, j))
                           for i in range(case["count"]) for j in range(case["length"])))
    harness = out / "spatter_trace_check.c"
    harness.write_text(HARNESS)
    llvm_bin = args.llvm_bin.resolve()
    llvm_mlir = out / "spatter.llvm.mlir"
    llvm_ir = out / "spatter.ll"
    executable = out / "spatter.exe"
    run([str(llvm_bin / "mlir-opt"), str(parallel), "--llvm-request-c-wrappers",
         "--expand-strided-metadata", "--lower-affine",
         "--convert-scf-to-cf", "--convert-arith-to-llvm",
         "--finalize-memref-to-llvm", "--convert-func-to-llvm",
         "--convert-cf-to-llvm", "--reconcile-unrealized-casts",
         "-o", str(llvm_mlir)], out / "convert.log")
    run([str(llvm_bin / "mlir-translate"), "--mlir-to-llvmir", str(llvm_mlir),
         "-o", str(llvm_ir)], out / "translate.log")
    run([str(llvm_bin / "clang"), "-O2", str(llvm_ir), str(harness),
         "-o", str(executable)], out / "clang.log")
    result = json.loads(run([str(executable), str(case["count"]), str(case["length"]),
                             str(case["src_length"]), str(source_data), str(pattern),
                             str(expected)], out / "run.log"))
    if (result["transfers"] != captured["transfer_words_compared"] or
            result["final_digest"] != captured["source_final_digest"]):
        raise RuntimeError("compiled Spatter trace differs from source")
    receipt = {"schema": "radiance_model2mlir_spatter_gather_host_execution.v1",
               "status": "full_read_trace_host_execution_not_muon_writes",
               "source_revision": SOURCE_REVISION,
               "capture_receipt_sha256": sha(capture_receipt),
               "model2mlir_revision": captured["model2mlir_revision"],
               "parallel_mlir_sha256": sha(parallel),
               "llvm_ir_sha256": sha(llvm_ir),
               "host_executable_sha256": sha(executable), **result}
    destination = out / "receipt.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(destination), **result}))


if __name__ == "__main__":
    main()
