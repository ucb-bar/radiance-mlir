"""Execute model2MLIR-derived STREAM loops on a host against source data.

This exercises captured and upstream-lowered MLIR, not Muon device code.
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
HARNESS = r'''
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  void *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRef1D;

extern void _mlir_ciface_forward(MemRef1D *result, const MemRef1D *a,
                                 const MemRef1D *b, const MemRef1D *c);

static float *read_words(const char *path, int64_t count) {
  FILE *file = fopen(path, "rb");
  if (!file) { perror(path); exit(2); }
  float *array = malloc((size_t)count * sizeof(float));
  if (!array || fread(array, sizeof(float), (size_t)count, file) != (size_t)count ||
      fgetc(file) != EOF) { fprintf(stderr, "bad input: %s\n", path); exit(2); }
  fclose(file);
  return array;
}

int main(int argc, char **argv) {
  if (argc != 6) return 2;
  int64_t count = strtoll(argv[1], NULL, 10);
  float *a = read_words(argv[2], count), *b = read_words(argv[3], count);
  float *c = read_words(argv[4], count), *expected = read_words(argv[5], count);
  MemRef1D aa = {a, a, 0, {count}, {1}};
  MemRef1D bb = {b, b, 0, {count}, {1}};
  MemRef1D cc = {c, c, 0, {count}, {1}};
  MemRef1D result = {0};
  _mlir_ciface_forward(&result, &aa, &bb, &cc);
  if (result.sizes[0] != count || result.strides[0] != 1 || !result.aligned) {
    fprintf(stderr, "unexpected result descriptor\n"); return 1;
  }
  uint64_t digest = UINT64_C(0xcbf29ce484222325);
  for (int64_t i = 0; i < count; ++i) {
    uint32_t actual_word, expected_word;
    memcpy(&actual_word, result.aligned + result.offset + i, sizeof(actual_word));
    memcpy(&expected_word, expected + i, sizeof(expected_word));
    if (actual_word != expected_word) {
      fprintf(stderr, "word %" PRId64 ": %08x expected %08x\n",
              i, actual_word, expected_word); return 1;
    }
    digest = (digest ^ actual_word) * UINT64_C(0x100000001b3);
  }
  printf("{\"words\":%" PRId64 ",\"digest\":\"%016" PRIx64 "\"}\n", count, digest);
  if (result.allocated != a && result.allocated != b && result.allocated != c)
    free(result.allocated);
  free(a); free(b); free(c); free(expected);
  return 0;
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


def write_words(path: Path, values) -> None:
    with path.open("wb") as stream:
        batch = []
        for value in values:
            batch.append(value)
            if len(batch) == 32768:
                stream.write(struct.pack(f"<{len(batch)}I", *batch))
                batch.clear()
        if batch:
            stream.write(struct.pack(f"<{len(batch)}I", *batch))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=1048576)
    args = parser.parse_args()
    if args.elements < 2 or args.elements & 1:
        parser.error("--elements must be an even integer of at least two")
    source_root = args.source_root.resolve()
    if subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                               text=True).strip() != SOURCE_REVISION:
        parser.error("radiance-kernels source differs from the pinned baseline")
    source = source_root / "kernels/stream/run.py"
    spec = importlib.util.spec_from_file_location("radiance_stream_reference", source)
    assert spec and spec.loader
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    capture_path = args.parallel_dir.resolve() / "receipt.json"
    capture = json.loads(capture_path.read_text())
    if (capture.get("schema") != "radiance_model2mlir_stream_capture.v1" or
            capture.get("source_revision") != SOURCE_REVISION):
        raise RuntimeError("parallel IR lacks the pinned model2MLIR capture receipt")
    captured_cases = {case["kind"]: case for case in capture["cases"]}
    inputs = {}
    for name in "abc":
        path = out / f"{name}.bin"
        write_words(path, (reference.float_word(reference.input_value(name, i))
                           for i in range(args.elements)))
        inputs[name] = path
    harness = out / "stream_host_check.c"
    harness.write_text(HARNESS)
    llvm_bin = args.llvm_bin.resolve()
    cases = []
    for kind in reference.KINDS:
        source_ir = args.parallel_dir.resolve() / f"stream_{kind}.parallel.mlir"
        if sha(source_ir) != captured_cases[kind]["parallel_mlir_sha256"]:
            raise RuntimeError(f"{kind} parallel IR differs from capture receipt")
        llvm_mlir = out / f"{kind}.llvm.mlir"
        llvm_ir = out / f"{kind}.ll"
        executable = out / f"{kind}.exe"
        expected = out / f"{kind}.expected.bin"
        write_words(expected, (reference.float_word(reference.result_value(kind, i))
                               for i in range(args.elements)))
        run([str(llvm_bin / "mlir-opt"), str(source_ir),
             "--llvm-request-c-wrappers", "--convert-scf-to-cf",
             "--convert-arith-to-llvm", "--finalize-memref-to-llvm",
             "--convert-func-to-llvm", "--convert-cf-to-llvm",
             "--reconcile-unrealized-casts", "-o", str(llvm_mlir)],
            out / f"{kind}.convert.log")
        run([str(llvm_bin / "mlir-translate"), "--mlir-to-llvmir",
             str(llvm_mlir), "-o", str(llvm_ir)], out / f"{kind}.translate.log")
        run([str(llvm_bin / "clang"), "-O2", str(llvm_ir), str(harness),
             "-o", str(executable)], out / f"{kind}.clang.log")
        result = json.loads(run([str(executable), str(args.elements),
                                 *(str(inputs[name]) for name in "abc"),
                                 str(expected)], out / f"{kind}.run.log"))
        if (result["words"] != args.elements or result["digest"] !=
                f"{reference.digest(reference.float_word(reference.result_value(kind, i))
                                    for i in range(args.elements)):016x}"):
            raise RuntimeError(f"{kind} execution digest differs from source")
        cases.append({"kind": kind, "words_compared": result["words"],
                      "digest": result["digest"], "parallel_mlir_sha256": sha(source_ir),
                      "llvm_ir_sha256": sha(llvm_ir), "host_executable_sha256": sha(executable)})
    receipt = {"schema": "radiance_model2mlir_stream_host_execution.v1",
               "status": "full_output_host_execution_not_muon_device",
               "source_revision": SOURCE_REVISION, "source_run_sha256": sha(source),
               "capture_receipt_sha256": sha(capture_path),
               "model2mlir_revision": capture["model2mlir_revision"],
               "model2mlir_source_tree_sha256": capture["model2mlir_source_tree_sha256"],
               "mlir_opt_sha256": sha(llvm_bin / "mlir-opt"),
               "mlir_translate_sha256": sha(llvm_bin / "mlir-translate"),
               "clang_sha256": sha(llvm_bin / "clang"), "harness_sha256": sha(harness),
               "cases": cases}
    destination = out / "receipt.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "words_per_case": args.elements,
                      "receipt": str(destination)}))


if __name__ == "__main__":
    main()
