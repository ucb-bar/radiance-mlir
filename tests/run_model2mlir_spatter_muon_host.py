"""Run the captured Spatter Gather read trace through the Muon callback ABI."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from run_model2mlir_spatter_gather_host import (
    SOURCE_REVISION, SUITE, run, sha, write_words,
)

HARNESS = r'''
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int64_t trace_source[262144][1];
int64_t trace_pattern[256];
int64_t trace_output[1024][256][1];
extern void entry(void);
typedef void (*callback_t)(void *, uint32_t, uint32_t, uint32_t);
void mu_schedule(callback_t callback, void *arg, uint32_t occupancy) {
  uint32_t threads = occupancy * 16;
  for (uint32_t block = 0; block < 2; ++block)
    for (uint32_t tid = 0; tid < threads; ++tid)
      callback(arg, tid, threads, block);
}
void muon_mlir_fence(void) { __asm__ __volatile__("" ::: "memory"); }
static void read_words(const char *path, int64_t *dst, size_t count) {
  FILE *file = fopen(path, "rb");
  if (!file) { perror(path); exit(2); }
  if (fread(dst, sizeof(int64_t), count, file) != count || fgetc(file) != EOF) {
    fprintf(stderr, "bad input: %s\n", path); exit(2);
  }
  fclose(file);
}
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  read_words(argv[1], &trace_source[0][0], 262144);
  read_words(argv[2], trace_pattern, 256);
  entry();
  FILE *expected = fopen(argv[3], "rb");
  if (!expected) { perror(argv[3]); return 2; }
  uint64_t digest = UINT64_C(0xcbf29ce484222325);
  for (uint32_t i = 0; i < 1024; ++i)
    for (uint32_t j = 0; j < 256; ++j) {
      uint64_t wanted;
      uint64_t actual = (uint64_t)trace_output[i][j][0];
      if (fread(&wanted, sizeof(wanted), 1, expected) != 1 || actual != wanted) {
        fprintf(stderr, "read (%u,%u) differs\n", i, j); return 1;
      }
      if (i == 1023) {
        digest = (digest ^ (uint32_t)actual) * UINT64_C(0x100000001b3);
        digest = (digest ^ (actual >> 32)) * UINT64_C(0x100000001b3);
      }
    }
  if (fgetc(expected) != EOF) return 2;
  fclose(expected);
  printf("{\"transfers\":262144,\"final_digest\":\"%016" PRIx64 "\"}\n",
         digest);
  return 0;
}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--muon-opt", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    if subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                               text=True).strip() != SOURCE_REVISION:
        parser.error("radiance-kernels source differs from pinned baseline")
    capture_dir = args.capture_dir.resolve()
    capture_path = capture_dir / "receipt.json"
    captured = json.loads(capture_path.read_text())
    parallel = capture_dir / "spatter_gather_trace.parallel.mlir"
    if (captured.get("schema") != "radiance_model2mlir_spatter_gather_capture.v1" or
            captured.get("source_revision") != SOURCE_REVISION or
            sha(parallel) != captured.get("parallel_mlir_sha256")):
        parser.error("Spatter captured IR differs from receipt")
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
    if (case["count"], case["length"], case["src_length"]) != (1024, 256, 262144):
        parser.error("source Gather case shape changed")
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
    harness = out / "spatter_muon_host.c"
    harness.write_text(HARNESS)
    lowered = out / "spatter.muon.mlir"
    llvm_mlir = out / "spatter.llvm.mlir"
    llvm_ir = out / "spatter.ll"
    executable = out / "spatter.exe"
    muon_opt = args.muon_opt.resolve()
    llvm_bin = args.llvm_bin.resolve()
    run([str(muon_opt),
         "--outline-forward-to-muon=inputs=trace_source,trace_pattern output=trace_output warps=4",
         "--distribute-scf-parallel-to-muon=blocks=2", "--lower-muon-runtime",
         str(parallel), "-o", str(lowered)], out / "outline.log")
    lowered_text = lowered.read_text()
    if "scf.parallel" in lowered_text or "call @mu_schedule" not in lowered_text:
        raise RuntimeError("Spatter trace did not become a Muon callback")
    run([str(llvm_bin / "mlir-opt"), str(lowered),
         "--expand-strided-metadata", "--lower-affine", "--convert-scf-to-cf",
         "--convert-arith-to-llvm", "--finalize-memref-to-llvm",
         "--convert-func-to-llvm", "--convert-cf-to-llvm",
         "--reconcile-unrealized-casts", "-o", str(llvm_mlir)], out / "convert.log")
    run([str(llvm_bin / "mlir-translate"), "--mlir-to-llvmir", str(llvm_mlir),
         "-o", str(llvm_ir)], out / "translate.log")
    run([str(llvm_bin / "clang"), "-O2", str(llvm_ir), str(harness),
         "-o", str(executable)], out / "clang.log")
    result = json.loads(run([str(executable), str(source_data), str(pattern),
                             str(expected)], out / "run.log"))
    if (result["transfers"] != captured["transfer_words_compared"] or
            result["final_digest"] != captured["source_final_digest"]):
        raise RuntimeError("Muon callback Spatter read trace differs from source")
    receipt = {"schema": "radiance_model2mlir_spatter_muon_host.v1",
               "status": "muon_callback_full_read_trace_host_not_destination_writes",
               "source_revision": SOURCE_REVISION,
               "capture_receipt_sha256": sha(capture_path),
               "model2mlir_revision": captured["model2mlir_revision"],
               "parallel_mlir_sha256": sha(parallel),
               "muon_opt_sha256": sha(muon_opt),
               "lowered_mlir_sha256": sha(lowered),
               "llvm_ir_sha256": sha(llvm_ir),
               "host_executable_sha256": sha(executable), **result}
    destination = out / "receipt.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(destination), **result}))


if __name__ == "__main__":
    main()
