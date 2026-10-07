"""Run source-captured SIMT GEMM through staged Muon callbacks on a host."""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

from run_model2mlir_gemm_host import SOURCE_REVISION, run, sha, words

HARNESS = r'''
#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint16_t gemm_a[64][64], gemm_b[64][64], gemm_c[64][64];
extern void entry(void);
typedef void (*callback_t)(void *, uint32_t, uint32_t, uint32_t);
static pthread_barrier_t stage_barrier;
typedef struct { callback_t callback; void *arg; uint32_t tid, threads; } Work;
static void *worker(void *opaque) {
  Work *work = (Work *)opaque;
  work->callback(work->arg, work->tid, work->threads, 0);
  return NULL;
}
void muon_mlir_fence(void) { atomic_thread_fence(memory_order_seq_cst); }
void muon_mlir_barrier(uint32_t id, uint32_t warps) {
  if (id != 0 || warps != 4) abort();
  pthread_barrier_wait(&stage_barrier);
}
void mu_schedule(callback_t callback, void *arg, uint32_t occupancy) {
  uint32_t threads = occupancy * 16;
  if (threads != 64 || pthread_barrier_init(&stage_barrier, NULL, threads)) abort();
  pthread_t handles[64]; Work work[64];
  for (uint32_t tid = 0; tid < threads; ++tid) {
    work[tid] = (Work){callback, arg, tid, threads};
    if (pthread_create(&handles[tid], NULL, worker, &work[tid])) abort();
  }
  for (uint32_t tid = 0; tid < threads; ++tid)
    if (pthread_join(handles[tid], NULL)) abort();
  pthread_barrier_destroy(&stage_barrier);
}
static void read_words(const char *path, uint16_t *dst) {
  FILE *file = fopen(path, "rb");
  if (!file) { perror(path); exit(2); }
  if (fread(dst, sizeof(uint16_t), 4096, file) != 4096 || fgetc(file) != EOF) {
    fprintf(stderr, "bad input: %s\n", path); exit(2);
  }
  fclose(file);
}
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  read_words(argv[1], &gemm_a[0][0]);
  read_words(argv[2], &gemm_b[0][0]);
  entry();
  uint16_t expected[4096];
  read_words(argv[3], expected);
  for (uint32_t i = 0; i < 4096; ++i)
    if (((uint16_t *)gemm_c)[i] != expected[i]) {
      fprintf(stderr, "GEMM word %u: %04x expected %04x\n",
              i, ((uint16_t *)gemm_c)[i], expected[i]); return 1;
    }
  puts("{\"words\":4096,\"mismatches\":0}");
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
    source_ir = capture_dir / "gemm_simt.model2mlir.mlir"
    if (captured.get("schema") != "radiance_model2mlir_gemm_capture.v1" or
            captured.get("source_revision") != SOURCE_REVISION or
            sha(source_ir) != captured.get("captured_mlir_sha256")):
        parser.error("SIMT GEMM capture does not match receipt")
    reference = capture_dir / "reference_data"
    if sha(reference / "expected") != captured["source_expected_sha256"]:
        parser.error("source golden differs from capture receipt")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, source_file in (("A_raw", reference / "data"),
                              ("B_raw", reference / "data"),
                              ("expected_C_raw", reference / "expected")):
        (out / f"{name}.bin").write_bytes(struct.pack("<4096H", *words(source_file, name)))
    harness = out / "gemm_muon_host.c"
    harness.write_text(HARNESS)
    parallel = out / "gemm.parallel.mlir"
    lowered = out / "gemm.muon.mlir"
    llvm_mlir = out / "gemm.llvm.mlir"
    llvm_ir = out / "gemm.ll"
    executable = out / "gemm.exe"
    llvm_bin = args.llvm_bin.resolve()
    muon_opt = args.muon_opt.resolve()
    run([str(llvm_bin / "mlir-opt"), str(source_ir),
         "--one-shot-bufferize=bufferize-function-boundaries",
         "--convert-linalg-to-parallel-loops", "-o", str(parallel)],
        out / "bufferize.log")
    if parallel.read_text().count("scf.parallel") != 5:
        raise RuntimeError("captured GEMM did not lower to the expected five stages")
    run([str(muon_opt),
         "--outline-forward-to-muon=inputs=gemm_a,gemm_b output=gemm_c warps=4 shared-scratch=true",
         "--distribute-scf-parallel-to-muon=blocks=1", "--lower-muon-runtime",
         str(parallel), "-o", str(lowered)], out / "outline.log")
    lowered_text = lowered.read_text()
    if ("scf.parallel" in lowered_text or
            lowered_text.count("call @muon_mlir_barrier") != 5 or
            "call @mu_schedule" not in lowered_text):
        raise RuntimeError("GEMM stages did not become synchronized Muon callbacks")
    run([str(llvm_bin / "mlir-opt"), str(lowered),
         "--expand-strided-metadata", "--lower-affine", "--convert-scf-to-cf",
         "--convert-arith-to-llvm", "--finalize-memref-to-llvm",
         "--convert-func-to-llvm", "--convert-cf-to-llvm",
         "--reconcile-unrealized-casts", "-o", str(llvm_mlir)], out / "convert.log")
    run([str(llvm_bin / "mlir-translate"), "--mlir-to-llvmir", str(llvm_mlir),
         "-o", str(llvm_ir)], out / "translate.log")
    run([str(llvm_bin / "clang"), "-O2", "-pthread", str(llvm_ir), str(harness),
         "-o", str(executable)], out / "clang.log")
    result = json.loads(run([str(executable), str(out / "A_raw.bin"),
                             str(out / "B_raw.bin"), str(out / "expected_C_raw.bin")],
                            out / "run.log"))
    if result != {"words": 4096, "mismatches": 0}:
        raise RuntimeError("Muon callback GEMM differs from source golden")
    receipt = {"schema": "radiance_model2mlir_gemm_muon_host.v1",
               "status": "muon_callback_full_output_host_not_device",
               "source_revision": SOURCE_REVISION,
               "capture_receipt_sha256": sha(capture_path),
               "model2mlir_revision": captured["model2mlir_revision"],
               "source_golden_sha256": sha(reference / "expected"),
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
