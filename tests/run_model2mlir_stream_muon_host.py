"""Execute Muon-outlined model2MLIR STREAM computations on a host.

The inputs are real PyTorch/model2MLIR captures and the expected words come
from radiance-kernels. Host scheduling exercises the Muon callback ABI; this
does not establish native device execution.
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

#ifndef COUNT
#error COUNT is required
#endif
float stream_a[COUNT], stream_b[COUNT], stream_c[COUNT];
extern void entry(void);
typedef void (*callback_t)(void *, uint32_t, uint32_t, uint32_t);
void mu_schedule(callback_t callback, void *arg, uint32_t occupancy) {
  uint32_t threads = occupancy * 16;
  for (uint32_t block = 0; block < 2; ++block)
    for (uint32_t tid = 0; tid < threads; ++tid)
      callback(arg, tid, threads, block);
}
void muon_mlir_fence(void) { __asm__ __volatile__("" ::: "memory"); }
static void read_words(const char *path, float *dst) {
  FILE *file = fopen(path, "rb");
  if (!file) { perror(path); exit(2); }
  if (fread(dst, sizeof(float), COUNT, file) != COUNT || fgetc(file) != EOF) {
    fprintf(stderr, "bad input: %s\n", path); exit(2);
  }
  fclose(file);
}
int main(int argc, char **argv) {
  if (argc != 6) return 2;
  read_words(argv[1], stream_a);
  read_words(argv[2], stream_b);
  read_words(argv[3], stream_c);
  float *out = strcmp(argv[5], "a") == 0 ? stream_a :
               strcmp(argv[5], "b") == 0 ? stream_b : stream_c;
  memset(out, 0, COUNT * sizeof(float));
  entry();
  FILE *expected = fopen(argv[4], "rb");
  if (!expected) { perror(argv[4]); return 2; }
  uint64_t digest = UINT64_C(0xcbf29ce484222325);
  for (uint32_t i = 0; i < COUNT; ++i) {
    uint32_t actual, wanted;
    memcpy(&actual, out + i, sizeof(actual));
    if (fread(&wanted, sizeof(wanted), 1, expected) != 1 || actual != wanted) {
      fprintf(stderr, "word %u: %08x expected %08x\n", i, actual, wanted);
      return 1;
    }
    digest = (digest ^ actual) * UINT64_C(0x100000001b3);
  }
  if (fgetc(expected) != EOF) return 2;
  fclose(expected);
  printf("{\"words\":%u,\"digest\":\"%016" PRIx64 "\"}\n", COUNT, digest);
  return 0;
}
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], log: Path) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command}; see {log}")
    return result.stdout


def write_words(path: Path, values) -> None:
    with path.open("wb") as stream:
        batch = []
        for word in values:
            batch.append(word)
            if len(batch) == 32768:
                stream.write(struct.pack(f"<{len(batch)}I", *batch))
                batch.clear()
        if batch:
            stream.write(struct.pack(f"<{len(batch)}I", *batch))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--muon-opt", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=1048576)
    args = parser.parse_args()
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
    capture_dir = args.capture_dir.resolve()
    capture_path = capture_dir / "receipt.json"
    capture = json.loads(capture_path.read_text())
    if (capture.get("schema") != "radiance_model2mlir_stream_capture.v1" or
            capture.get("source_revision") != SOURCE_REVISION):
        parser.error("capture receipt does not match the source baseline")
    cases_by_kind = {case["kind"]: case for case in capture["cases"]}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in "abc":
        write_words(out / f"{name}.bin",
                    (reference.float_word(reference.input_value(name, i))
                     for i in range(args.elements)))
    harness = out / "muon_stream_host.c"
    harness.write_text(HARNESS)
    llvm_bin = args.llvm_bin.resolve()
    muon_opt = args.muon_opt.resolve()
    cases = []
    for kind in reference.KINDS:
        source_ir = capture_dir / f"stream_{kind}.parallel.mlir"
        if sha(source_ir) != cases_by_kind[kind]["parallel_mlir_sha256"]:
            raise RuntimeError(f"captured {kind} IR hash differs from receipt")
        target = reference.OUTPUT[kind]
        lowered = out / f"{kind}.muon.mlir"
        llvm_mlir = out / f"{kind}.llvm.mlir"
        llvm_ir = out / f"{kind}.ll"
        executable = out / f"{kind}.exe"
        expected = out / f"{kind}.expected.bin"
        write_words(expected,
                    (reference.float_word(reference.result_value(kind, i))
                     for i in range(args.elements)))
        run([str(muon_opt),
             f"--outline-forward-to-muon=inputs=stream_a,stream_b,stream_c output=stream_{target} warps=4",
             "--distribute-scf-parallel-to-muon=blocks=2", "--lower-muon-runtime",
             str(source_ir), "-o", str(lowered)], out / f"{kind}.outline.log")
        lowered_text = lowered.read_text()
        if "scf.parallel" in lowered_text or "call @mu_schedule" not in lowered_text:
            raise RuntimeError(f"{kind} did not become a scheduled Muon callback")
        run([str(llvm_bin / "mlir-opt"), str(lowered),
             "--expand-strided-metadata", "--convert-scf-to-cf",
             "--convert-arith-to-llvm", "--finalize-memref-to-llvm",
             "--convert-func-to-llvm", "--convert-cf-to-llvm",
             "--reconcile-unrealized-casts", "-o", str(llvm_mlir)],
            out / f"{kind}.convert.log")
        run([str(llvm_bin / "mlir-translate"), "--mlir-to-llvmir",
             str(llvm_mlir), "-o", str(llvm_ir)], out / f"{kind}.translate.log")
        run([str(llvm_bin / "clang"), "-O2", f"-DCOUNT={args.elements}",
             str(llvm_ir), str(harness), "-o", str(executable)],
            out / f"{kind}.clang.log")
        result = json.loads(run([str(executable), *(str(out / f"{name}.bin")
                                                 for name in "abc"),
                                 str(expected), target], out / f"{kind}.run.log"))
        digest = f"{reference.digest(reference.float_word(reference.result_value(kind, i)) for i in range(args.elements)):016x}"
        if result != {"words": args.elements, "digest": digest}:
            raise RuntimeError(f"{kind} Muon callback output differs from source")
        cases.append({"kind": kind, "words_compared": args.elements,
                      "digest": digest, "lowered_mlir_sha256": sha(lowered),
                      "llvm_ir_sha256": sha(llvm_ir)})
    receipt = {"schema": "radiance_model2mlir_stream_muon_host.v1",
               "status": "muon_callback_full_output_host_execution_not_device",
               "source_revision": SOURCE_REVISION,
               "source_run_sha256": sha(source),
               "capture_receipt_sha256": sha(capture_path),
               "model2mlir_revision": capture["model2mlir_revision"],
               "muon_opt_sha256": sha(muon_opt),
               "harness_sha256": sha(harness), "cases": cases}
    destination = out / "receipt.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "receipt": str(destination)}))


if __name__ == "__main__":
    main()
