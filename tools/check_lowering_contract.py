#!/usr/bin/env python3
"""Check route ownership, source coverage and evidence before promotion.

The normal check validates current, explicitly limited evidence. The stronger
--require-complete-claims check checks receipt metadata for every
Makefile-declared Radiance ELF. Neither mode reruns compiler or target tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTES = {
    "mx_commands": ("mx-gemmini-mlir", "typed_mx_contract_to_physical_commands", "issuer_backends_only", "rv64_rocc_or_muon_mmio"),
    "muon": ("muon-mlir", "typed_muon_ops_to_runtime_abi", "native_muon_rv32_backend_required", "muon_rv32_elf"),
    "radiance": ("radiance-mlir", "compose_muon_and_mx_without_new_compute_dialect", "rv64_host_and_native_muon_rv32_backend", "mx_mmio_from_muon_plus_host_abi"),
}
LEVELS = {"frontend_handoff", "source_payload", "host_callback", "target_executed"}
CURRENT_PROOFS = {
    "radiance_model2mlir_stream_muon_host.v1": ("stream", "host_callback", "source_golden_output", "radiance"),
    "radiance_model2mlir_gemm_muon_host.v1": ("gemm_simt", "host_callback", "source_golden_output", "radiance"),
    "radiance_model2mlir_spatter_muon_host.v1": ("spatter", "host_callback", "read_trace_only", "radiance"),
    "mx_gemmini_model2mlir_radiance_gemm_capture.v1": ("gemm_mxgemmini", "frontend_handoff", "shape_only", "radiance"),
    "mx_gemmini.source_payload_check.v1": ("gemm_mxgemmini", "source_payload", "fp8_m128n128k512_payload", "radiance"),
}
REQUIRED_ROUTE_PROOFS = {
    "mx_commands": ["physical_schedule", "rocc_issuer", "muon_mmio_issuer", "rocc_source_output_parity", "mmio_source_output_parity"],
    "muon": ["stack_abi", "rv32_executable", "source_output_parity"],
    "radiance": ["profile_identity", "mixed_artifact_link", "handoff_order", "mixed_execution", "source_output_parity"],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def confined(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"path escapes repository: {relative}")
    require(path.is_file(), f"missing evidence file: {relative}")
    return path


def projection(inventory: dict) -> list[dict]:
    require(inventory["schema"] == "muon_mlir_kernel_inventory.v2", "unsupported source inventory")
    return [{"name": f["name"], "radiance_elfs": f["build_targets"]["radiance_elfs"]}
            for f in inventory["families"]]


def check(root: Path = ROOT, *, inventory_path: Path | None = None,
          require_complete_claims: bool = False) -> dict:
    policy = read_json(root / "contracts/lowering_routes.json")
    require(policy["schema"] == "radiance.lowering_routes.v1", "wrong route schema")
    require(policy["merlin_boundary"] == "target_neutral_provider_and_artifact_bundle_only",
            "Merlin boundary changed")
    routes = {route["id"]: route for route in policy["routes"]}
    require(len(routes) == len(policy["routes"]) and set(routes) == set(ROUTES),
            "route set changed without updating its contract check")
    for name, expected in ROUTES.items():
        route = routes[name]
        actual = tuple(route[key] for key in
                       ("owner", "accelerator_lowering", "llvm_role", "transport"))
        require(actual == expected, f"{name}: lowering ownership or transport changed")
        require(route["status"] in {"planned", "partial", "validated"},
                f"{name}: invalid status")
        require(route["required_proofs"] == REQUIRED_ROUTE_PROOFS[name],
                f"{name}: qualification requirements changed")
        provided = set()
        for proof in route["validation_receipts"]:
            proof_id = proof["proof"]
            require(proof_id in route["required_proofs"] and proof_id not in provided,
                    f"{name}: duplicate or unknown route proof")
            provided.add(proof_id)
            proof_path = confined(root, proof["receipt"])
            require(digest(proof_path) == proof["receipt_sha256"],
                    f"{name}: route proof bytes changed")
            result = read_json(proof_path)
            require(result.get("schema") == "radiance.route_proof.v1" and
                    result.get("route") == name and result.get("proof") == proof_id and
                    result.get("status") == "verified" and
                    result.get("test_exit_code") == 0 and
                    result.get("artifact_sha256") and result.get("profile_sha256"),
                    f"{name}: invalid route proof")
        if route["status"] == "validated":
            require(provided == set(route["required_proofs"]),
                    f"{name}: validated route lacks qualification proofs")

    roster = read_json(root / "contracts/source_roster.json")
    coverage = read_json(root / "contracts/coverage.json")
    require(roster["schema"] == "radiance.source_roster.v1", "wrong roster schema")
    require(coverage["schema"] == "radiance.coverage.v1", "wrong coverage schema")
    require(roster["source_revision"] == coverage["source_revision"],
            "source revisions differ")
    require(len(roster["families"]) == len({f["name"] for f in roster["families"]}),
            "duplicate source family")
    require(roster["families"] == sorted(roster["families"], key=lambda f: f["name"]),
            "source roster must be sorted")
    targets = {(f["name"], target) for f in roster["families"]
               for target in f["radiance_elfs"]}
    require(len(targets) == sum(len(f["radiance_elfs"]) for f in roster["families"]),
            "duplicate source build target")
    inventory_path = inventory_path or root / "evidence/upstream/kernel-inventory-20261006.json"
    inventory = read_json(inventory_path)
    require("source_root" not in inventory,
            "inventory must use source revision and relative file paths, not a checkout path")
    require(digest(inventory_path) == roster["inventory_sha256"],
            "source inventory snapshot changed")
    require(inventory["source_git_revision"] == roster["source_revision"],
            "source revision changed")
    require(projection(inventory) == roster["families"],
            "source Makefile target roster changed")

    names = {f["name"] for f in roster["families"]}
    executed = set()
    claim_keys = set()
    for claim in coverage["claims"]:
        family, level, route = claim["family"], claim["level"], claim["route"]
        claim_key = (family, level, claim.get("target"), claim["scope"])
        require(claim_key not in claim_keys, f"duplicate coverage claim: {claim_key}")
        claim_keys.add(claim_key)
        require(family in names and level in LEVELS and route in routes,
                f"invalid coverage claim: {family}/{level}/{route}")
        require(bool(claim["scope"]), f"{family}: missing claim scope")
        receipt_path = confined(root, claim["receipt"])
        require(digest(receipt_path) == claim["receipt_sha256"],
                f"{family}: evidence changed without requalification")
        receipt = read_json(receipt_path)
        require(receipt.get("schema") == claim["receipt_schema"] and
                receipt.get("status") == claim["receipt_status"] and
                receipt.get("source_revision") == roster["source_revision"],
                f"{family}: receipt identity or status differs")
        if level == "target_executed":
            target = claim.get("target")
            require((family, target) in targets, f"{family}: unknown executed target")
            require(route in {"radiance", "muon", "mx_commands"},
                    f"{family}: wrong device route")
            oracle = receipt.get("oracle", {})
            require(isinstance(oracle, dict) and
                    oracle.get("kind") in {"output_words", "memory_trace", "control_trace"} and
                    isinstance(oracle.get("observations_compared"), int) and
                    oracle["observations_compared"] > 0 and
                    oracle.get("source_digest") and
                    oracle.get("source_digest") == oracle.get("compiled_digest"),
                    f"{family}/{target}: missing source-equivalence oracle")
            require(receipt["schema"] == "radiance.target_execution.v1" and
                    receipt["status"] == "executed_and_compared" and
                    receipt.get("target") == target and receipt.get("route") == route and
                    receipt.get("profile_sha256") and receipt.get("artifact_bundle_sha256") and
                    receipt.get("execution_environment") in {"firesim", "fpga", "simulator"},
                    f"{family}/{target}: incomplete target execution proof")
            bundle = confined(root, receipt["artifact_bundle"])
            require(digest(bundle) == receipt["artifact_bundle_sha256"],
                    f"{family}/{target}: artifact bundle changed")
            executed.add((family, target))
        else:
            require("target" not in claim, f"{family}: non-device claim cannot cover a target")
            require(CURRENT_PROOFS.get(receipt["schema"]) ==
                    (family, level, claim["scope"], route),
                    f"{family}: claimed scope exceeds the receipt's proof")

    missing = sorted(targets - executed)
    if require_complete_claims:
        require(not missing, f"incomplete: {len(missing)} source ELF targets lack target execution receipts; first: {missing[:5]}")
        require(all(route["status"] == "validated" for route in routes.values()),
                "incomplete: target routes lack qualification receipts")
    return {"families": len(names), "source_elf_targets": len(targets),
            "claims": len(coverage["claims"]), "executed_targets": len(executed),
            "missing_executed_targets": len(missing)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, help="verify against the original Muon source inventory")
    parser.add_argument("--require-complete-claims", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(check(inventory_path=args.inventory,
                               require_complete_claims=args.require_complete_claims), sort_keys=True))
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        parser.exit(1, f"lowering contract: {error}\n")


if __name__ == "__main__":
    main()
