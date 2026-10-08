"""Contract checks that run without Chipyard or an MLIR toolchain."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_lowering_contract import check  # noqa: E402


class LoweringContractTests(unittest.TestCase):
    def test_current_claims_are_limited(self) -> None:
        result = check()
        self.assertEqual(result["families"], 64)
        self.assertEqual(result["source_elf_targets"], 136)
        self.assertEqual(result["executed_targets"], 0)
        with self.assertRaisesRegex(ValueError, "136 source ELF targets"):
            check(require_complete_claims=True)

    def test_changed_route_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.copy_contract(Path(directory))
            path = root / "contracts/lowering_routes.json"
            data = json.loads(path.read_text())
            next(x for x in data["routes"] if x["id"] == "mx_commands")["llvm_role"] = "required_for_commands"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "mx_commands: lowering ownership"):
                check(root)

    def test_route_cannot_be_validated_by_status_edit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.copy_contract(Path(directory))
            path = root / "contracts/lowering_routes.json"
            data = json.loads(path.read_text())
            next(x for x in data["routes"] if x["id"] == "muon")["status"] = "validated"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "muon: validated route lacks"):
                check(root)

    def test_receipt_status_cannot_be_promoted_by_editing_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.copy_contract(Path(directory))
            path = root / "contracts/coverage.json"
            data = json.loads(path.read_text())
            data["claims"][0]["receipt_status"] = "executed_and_compared"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "receipt identity or status differs"):
                check(root)

    def test_partial_trace_cannot_be_called_full_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.copy_contract(Path(directory))
            path = root / "contracts/coverage.json"
            data = json.loads(path.read_text())
            next(x for x in data["claims"] if x["family"] == "spatter")["scope"] = "source_golden_output"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "claimed scope exceeds"):
                check(root)

    def test_source_target_roster_must_match_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.copy_contract(Path(directory))
            path = root / "contracts/source_roster.json"
            data = json.loads(path.read_text())
            data["families"][0]["radiance_elfs"].append("invented.radiance.elf")
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "source Makefile target roster changed"):
                check(root)

    def test_inventory_cannot_publish_checkout_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.copy_contract(Path(directory))
            path = root / "evidence/upstream/kernel-inventory-20261006.json"
            data = json.loads(path.read_text())
            data["source_root"] = "/private/checkout"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "not a checkout path"):
                check(root)

    @staticmethod
    def copy_contract(root: Path) -> Path:
        shutil.copytree(ROOT / "contracts", root / "contracts")
        shutil.copytree(ROOT / "evidence", root / "evidence")
        return root


if __name__ == "__main__":
    unittest.main()
