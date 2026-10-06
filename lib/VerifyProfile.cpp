#include "Radiance/Passes.h"
#include "Muon/MuonOps.h"
#include "MxGemmini/MxOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <cctype>

using namespace mlir;

namespace {
class VerifyProfilePass : public PassWrapper<VerifyProfilePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyProfilePass)
  VerifyProfilePass() = default;
  VerifyProfilePass(const VerifyProfilePass &other) : PassWrapper(other) {}
  Option<std::string> selectedName{*this, "selected-name",
      llvm::cl::desc("name from the validated SoC profile"), llvm::cl::init("")};
  Option<std::string> selectedDigest{*this, "selected-sha256",
      llvm::cl::desc("SHA-256 of the validated SoC profile file"), llvm::cl::init("")};
  StringRef getArgument() const final { return "radiance-verify-profile"; }
  StringRef getDescription() const final { return "Check composed Muon/MX IR against the selected Radiance SoC profile"; }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto name = module->getAttrOfType<StringAttr>("radiance.profile");
    auto digest = module->getAttrOfType<StringAttr>("radiance.profile_sha256");
    if (selectedName.empty() || selectedDigest.empty() ||
        !name || name.getValue() != selectedName ||
        !digest || digest.getValue() != selectedDigest) {
      module.emitError("module profile does not match the selected profile name and SHA-256");
      signalPassFailure();
      return;
    }
    if (name.getValue() != "u250-e4m3" &&
        name.getValue() != "single-cluster-full-mx") {
      module.emitError("requires a selected radiance.profile");
      signalPassFailure();
      return;
    }
    if (!digest || digest.getValue().size() != 64 ||
        !llvm::all_of(digest.getValue(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)); })) {
      module.emitError("requires a 64-digit radiance.profile_sha256");
      signalPassFailure();
      return;
    }
    bool failed = false;
    module.walk([&](Operation *op) {
      if (!op->getName().getStringRef().starts_with("mx_gemmini.")) return;
      if (name.getValue() == "u250-e4m3") {
        auto format = op->getAttrOfType<StringAttr>("format");
        if (format && format.getValue() != "mxfp8") {
          op->emitError("MX format is absent from the pinned U250 E4M3 mesh");
          failed = true;
        }
        if (op->getName().getStringRef() == "mx_gemmini.load_lut") {
          op->emitError("the pinned U250 mesh has no QuantLut");
          failed = true;
        }
      }
    });
    // A readout visible to Muon must complete, be fenced, and cross a warp
    // barrier before the receiving warps consume shared memory.
    module.walk([&](Block *block) {
      bool pending = false, waited = false, fenced = false;
      bool muonPending = false, muonBarrier = false, muonFenced = false;
      StringAttr pendingSite;
      for (Operation &op : *block) {
        StringRef opName = op.getName().getStringRef();
        if (opName == "muon.launch") {
          if (pending) {
            op.emitError("Muon launch precedes the MX wait/fence/barrier handoff");
            failed = true;
          }
          muonPending = true;
          muonBarrier = muonFenced = false;
        } else if (muonPending && opName == "muon.barrier") {
          muonBarrier = true;
        } else if (muonPending && opName == "muon.smem_fence") {
          if (!muonBarrier) {
            op.emitError("Muon-to-MX shared-memory fence precedes the Muon barrier");
            failed = true;
          }
          muonFenced = true;
        } else if (muonPending && opName.starts_with("mx_gemmini.")) {
          if (!muonBarrier || !muonFenced) {
            op.emitError("MX use after Muon launch requires a Muon barrier and shared-memory fence");
            failed = true;
          }
          muonPending = false;
        }
        if (opName == "mx_gemmini.readout_to_smem") {
          if (pending) {
            op.emitError("previous shared-memory readout has no completed handoff");
            failed = true;
          }
          pending = true;
          pendingSite = op.getAttrOfType<StringAttr>("site_id");
          waited = fenced = false;
        } else if (pending && opName == "mx_gemmini.wait") {
          if (op.getAttrOfType<StringAttr>("site_id") != pendingSite) {
            op.emitError("MX wait site differs from pending shared-memory readout");
            failed = true;
          }
          waited = true;
        } else if (pending && opName == "muon.smem_fence") {
          if (!waited) {
            op.emitError("shared-memory fence precedes MX completion");
            failed = true;
          }
          fenced = true;
        } else if (pending && opName == "muon.barrier") {
          if (!waited || !fenced) {
            op.emitError("Muon barrier requires MX wait and shared-memory fence");
            failed = true;
          }
          pending = false;
        }
      }
      if (pending) {
        block->getParentOp()->emitError("MX shared-memory readout has no completed Muon handoff");
        failed = true;
      }
    });
    if (failed) signalPassFailure();
  }
};
} // namespace

namespace radiance {
void registerVerifyProfilePass() { PassRegistration<VerifyProfilePass>(); }
} // namespace radiance
