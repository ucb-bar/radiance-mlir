#include "Radiance/Passes.h"
#include "Muon/MuonOps.h"
#include "MxGemmini/MxOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <cctype>
#include <map>
#include <set>

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
  Option<std::string> selectedFormats{*this, "selected-formats",
      llvm::cl::desc("comma-separated MX formats from the validated SoC profile"), llvm::cl::init("")};
  Option<bool> selectedLut{*this, "selected-lut",
      llvm::cl::desc("whether the validated SoC profile has an MX QuantLut"), llvm::cl::init(false)};
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
    SmallVector<StringRef> formatNames;
    StringRef(selectedFormats.getValue()).split(formatNames, ',');
    std::set<std::string> admittedFormats;
    for (StringRef format : formatNames) {
      if (format != "mxfp8" && format != "mxfp6" && format != "mxfp4") {
        module.emitError("selected SoC profile has an unsupported MX format");
        signalPassFailure();
        return;
      }
      admittedFormats.insert(format.str());
    }
    if (admittedFormats.empty() || admittedFormats.size() != formatNames.size() ||
        (name.getValue() == "u250-e4m3" &&
         (admittedFormats != std::set<std::string>{"mxfp8"} || selectedLut))) {
      module.emitError("selected MX capabilities disagree with the bound SoC profile");
      signalPassFailure();
      return;
    }
    bool failed = false;
    module.walk([&](Operation *op) {
      if (!op->getName().getStringRef().starts_with("mx_gemmini.")) return;
      auto format = op->getAttrOfType<StringAttr>("format");
      if (format && !admittedFormats.count(format.getValue().str())) {
        op->emitError("MX format is absent from the selected SoC profile");
        failed = true;
      }
      if (!selectedLut && op->getName().getStringRef() == "mx_gemmini.load_lut") {
        op->emitError("the selected SoC profile has no QuantLut");
        failed = true;
      }
    });
    // Track handoffs along every reachable CFG edge. At a join, differing
    // outstanding effects are refused because either path could reach the
    // consumer with a different completion state.
    struct HandoffState {
      bool pending = false, waited = false, fenced = false;
      bool muonPending = false, muonBarrier = false, muonFenced = false;
      StringAttr pendingSite;
      bool operator==(const HandoffState &other) const {
        return pending == other.pending && waited == other.waited &&
               fenced == other.fenced && muonPending == other.muonPending &&
               muonBarrier == other.muonBarrier && muonFenced == other.muonFenced &&
               pendingSite == other.pendingSite;
      }
    };
    auto checkBlock = [&](Block *block, HandoffState state) {
      for (Operation &op : *block) {
        // Function definitions are declarations in the surrounding module,
        // not executed between adjacent top-level handoff operations.
        if (isa<SymbolOpInterface>(op)) continue;
        StringRef opName = op.getName().getStringRef();
        if (op.getNumRegions() && (state.pending || state.muonPending)) {
          op.emitError("cross-engine handoff across a nested region requires explicit region effects");
          failed = true;
        }
        if (opName == "muon.launch") {
          if (state.pending) {
            op.emitError("Muon launch precedes the MX wait/fence/barrier handoff");
            failed = true;
          }
          state.muonPending = true;
          state.muonBarrier = state.muonFenced = false;
        } else if (state.muonPending && opName == "muon.barrier") {
          state.muonBarrier = true;
        } else if (state.muonPending && opName == "muon.smem_fence") {
          if (!state.muonBarrier) {
            op.emitError("Muon-to-MX shared-memory fence precedes the Muon barrier");
            failed = true;
          }
          state.muonFenced = true;
        } else if (state.muonPending && opName.starts_with("mx_gemmini.")) {
          if (!state.muonBarrier || !state.muonFenced) {
            op.emitError("MX use after Muon launch requires a Muon barrier and shared-memory fence");
            failed = true;
          }
          state.muonPending = false;
        }
        if (opName == "mx_gemmini.readout_to_smem") {
          if (state.pending) {
            op.emitError("previous shared-memory readout has no completed handoff");
            failed = true;
          }
          state.pending = true;
          state.pendingSite = op.getAttrOfType<StringAttr>("site_id");
          state.waited = state.fenced = false;
        } else if (state.pending && opName == "mx_gemmini.wait") {
          if (op.getAttrOfType<StringAttr>("site_id") != state.pendingSite) {
            op.emitError("MX wait site differs from pending shared-memory readout");
            failed = true;
          }
          state.waited = true;
        } else if (state.pending && opName == "muon.smem_fence") {
          if (!state.waited) {
            op.emitError("shared-memory fence precedes MX completion");
            failed = true;
          }
          state.fenced = true;
        } else if (state.pending && opName == "muon.barrier") {
          if (!state.waited || !state.fenced) {
            op.emitError("Muon barrier requires MX wait and shared-memory fence");
            failed = true;
          }
          state.pending = false;
          state.pendingSite = {};
          state.waited = state.fenced = false;
        }
      }
      return state;
    };
    module.walk([&](Operation *parent) {
      for (Region &region : parent->getRegions()) {
        if (region.empty()) continue;
        std::map<Block *, HandoffState> incoming;
        SmallVector<Block *> work{&region.front()};
        incoming.emplace(&region.front(), HandoffState{});
        while (!work.empty()) {
          Block *block = work.pop_back_val();
          HandoffState outgoing = checkBlock(block, incoming.at(block));
          Operation *terminator = block->empty() ? nullptr : &block->back();
          if (!terminator || terminator->getNumSuccessors() == 0) {
            if (outgoing.pending) {
              parent->emitError("MX shared-memory readout has no completed Muon handoff");
              failed = true;
            }
            continue;
          }
          for (Block *successor : terminator->getSuccessors()) {
            auto [it, inserted] = incoming.emplace(successor, outgoing);
            if (inserted) work.push_back(successor);
            else if (!(it->second == outgoing)) {
              terminator->emitError("cross-engine handoff state differs across CFG predecessors");
              failed = true;
            }
          }
        }
      }
    });
    if (failed) signalPassFailure();
  }
};
} // namespace

namespace radiance {
void registerVerifyProfilePass() { PassRegistration<VerifyProfilePass>(); }
} // namespace radiance
