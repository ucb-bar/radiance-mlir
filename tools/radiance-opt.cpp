#include "Radiance/Passes.h"
#include "Muon/MuonDialect.h"
#include "Muon/Passes.h"
#include "MxGemmini/MxDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
int main(int argc, char **argv) {
  radiance::registerVerifyProfilePass();
  mlir::muon::registerLowerRuntimePass();
  mlir::DialectRegistry registry;
  registry.insert<mlir::muon::MuonDialect, mlir::mx_gemmini::MxGemminiDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::bufferization::BufferizationDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv,
                                                  "Radiance composed compiler\n", registry));
}
