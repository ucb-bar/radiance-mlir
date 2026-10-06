add_executable(radiance-opt radiance-opt.cpp)
target_link_libraries(radiance-opt PRIVATE RadiancePasses MLIRMuon MLIRMxGemmini
  MLIROptLib MLIRIR MLIRFuncDialect MLIRLLVMDialect MLIRArithDialect
  MLIRMemRefDialect MLIRSCFDialect MLIRBufferizationDialect)
mlir_check_all_link_libraries(radiance-opt)
