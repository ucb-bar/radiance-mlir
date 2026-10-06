add_library(RadiancePasses STATIC VerifyProfile.cpp)
target_link_libraries(RadiancePasses PUBLIC MLIRMuon MLIRMxGemmini MLIRIR MLIRPass)
