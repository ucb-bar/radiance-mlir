// Structural QK -> Muon softmax/quantization -> PV contract.
// The callback body and MX command lowering are intentionally external.
module attributes {
  radiance.profile = "u250-e4m3",
  radiance.profile_sha256 = "2acc27561a647e54a91828d443e4bad8f0859527ad066b5613225803c6f343b6",
  mx.contract_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  mx.policy_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  prov.quantization_manifest_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
} {
  func.func private @softmax_quant_kernel(!llvm.ptr, i32, i32, i32)
  func.func @attention(%q: tensor<16x16xf32>, %k: tensor<16x16xf32>,
                       %v: tensor<16x16xf32>, %score_smem: memref<16x16xbf16>,
                       %prob_smem: memref<16x16xf32>, %muon_args: !llvm.ptr)
      -> tensor<16x16xbf16> {
    %qcodes, %qscales = "mx_gemmini.encode"(%q) {
      site_id = "qk", format = "mxfp8",
      contract_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      policy_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      manifest_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    } : (tensor<16x16xf32>) -> (tensor<16x16xi8>, tensor<16xi8>)
    %kcodes, %kscales = "mx_gemmini.encode"(%k) {
      site_id = "qk", format = "mxfp8",
      contract_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      policy_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      manifest_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    } : (tensor<16x16xf32>) -> (tensor<16x16xi8>, tensor<16xi8>)
    %qk = "mx_gemmini.contract"(%qcodes, %qscales, %kcodes, %kscales) {
      site_id = "qk", format = "mxfp8",
      contract_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      policy_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      manifest_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    } : (tensor<16x16xi8>, tensor<16xi8>, tensor<16x16xi8>, tensor<16xi8>) -> tensor<16x16xbf16>
    "mx_gemmini.readout_to_smem"(%qk, %score_smem) {
      site_id = "qk",
      contract_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      policy_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      manifest_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    } : (tensor<16x16xbf16>, memref<16x16xbf16>) -> ()
    "mx_gemmini.wait"() {
      site_id = "qk",
      contract_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      policy_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      manifest_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    } : () -> ()
    "muon.smem_fence"() : () -> ()
    "muon.barrier"() {barrier_id = 1 : i32, num_warps = 8 : i32} : () -> ()
    // %muon_args must name score_smem and prob_smem in the runtime ABI.
    "muon.launch"(%muon_args) {kernel = @softmax_quant_kernel, warps_per_core = 4 : i32} : (!llvm.ptr) -> ()
    "muon.barrier"() {barrier_id = 2 : i32, num_warps = 8 : i32} : () -> ()
    "muon.smem_fence"() : () -> ()
    %prob = bufferization.to_tensor %prob_smem restrict : memref<16x16xf32> to tensor<16x16xf32>
    %pcodes, %pscales = "mx_gemmini.encode"(%prob) {
      site_id = "pv", format = "mxfp8",
      contract_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      policy_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      manifest_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    } : (tensor<16x16xf32>) -> (tensor<16x16xi8>, tensor<16xi8>)
    %vcodes, %vscales = "mx_gemmini.encode"(%v) {
      site_id = "pv", format = "mxfp8",
      contract_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      policy_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      manifest_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    } : (tensor<16x16xf32>) -> (tensor<16x16xi8>, tensor<16xi8>)
    %pv = "mx_gemmini.contract"(%pcodes, %pscales, %vcodes, %vscales) {
      site_id = "pv", format = "mxfp8",
      contract_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      policy_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      manifest_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    } : (tensor<16x16xi8>, tensor<16xi8>, tensor<16x16xi8>, tensor<16xi8>) -> tensor<16x16xbf16>
    %out = "mx_gemmini.readout_bf16"(%pv) {
      site_id = "pv",
      contract_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      policy_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      manifest_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    } : (tensor<16x16xbf16>) -> tensor<16x16xbf16>
    return %out : tensor<16x16xbf16>
  }
}
