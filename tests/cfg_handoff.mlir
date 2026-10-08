module attributes {
  radiance.profile = "u250-e4m3",
  radiance.profile_sha256 = "2acc27561a647e54a91828d443e4bad8f0859527ad066b5613225803c6f343b6",
  mx.contract_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  mx.policy_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  prov.quantization_manifest_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
} {
  func.func @handoff(%acc: tensor<16x16xbf16>, %shared: memref<16x16xbf16>) {
    "mx_gemmini.readout_to_smem"(%acc, %shared) {
      site_id = "qk", contract_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      policy_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      manifest_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    } : (tensor<16x16xbf16>, memref<16x16xbf16>) -> ()
    cf.br ^bb1
  ^bb1:
    "mx_gemmini.wait"() {
      site_id = "qk", contract_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      policy_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      manifest_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    } : () -> ()
    cf.br ^bb2
  ^bb2:
    "muon.smem_fence"() : () -> ()
    "muon.barrier"() {barrier_id = 1 : i32, num_warps = 8 : i32} : () -> ()
    return
  }
}
