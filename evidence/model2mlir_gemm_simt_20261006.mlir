builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<64x64xbf16>, %1: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<64x64xbf16>) outs(%2 : tensor<64x64xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb0(%4: bf16, %5: f32):
      %6 = arith.extf %4 : bf16 to f32
      linalg.yield %6 : f32
    } -> tensor<64x64xf32>
    %7 = tensor.empty() : tensor<64x64xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<64x64xbf16>) outs(%7 : tensor<64x64xf32>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb1(%9: bf16, %10: f32):
      %11 = arith.extf %9 : bf16 to f32
      linalg.yield %11 : f32
    } -> tensor<64x64xf32>
    %12 = arith.constant {prov.region_id = "matmul_0", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %13 = tensor.splat %12 {prov.region_id = "matmul_0", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} : tensor<64x64xf32>
    %14 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%3, %8 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %15 = tensor.empty() : tensor<64x64xbf16>
    %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<64x64xf32>) outs(%15 : tensor<64x64xbf16>) attrs =  {prov.region_id = "dtype_cast_2", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bfloat16"} {
    ^bb2(%17: f32, %18: bf16):
      %19 = arith.truncf %17 : f32 to bf16
      linalg.yield %19 : bf16
    } -> tensor<64x64xbf16>
    func.return %16 : tensor<64x64xbf16>
  }
}
