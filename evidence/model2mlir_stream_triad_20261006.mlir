builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<1048576xf32>, %1: tensor<1048576xf32>, %2: tensor<1048576xf32>) -> tensor<1048576xf32> {
    %3 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 2.000000e+00 : f32
    %4 = tensor.splat %3 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1048576xf32>
    %5 = tensor.empty() : tensor<1048576xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2, %4 : tensor<1048576xf32>, tensor<1048576xf32>) outs(%5 : tensor<1048576xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb0(%7: f32, %8: f32, %9: f32):
      %10 = arith.mulf %7, %8 : f32
      linalg.yield %10 : f32
    } -> tensor<1048576xf32>
    %11 = tensor.empty() : tensor<1048576xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %6 : tensor<1048576xf32>, tensor<1048576xf32>) outs(%11 : tensor<1048576xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb1(%13: f32, %14: f32, %15: f32):
      %16 = arith.addf %13, %14 : f32
      linalg.yield %16 : f32
    } -> tensor<1048576xf32>
    func.return %12 : tensor<1048576xf32>
  }
}
