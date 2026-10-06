#map = affine_map<(d0) -> (d0)>
module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%arg0: tensor<1048576xf32>, %arg1: tensor<1048576xf32>, %arg2: tensor<1048576xf32>) -> tensor<1048576xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = tensor.empty() : tensor<1048576xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg1, %arg2 : tensor<1048576xf32>, tensor<1048576xf32>) outs(%0 : tensor<1048576xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.mulf %in_0, %cst : f32
      %3 = arith.addf %in, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<1048576xf32>
    return %1 : tensor<1048576xf32>
  }
}
