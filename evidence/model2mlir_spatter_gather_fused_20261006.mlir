#map = affine_map<(d0, d1, d2) -> (0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%arg0: tensor<262144x1xi64>, %arg1: tensor<256xi64>) -> tensor<1024x256x1xi64> {
    %c0 = arith.constant 0 : index
    %c256_i64 = arith.constant 256 : i64
    %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [1, 256] {prov._pattern_hint = "unsqueeze", prov.aten = "aten.unsqueeze.default", prov.family = "layout", prov.op = "unsqueeze", prov.orig_dtype = "int64", prov.region_id = "unsqueeze_0"} : tensor<256xi64> into tensor<1x256xi64>
    %0 = tensor.empty() : tensor<1024x256x1xi64>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x256xi64>) outs(%0 : tensor<1024x256x1xi64>) {
    ^bb0(%in: i64, %out: i64):
      %2 = linalg.index 0 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = arith.muli %3, %c256_i64 : i64
      %5 = arith.addi %in, %4 : i64
      %6 = arith.index_cast %5 : i64 to index
      %extracted = tensor.extract %arg0[%6, %c0] : tensor<262144x1xi64>
      linalg.yield %extracted : i64
    } -> tensor<1024x256x1xi64>
    return %1 : tensor<1024x256x1xi64>
  }
}
