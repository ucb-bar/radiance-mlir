builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<262144x1xi64>, %1: tensor<256xi64>) -> tensor<1024x256x1xi64> {
    %2 = tensor.empty() : tensor<1024xi64>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%2 : tensor<1024xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb0(%4: i64):
      %5 = linalg.index 0 : index
      %6 = arith.index_cast %5 : index to i64
      %7 = arith.constant 1 : i64
      %8 = arith.muli %6, %7 : i64
      %9 = arith.constant 0 : i64
      %10 = arith.addi %9, %8 : i64
      linalg.yield %10 : i64
    } -> tensor<1024xi64>
    %11 = tensor.expand_shape %1 [[0 : i64, 1 : i64]] output_shape [1, 256] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<256xi64> into tensor<1x256xi64>
    %12 = tensor.expand_shape %3 [[0 : i64, 1 : i64]] output_shape [1024, 1] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1024xi64> into tensor<1024x1xi64>
    %13 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} 256 : i64
    %14 = tensor.splat %13 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} : tensor<1024x1xi64>
    %15 = tensor.empty() : tensor<1024x1xi64>
    %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12, %14 : tensor<1024x1xi64>, tensor<1024x1xi64>) outs(%15 : tensor<1024x1xi64>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} {
    ^bb1(%17: i64, %18: i64, %19: i64):
      %20 = arith.muli %17, %18 : i64
      linalg.yield %20 : i64
    } -> tensor<1024x1xi64>
    %21 = tensor.empty() : tensor<1024x256xi64>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%11, %16 : tensor<1x256xi64>, tensor<1024x1xi64>) outs(%21 : tensor<1024x256xi64>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb2(%23: i64, %24: i64, %25: i64):
      %26 = arith.addi %23, %24 : i64
      linalg.yield %26 : i64
    } -> tensor<1024x256xi64>
    %27 = tensor.empty() : tensor<1024x256x1xi64>
    %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22 : tensor<1024x256xi64>) outs(%27 : tensor<1024x256x1xi64>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "int64"} {
    ^bb3(%29: i64, %30: i64):
      %31 = arith.index_cast %29 : i64 to index
      %32 = linalg.index 2 : index
      %33 = tensor.extract %0[%31, %32] : tensor<262144x1xi64>
      linalg.yield %33 : i64
    } -> tensor<1024x256x1xi64>
    func.return %28 : tensor<1024x256x1xi64>
  }
}
