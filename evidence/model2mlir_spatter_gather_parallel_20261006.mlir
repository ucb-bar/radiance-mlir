module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%arg0: memref<262144x1xi64, strided<[?, ?], offset: ?>>, %arg1: memref<256xi64, strided<[?], offset: ?>>) -> memref<1024x256x1xi64> {
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c256_i64 = arith.constant 256 : i64
    %expand_shape = memref.expand_shape %arg1 [[0, 1]] output_shape [1, 256] : memref<256xi64, strided<[?], offset: ?>> into memref<1x256xi64, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024x256x1xi64>
    scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c1024, %c256, %c1) step (%c1, %c1, %c1) {
      %0 = memref.load %expand_shape[%c0, %arg3] : memref<1x256xi64, strided<[?, ?], offset: ?>>
      %1 = arith.index_cast %arg2 : index to i64
      %2 = arith.muli %1, %c256_i64 : i64
      %3 = arith.addi %0, %2 : i64
      %4 = arith.index_cast %3 : i64 to index
      %5 = memref.load %arg0[%4, %c0] : memref<262144x1xi64, strided<[?, ?], offset: ?>>
      memref.store %5, %alloc[%arg2, %arg3, %arg4] : memref<1024x256x1xi64>
      scf.reduce
    }
    return %alloc : memref<1024x256x1xi64>
  }
}
