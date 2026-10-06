module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%arg0: memref<1048576xf32, strided<[?], offset: ?>>, %arg1: memref<1048576xf32, strided<[?], offset: ?>>, %arg2: memref<1048576xf32, strided<[?], offset: ?>>) -> memref<1048576xf32> {
    %c1 = arith.constant 1 : index
    %c1048576 = arith.constant 1048576 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1048576xf32>
    scf.parallel (%arg3) = (%c0) to (%c1048576) step (%c1) {
      %0 = memref.load %arg1[%arg3] : memref<1048576xf32, strided<[?], offset: ?>>
      %1 = memref.load %arg2[%arg3] : memref<1048576xf32, strided<[?], offset: ?>>
      %2 = arith.mulf %1, %cst : f32
      %3 = arith.addf %0, %2 : f32
      memref.store %3, %alloc[%arg3] : memref<1048576xf32>
      scf.reduce
    }
    return %alloc : memref<1048576xf32>
  }
}
