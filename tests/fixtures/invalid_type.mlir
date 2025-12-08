// Invalid MLIR - type mismatch
module {
  func.func @type_error(%arg0: i32, %arg1: i64) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    func.return %0 : i32
  }
}
