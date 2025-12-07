// Simple valid MLIR module
module {
  func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    func.return %0 : i32
  }
}
