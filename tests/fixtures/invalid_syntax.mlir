// Invalid MLIR - missing closing brace
module {
  func.func @broken(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    func.return %0 : i32
  // Missing closing brace here
}
