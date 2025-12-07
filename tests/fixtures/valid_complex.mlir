// More complex MLIR with control flow
module {
  func.func @conditional(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
    %result = scf.if %arg0 -> (i32) {
      scf.yield %arg1 : i32
    } else {
      scf.yield %arg2 : i32
    }
    func.return %result : i32
  }

  func.func @loop(%arg0: index, %arg1: index, %arg2: index) -> index {
    %sum = scf.for %i = %arg0 to %arg1 step %arg2 iter_args(%iter = %arg0) -> (index) {
      %next = arith.addi %iter, %i : index
      scf.yield %next : index
    }
    func.return %sum : index
  }
}
