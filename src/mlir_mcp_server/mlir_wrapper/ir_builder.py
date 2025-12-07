"""MLIR IR builder wrappers.

This module provides high-level wrappers around MLIR Python IR builders
to simplify MLIR code generation.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def build_function(
    name: str,
    arg_types: list[str],
    result_types: list[str],
    body_builder: Optional[Any] = None,
) -> str:
    """Build an MLIR function with specified signature.

    Args:
        name: Function name.
        arg_types: List of argument type strings (e.g., ["i32", "f64"]).
        result_types: List of result type strings.
        body_builder: Optional callable to build function body.

    Returns:
        Generated MLIR function as a string.

    Raises:
        ImportError: If MLIR Python bindings are not available.
    """
    try:
        from mlir import ir
        from mlir.dialects import func
    except ImportError as e:
        raise ImportError(f"MLIR Python bindings not available: {e}")

    with ir.Context() as ctx, ir.Location.unknown():
        # Load required dialects
        ctx.load_all_available_dialects()

        # Create module
        module = ir.Module.create()

        with ir.InsertionPoint(module.body):
            # Parse argument types
            parsed_arg_types = []
            for arg_type_str in arg_types:
                parsed_arg_types.append(ir.Type.parse(arg_type_str))

            # Parse result types
            parsed_result_types = []
            for result_type_str in result_types:
                parsed_result_types.append(ir.Type.parse(result_type_str))

            # Create function type
            func_type = ir.FunctionType.get(
                inputs=parsed_arg_types,
                results=parsed_result_types,
            )

            # Create function operation
            func_op = func.FuncOp(
                name=name,
                type=func_type,
            )

            # Build body if builder provided
            if body_builder:
                with ir.InsertionPoint(func_op.add_entry_block()):
                    body_builder(func_op.entry_block.arguments)
            else:
                # Create empty body with return
                with ir.InsertionPoint(func_op.add_entry_block()):
                    # Just create empty return for now
                    func.ReturnOp([])

        return str(module)


def build_operation(
    op_name: str,
    operands: list[str],
    result_types: list[str],
    attributes: Optional[dict[str, Any]] = None,
) -> str:
    """Build a standalone MLIR operation.

    Args:
        op_name: Operation name (e.g., "arith.addi").
        operands: List of operand SSA value names.
        result_types: List of result type strings.
        attributes: Optional dictionary of operation attributes.

    Returns:
        Generated MLIR operation as a string.

    Raises:
        ImportError: If MLIR Python bindings are not available.
    """
    try:
        from mlir import ir
    except ImportError as e:
        raise ImportError(f"MLIR Python bindings not available: {e}")

    # Build a simple representation
    # For a more complete implementation, we would need to actually construct
    # the operation in a module context
    attr_str = ""
    if attributes:
        attr_parts = [f"{k} = {v}" for k, v in attributes.items()]
        attr_str = " {" + ", ".join(attr_parts) + "}"

    operands_str = ", ".join(operands) if operands else ""
    result_type_str = ", ".join(result_types) if result_types else ""

    if result_type_str:
        return f"%result = {op_name} {operands_str}{attr_str} : {result_type_str}"
    else:
        return f"{op_name} {operands_str}{attr_str}"


def build_module_from_operations(operations: list[str]) -> str:
    """Build an MLIR module containing the given operations.

    Args:
        operations: List of MLIR operation strings.

    Returns:
        Generated MLIR module as a string.

    Raises:
        ImportError: If MLIR Python bindings are not available.
    """
    try:
        from mlir import ir
    except ImportError as e:
        raise ImportError(f"MLIR Python bindings not available: {e}")

    with ir.Context() as ctx, ir.Location.unknown():
        ctx.load_all_available_dialects()
        module = ir.Module.create()

        # Parse and add each operation
        with ir.InsertionPoint(module.body):
            for op_str in operations:
                # Parse the operation string
                try:
                    parsed = ir.Operation.parse(op_str)
                except Exception as e:
                    logger.warning(f"Failed to parse operation: {op_str}: {e}")
                    continue

        return str(module)
