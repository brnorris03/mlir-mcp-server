"""MLIR analysis tools for MCP server.

This module implements tools for analyzing MLIR code structure and extracting
information about operations, functions, and data flow.
"""

import logging
from typing import Any

from ..config import MLIRConfig
from ..utils.error_handling import parse_mlir_diagnostic

logger = logging.getLogger(__name__)


def count_operations(config: MLIRConfig, mlir_code: str) -> dict[str, Any]:
    """Count operations by type in MLIR code.

    Args:
        config: MLIR configuration.
        mlir_code: MLIR code as string.

    Returns:
        Dictionary containing:
        - success: Whether analysis succeeded
        - operation_counts: Dict mapping operation names to counts
        - total_operations: Total number of operations
        - errors: List of errors (if failed)
    """
    try:
        from mlir import ir
    except ImportError as e:
        return {
            "success": False,
            "errors": [
                {
                    "message": f"MLIR Python bindings not available: {e}",
                    "severity": "error",
                }
            ],
        }

    try:
        with ir.Context():
            module = ir.Module.parse(mlir_code)

            operation_counts: dict[str, int] = {}
            total = 0

            # Walk all operations in the module
            def count_ops(op: Any) -> None:
                nonlocal total
                op_name = str(op.name)
                operation_counts[op_name] = operation_counts.get(op_name, 0) + 1
                total += 1

                # Recursively count operations in regions
                for region in op.regions:
                    for block in region.blocks:
                        for nested_op in block.operations:
                            count_ops(nested_op)

            # Count top-level operations
            for op in module.body.operations:
                count_ops(op)

            return {
                "success": True,
                "operation_counts": operation_counts,
                "total_operations": total,
            }

    except Exception as e:
        error_str = str(e)
        errors = parse_mlir_diagnostic(error_str)
        return {
            "success": False,
            "errors": [err.to_dict() for err in errors],
        }


def extract_function(
    config: MLIRConfig, mlir_code: str, function_name: str
) -> dict[str, Any]:
    """Extract a specific function from MLIR module.

    Args:
        config: MLIR configuration.
        mlir_code: MLIR code as string.
        function_name: Name of function to extract.

    Returns:
        Dictionary containing:
        - success: Whether extraction succeeded
        - function_code: Extracted function MLIR code
        - function_name: Name of extracted function
        - errors: List of errors (if failed)
    """
    try:
        from mlir import ir
    except ImportError as e:
        return {
            "success": False,
            "errors": [
                {
                    "message": f"MLIR Python bindings not available: {e}",
                    "severity": "error",
                }
            ],
        }

    try:
        with ir.Context():
            module = ir.Module.parse(mlir_code)

            # Search for function with matching name
            for op in module.body.operations:
                # Check if operation has sym_name attribute (functions have this)
                if "sym_name" in op.attributes:
                    sym_name = str(op.attributes["sym_name"]).strip('"')
                    if function_name == sym_name or function_name in sym_name:
                        return {
                            "success": True,
                            "function_code": str(op),
                            "function_name": function_name,
                        }

            # Function not found
            return {
                "success": False,
                "errors": [
                    {
                        "message": f"Function '{function_name}' not found in module",
                        "severity": "error",
                    }
                ],
            }

    except Exception as e:
        error_str = str(e)
        errors = parse_mlir_diagnostic(error_str)
        return {
            "success": False,
            "errors": [err.to_dict() for err in errors],
        }


def get_operation_operands(
    config: MLIRConfig, mlir_code: str, op_filter: str | None = None
) -> dict[str, Any]:
    """Get operands of operations, optionally filtered by operation name.

    Args:
        config: MLIR configuration.
        mlir_code: MLIR code as string.
        op_filter: Optional operation name filter (e.g., "arith.addi").

    Returns:
        Dictionary containing:
        - success: Whether analysis succeeded
        - operations: List of operation info with operands
        - errors: List of errors (if failed)
    """
    try:
        from mlir import ir
    except ImportError as e:
        return {
            "success": False,
            "errors": [
                {
                    "message": f"MLIR Python bindings not available: {e}",
                    "severity": "error",
                }
            ],
        }

    try:
        with ir.Context():
            module = ir.Module.parse(mlir_code)

            operations_info = []

            # Walk all operations
            def analyze_ops(op: Any) -> None:
                op_name = str(op.name)

                # Filter if requested
                if op_filter and op_filter not in op_name:
                    # Still recurse into nested operations
                    for region in op.regions:
                        for block in region.blocks:
                            for nested_op in block.operations:
                                analyze_ops(nested_op)
                    return

                # Extract operand information
                operands = []
                for operand in op.operands:
                    operands.append(
                        {
                            "type": str(operand.type),
                        }
                    )

                operations_info.append(
                    {
                        "name": op_name,
                        "num_operands": len(operands),
                        "operands": operands,
                        "num_results": len(op.results),
                    }
                )

                # Recurse into nested operations
                for region in op.regions:
                    for block in region.blocks:
                        for nested_op in block.operations:
                            analyze_ops(nested_op)

            # Analyze top-level operations
            for op in module.body.operations:
                analyze_ops(op)

            return {
                "success": True,
                "operations": operations_info,
                "filter": op_filter,
            }

    except Exception as e:
        error_str = str(e)
        errors = parse_mlir_diagnostic(error_str)
        return {
            "success": False,
            "errors": [err.to_dict() for err in errors],
        }
