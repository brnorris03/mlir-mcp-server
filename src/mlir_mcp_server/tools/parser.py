"""MLIR parsing tools for MCP server.

This module implements tools for parsing and validating MLIR code using
the MLIR Python bindings.
"""

import logging
from typing import Any

from ..config import MLIRConfig
from ..utils.error_handling import parse_mlir_diagnostic

logger = logging.getLogger(__name__)


def parse_mlir(config: MLIRConfig, mlir_code: str) -> dict[str, Any]:
    """Parse MLIR code into an IR module.

    This tool parses MLIR textual representation and returns information about
    the parsed module, or error details if parsing fails.

    Args:
        config: MLIR configuration with toolchain information.
        mlir_code: MLIR code as a string.

    Returns:
        Dictionary containing:
        - success: Whether parsing succeeded
        - module_str: String representation of the module (if successful)
        - operations: List of top-level operation names (if successful)
        - errors: List of error dictionaries (if failed)
    """
    try:
        # Import MLIR Python bindings
        try:
            from mlir import ir
        except ImportError as e:
            return {
                "success": False,
                "errors": [
                    {
                        "message": f"MLIR Python bindings not available: {e}. "
                        "Install with: pip install mlir (from LLVM build)",
                        "severity": "error",
                    }
                ],
            }

        # Parse the MLIR code
        with ir.Context() as ctx:
            try:
                module = ir.Module.parse(mlir_code)

                # Extract operation information
                operations = []
                for op in module.body.operations:
                    operations.append(op.name)

                return {
                    "success": True,
                    "module_str": str(module),
                    "operations": operations,
                    "operation_count": len(operations),
                }

            except Exception as parse_error:
                # Parse MLIR diagnostics from error message
                error_str = str(parse_error)
                errors = parse_mlir_diagnostic(error_str)

                return {
                    "success": False,
                    "errors": [err.to_dict() for err in errors],
                }

    except Exception as e:
        logger.exception("Unexpected error in parse_mlir")
        return {
            "success": False,
            "errors": [{"message": f"Unexpected error: {str(e)}", "severity": "error"}],
        }


def validate_mlir(config: MLIRConfig, mlir_code: str) -> dict[str, Any]:
    """Validate MLIR code syntax and semantics.

    This tool validates MLIR code and returns detailed error information if
    validation fails.

    Args:
        config: MLIR configuration with toolchain information.
        mlir_code: MLIR code as a string.

    Returns:
        Dictionary containing:
        - valid: Whether the code is valid
        - errors: List of error dictionaries (if invalid)
        - warnings: List of warning dictionaries
    """
    result = parse_mlir(config, mlir_code)

    if result["success"]:
        return {
            "valid": True,
            "message": "MLIR code is valid",
        }
    else:
        return {
            "valid": False,
            "errors": result["errors"],
        }


def get_module_info(config: MLIRConfig, mlir_code: str) -> dict[str, Any]:
    """Extract detailed information from an MLIR module.

    This tool parses MLIR code and extracts comprehensive structural information
    including operations, blocks, regions, and arguments.

    Args:
        config: MLIR configuration with toolchain information.
        mlir_code: MLIR code as a string.

    Returns:
        Dictionary containing:
        - success: Whether parsing succeeded
        - operations: List of operation details
        - errors: List of error dictionaries (if failed)
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
        with ir.Context() as ctx:
            module = ir.Module.parse(mlir_code)

            operations = []
            for op in module.body.operations:
                op_info = {
                    "name": op.name,
                    "num_operands": len(op.operands),
                    "num_results": len(op.results),
                    "num_regions": len(op.regions),
                    "attributes": {},
                }

                # Extract attributes
                try:
                    for attr_name in op.attributes:
                        attr = op.attributes[attr_name]
                        op_info["attributes"][attr_name] = str(attr)
                except Exception:
                    pass  # Some operations may not expose attributes this way

                # Extract result types
                op_info["result_types"] = [str(res.type) for res in op.results]

                # Count blocks in regions
                op_info["blocks_per_region"] = []
                for region in op.regions:
                    op_info["blocks_per_region"].append(len(region.blocks))

                operations.append(op_info)

            return {
                "success": True,
                "operation_count": len(operations),
                "operations": operations,
            }

    except Exception as e:
        error_str = str(e)
        errors = parse_mlir_diagnostic(error_str)
        return {
            "success": False,
            "errors": [err.to_dict() for err in errors],
        }
