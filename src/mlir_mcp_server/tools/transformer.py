"""MLIR transformation tools for MCP server.

This module implements tools for transforming MLIR code using mlir-opt
and various MLIR passes.
"""

import logging
from typing import Any

from ..config import MLIRConfig
from ..utils.subprocess_runner import MLIRToolRunner

logger = logging.getLogger(__name__)


def apply_pass(config: MLIRConfig, mlir_code: str, pass_name: str) -> dict[str, Any]:
    """Apply a single MLIR pass to code.

    Args:
        config: MLIR configuration.
        mlir_code: MLIR code as string.
        pass_name: Name of the pass to apply (e.g., "canonicalize", "cse").

    Returns:
        Dictionary containing:
        - success: Whether transformation succeeded
        - mlir_code: Transformed MLIR code
        - pass_applied: Name of pass applied
        - errors: List of errors (if failed)
    """
    try:
        runner = MLIRToolRunner(config)
        transformed = runner.run_mlir_opt(mlir_code, pass_name)

        return {
            "success": True,
            "mlir_code": transformed,
            "pass_applied": pass_name,
        }

    except FileNotFoundError as e:
        return {
            "success": False,
            "errors": [{"message": str(e), "severity": "error"}],
        }
    except RuntimeError as e:
        logger.exception("Error applying pass")
        return {
            "success": False,
            "errors": [{"message": str(e), "severity": "error"}],
        }


def apply_pass_pipeline(
    config: MLIRConfig, mlir_code: str, pipeline: str
) -> dict[str, Any]:
    """Apply a pipeline of MLIR passes.

    Args:
        config: MLIR configuration.
        mlir_code: MLIR code as string.
        pipeline: Pass pipeline specification (e.g., "canonicalize,cse,inline").

    Returns:
        Dictionary containing:
        - success: Whether transformation succeeded
        - mlir_code: Transformed MLIR code
        - pipeline: Pipeline applied
        - errors: List of errors (if failed)
    """
    try:
        runner = MLIRToolRunner(config)
        transformed = runner.run_mlir_opt(mlir_code, pipeline)

        return {
            "success": True,
            "mlir_code": transformed,
            "pipeline": pipeline,
        }

    except FileNotFoundError as e:
        return {
            "success": False,
            "errors": [{"message": str(e), "severity": "error"}],
        }
    except RuntimeError as e:
        logger.exception("Error applying pipeline")
        return {
            "success": False,
            "errors": [{"message": str(e), "severity": "error"}],
        }


def canonicalize(config: MLIRConfig, mlir_code: str) -> dict[str, Any]:
    """Canonicalize MLIR code.

    Applies the canonicalize pass to simplify operations using
    folding and rewrite patterns.

    Args:
        config: MLIR configuration.
        mlir_code: MLIR code as string.

    Returns:
        Dictionary with canonicalized MLIR code or errors.
    """
    return apply_pass(config, mlir_code, "canonicalize")


def cse(config: MLIRConfig, mlir_code: str) -> dict[str, Any]:
    """Apply Common Subexpression Elimination.

    Args:
        config: MLIR configuration.
        mlir_code: MLIR code as string.

    Returns:
        Dictionary with transformed MLIR code or errors.
    """
    return apply_pass(config, mlir_code, "cse")


def inline_functions(config: MLIRConfig, mlir_code: str) -> dict[str, Any]:
    """Inline function calls.

    Args:
        config: MLIR configuration.
        mlir_code: MLIR code as string.

    Returns:
        Dictionary with transformed MLIR code or errors.
    """
    return apply_pass(config, mlir_code, "inline")
