"""MLIR translation tools for MCP server.

This module implements tools for translating between MLIR and other representations
like LLVM IR using mlir-translate.
"""

import logging
from typing import Any

from ..config import MLIRConfig
from ..utils.subprocess_runner import MLIRToolRunner

logger = logging.getLogger(__name__)


def translate_to_llvmir(
    config: MLIRConfig,
    mlir_code: str,
) -> dict[str, Any]:
    """Translate MLIR code to LLVM IR.

    Args:
        config: MLIR configuration.
        mlir_code: MLIR code as string (must use LLVM dialect).

    Returns:
        Dictionary containing:
        - success: Whether translation succeeded
        - llvm_ir: Generated LLVM IR code
        - errors: List of errors (if failed)
    """
    try:
        runner = MLIRToolRunner(config)
        llvm_ir = runner.run_mlir_translate(mlir_code, ["--mlir-to-llvmir"])

        return {
            "success": True,
            "llvm_ir": llvm_ir,
        }

    except FileNotFoundError as e:
        return {
            "success": False,
            "errors": [
                {
                    "message": f"mlir-translate not found: {e}",
                    "severity": "error",
                }
            ],
        }
    except RuntimeError as e:
        error_msg = str(e)
        return {
            "success": False,
            "errors": [
                {
                    "message": f"Translation failed: {error_msg}",
                    "severity": "error",
                }
            ],
        }
    except Exception as e:
        logger.exception("Error in translate_to_llvmir")
        return {
            "success": False,
            "errors": [
                {
                    "message": f"Unexpected error: {str(e)}",
                    "severity": "error",
                }
            ],
        }


def translate_from_llvmir(
    config: MLIRConfig,
    llvm_ir: str,
) -> dict[str, Any]:
    """Translate LLVM IR to MLIR (LLVM dialect).

    Args:
        config: MLIR configuration.
        llvm_ir: LLVM IR code as string.

    Returns:
        Dictionary containing:
        - success: Whether translation succeeded
        - mlir_code: Generated MLIR code
        - errors: List of errors (if failed)
    """
    try:
        runner = MLIRToolRunner(config)
        mlir_code = runner.run_mlir_translate(llvm_ir, ["--import-llvm"])

        return {
            "success": True,
            "mlir_code": mlir_code,
        }

    except FileNotFoundError as e:
        return {
            "success": False,
            "errors": [
                {
                    "message": f"mlir-translate not found: {e}",
                    "severity": "error",
                }
            ],
        }
    except RuntimeError as e:
        error_msg = str(e)
        return {
            "success": False,
            "errors": [
                {
                    "message": f"Translation failed: {error_msg}",
                    "severity": "error",
                }
            ],
        }
    except Exception as e:
        logger.exception("Error in translate_from_llvmir")
        return {
            "success": False,
            "errors": [
                {
                    "message": f"Unexpected error: {str(e)}",
                    "severity": "error",
                }
            ],
        }


def get_translation_info(config: MLIRConfig) -> dict[str, Any]:
    """Get information about available translation capabilities.

    Args:
        config: MLIR configuration.

    Returns:
        Dictionary containing:
        - mlir_translate_available: Whether mlir-translate is available
        - mlir_translate_path: Path to mlir-translate
        - supported_translations: List of supported translation types
    """
    tool_status = config.validate_tools()

    supported = []
    if tool_status.get("mlir_translate", False):
        supported.extend(
            [
                {
                    "name": "mlir-to-llvmir",
                    "description": "Convert MLIR (LLVM dialect) to LLVM IR",
                    "direction": "mlir -> llvmir",
                },
                {
                    "name": "import-llvm",
                    "description": "Import LLVM IR to MLIR (LLVM dialect)",
                    "direction": "llvmir -> mlir",
                },
            ]
        )

    return {
        "mlir_translate_available": tool_status.get("mlir_translate", False),
        "mlir_translate_path": str(config.mlir_translate),
        "supported_translations": supported,
    }
