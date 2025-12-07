"""MLIR generation tools for MCP server.

This module implements tools for generating MLIR code programmatically
using IR builders and templates.
"""

import logging
from typing import Any, Optional

from ..config import MLIRConfig
from ..mlir_wrapper import ir_builder

logger = logging.getLogger(__name__)


def create_function(
    config: MLIRConfig,
    name: str,
    arg_types: list[str],
    result_types: list[str],
    add_return: bool = True,
) -> dict[str, Any]:
    """Generate an MLIR function with specified signature.

    Args:
        config: MLIR configuration.
        name: Function name.
        arg_types: List of argument type strings (e.g., ["i32", "i64"]).
        result_types: List of result type strings.
        add_return: Whether to add an empty return statement.

    Returns:
        Dictionary containing:
        - success: Whether generation succeeded
        - mlir_code: Generated MLIR function code
        - errors: List of errors (if failed)
    """
    try:
        mlir_code = ir_builder.build_function(
            name=name,
            arg_types=arg_types,
            result_types=result_types,
        )

        return {
            "success": True,
            "mlir_code": mlir_code,
            "function_name": name,
        }

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
    except Exception as e:
        logger.exception("Error in create_function")
        return {
            "success": False,
            "errors": [{"message": f"Generation failed: {str(e)}", "severity": "error"}],
        }


def create_operation(
    config: MLIRConfig,
    op_name: str,
    operands: Optional[list[str]] = None,
    result_types: Optional[list[str]] = None,
    attributes: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Generate a specific MLIR operation.

    Args:
        config: MLIR configuration.
        op_name: Operation name (e.g., "arith.addi", "func.call").
        operands: List of operand names or SSA values.
        result_types: List of result type strings.
        attributes: Optional dictionary of operation attributes.

    Returns:
        Dictionary containing:
        - success: Whether generation succeeded
        - mlir_code: Generated operation code
        - errors: List of errors (if failed)
    """
    try:
        operands = operands or []
        result_types = result_types or []
        attributes = attributes or {}

        mlir_code = ir_builder.build_operation(
            op_name=op_name,
            operands=operands,
            result_types=result_types,
            attributes=attributes,
        )

        return {
            "success": True,
            "mlir_code": mlir_code,
            "operation": op_name,
        }

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
    except Exception as e:
        logger.exception("Error in create_operation")
        return {
            "success": False,
            "errors": [{"message": f"Generation failed: {str(e)}", "severity": "error"}],
        }


# Template definitions
TEMPLATES = {
    "simple_function": """module {{
  func.func @{name}({args}) -> {results} {{
    func.return {return_vals} : {results}
  }}
}}""",
    "add_function": """module {{
  func.func @add(%arg0: {type}, %arg1: {type}) -> {type} {{
    %result = arith.addi %arg0, %arg1 : {type}
    func.return %result : {type}
  }}
}}""",
    "mul_function": """module {{
  func.func @mul(%arg0: {type}, %arg1: {type}) -> {type} {{
    %result = arith.muli %arg0, %arg1 : {type}
    func.return %result : {type}
  }}
}}""",
    "constant": """module {{
  func.func @constant() -> {type} {{
    %c = arith.constant {value} : {type}
    func.return %c : {type}
  }}
}}""",
    "if_else": """module {{
  func.func @conditional(%cond: i1, %true_val: {type}, %false_val: {type}) -> {type} {{
    %result = scf.if %cond -> ({type}) {{
      scf.yield %true_val : {type}
    }} else {{
      scf.yield %false_val : {type}
    }}
    func.return %result : {type}
  }}
}}""",
    "for_loop": """module {{
  func.func @loop(%lb: index, %ub: index, %step: index) -> index {{
    %sum = arith.constant 0 : index
    %result = scf.for %i = %lb to %ub step %step iter_args(%iter = %sum) -> (index) {{
      %next = arith.addi %iter, %i : index
      scf.yield %next : index
    }}
    func.return %result : index
  }}
}}""",
}


def generate_from_template(
    config: MLIRConfig,
    template_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Generate MLIR code from a template with parameters.

    Args:
        config: MLIR configuration.
        template_name: Name of the template to use.
        params: Dictionary of template parameters.

    Returns:
        Dictionary containing:
        - success: Whether generation succeeded
        - mlir_code: Generated MLIR code
        - template_used: Name of template used
        - errors: List of errors (if failed)
    """
    try:
        if template_name not in TEMPLATES:
            available = ", ".join(TEMPLATES.keys())
            return {
                "success": False,
                "errors": [
                    {
                        "message": f"Unknown template: {template_name}. "
                        f"Available templates: {available}",
                        "severity": "error",
                    }
                ],
            }

        template = TEMPLATES[template_name]

        # Substitute parameters
        try:
            mlir_code = template.format(**params)
        except KeyError as e:
            return {
                "success": False,
                "errors": [
                    {
                        "message": f"Missing template parameter: {e}",
                        "severity": "error",
                    }
                ],
            }

        return {
            "success": True,
            "mlir_code": mlir_code,
            "template_used": template_name,
        }

    except Exception as e:
        logger.exception("Error in generate_from_template")
        return {
            "success": False,
            "errors": [{"message": f"Generation failed: {str(e)}", "severity": "error"}],
        }


def list_templates(config: MLIRConfig) -> dict[str, Any]:
    """List available MLIR generation templates.

    Args:
        config: MLIR configuration.

    Returns:
        Dictionary containing:
        - templates: List of template information
    """
    templates_info = []

    template_docs = {
        "simple_function": "Simple function with custom signature",
        "add_function": "Integer addition function",
        "mul_function": "Integer multiplication function",
        "constant": "Function returning a constant value",
        "if_else": "Conditional function with if-else",
        "for_loop": "Loop function with accumulation",
    }

    for name, template in TEMPLATES.items():
        # Extract parameter placeholders
        import re

        params = re.findall(r"\{(\w+)\}", template)
        unique_params = sorted(set(params))

        templates_info.append(
            {
                "name": name,
                "description": template_docs.get(name, "No description"),
                "parameters": unique_params,
            }
        )

    return {
        "success": True,
        "templates": templates_info,
        "count": len(templates_info),
    }
