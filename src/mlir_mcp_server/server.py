"""MLIR MCP Server implementation.

This module implements the MCP server using the FastMCP framework,
registering tools, resources, and prompts for MLIR manipulation.
"""

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from .config import MLIRConfig
from .tools import analyzer, generator, parser, transformer, translator

logger = logging.getLogger(__name__)


def create_server(config: MLIRConfig) -> FastMCP:
    """Create and configure the MLIR MCP server.

    Args:
        config: MLIR configuration with validated toolchain path.

    Returns:
        Configured FastMCP server instance.
    """
    # Create FastMCP server
    mcp = FastMCP("mlir-mcp-server")

    # Store config for access in tools
    mcp.mlir_config = config  # type: ignore

    # Register a simple ping tool for health checking
    @mcp.tool()
    def ping() -> dict[str, str]:
        """Health check endpoint that returns server status.

        Returns:
            Dictionary with server status and toolchain information.
        """
        return {
            "status": "ok",
            "server": "mlir-mcp-server",
            "version": "0.1.0",
            "toolchain_path": str(config.toolchain_path),
        }

    # Register a toolchain_info tool
    @mcp.tool()
    def toolchain_info() -> dict[str, Any]:
        """Get information about the configured MLIR toolchain.

        Returns:
            Dictionary with toolchain path and tool availability.
        """
        tool_status = config.validate_tools()
        return {
            "toolchain_path": str(config.toolchain_path),
            "tools": {
                "mlir_opt": {"path": str(config.mlir_opt), "available": tool_status["mlir_opt"]},
                "mlir_translate": {
                    "path": str(config.mlir_translate),
                    "available": tool_status["mlir_translate"],
                },
                "mlir_reduce": {
                    "path": str(config.mlir_reduce),
                    "available": tool_status["mlir_reduce"],
                },
                "mlir_query": {
                    "path": str(config.mlir_query),
                    "available": tool_status["mlir_query"],
                },
                "mlir_runner": {
                    "path": str(config.mlir_runner),
                    "available": tool_status["mlir_runner"],
                },
            },
        }

    # Register MLIR parsing tools
    @mcp.tool()
    def parse_mlir(mlir_code: str) -> dict[str, Any]:
        """Parse MLIR code and return module information.

        Args:
            mlir_code: MLIR code as a string.

        Returns:
            Dictionary with parsing results including operations and any errors.
        """
        return parser.parse_mlir(config, mlir_code)

    @mcp.tool()
    def validate_mlir(mlir_code: str) -> dict[str, Any]:
        """Validate MLIR code syntax and semantics.

        Args:
            mlir_code: MLIR code as a string.

        Returns:
            Dictionary with validation results.
        """
        return parser.validate_mlir(config, mlir_code)

    @mcp.tool()
    def get_module_info(mlir_code: str) -> dict[str, Any]:
        """Extract detailed structural information from MLIR code.

        Args:
            mlir_code: MLIR code as a string.

        Returns:
            Dictionary with detailed module information including operations, regions, and blocks.
        """
        return parser.get_module_info(config, mlir_code)

    # Register MLIR generation tools
    @mcp.tool()
    def create_function(
        name: str,
        arg_types: list[str],
        result_types: list[str],
    ) -> dict[str, Any]:
        """Generate an MLIR function with specified signature.

        Args:
            name: Function name.
            arg_types: List of argument type strings (e.g., ["i32", "i64"]).
            result_types: List of result type strings.

        Returns:
            Dictionary with generated MLIR function code.
        """
        return generator.create_function(config, name, arg_types, result_types)

    @mcp.tool()
    def create_operation(
        op_name: str,
        operands: list[str] | None = None,
        result_types: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a specific MLIR operation.

        Args:
            op_name: Operation name (e.g., "arith.addi").
            operands: List of operand names.
            result_types: List of result type strings.
            attributes: Optional operation attributes.

        Returns:
            Dictionary with generated operation code.
        """
        return generator.create_operation(config, op_name, operands, result_types, attributes)

    @mcp.tool()
    def generate_from_template(template_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Generate MLIR code from a template.

        Args:
            template_name: Name of the template (e.g., "add_function", "if_else").
            params: Template parameters (e.g., {"type": "i32", "value": "42"}).

        Returns:
            Dictionary with generated MLIR code.
        """
        return generator.generate_from_template(config, template_name, params)

    @mcp.tool()
    def list_templates() -> dict[str, Any]:
        """List available MLIR generation templates.

        Returns:
            Dictionary with list of templates and their parameters.
        """
        return generator.list_templates(config)

    # Register MLIR transformation tools
    @mcp.tool()
    def apply_pass(mlir_code: str, pass_name: str) -> dict[str, Any]:
        """Apply a single MLIR pass to code.

        Args:
            mlir_code: MLIR code as string.
            pass_name: Name of the pass (e.g., "canonicalize", "cse", "inline").

        Returns:
            Dictionary with transformed MLIR code or errors.
        """
        return transformer.apply_pass(config, mlir_code, pass_name)

    @mcp.tool()
    def apply_pass_pipeline(mlir_code: str, pipeline: str) -> dict[str, Any]:
        """Apply a pipeline of MLIR passes.

        Args:
            mlir_code: MLIR code as string.
            pipeline: Pass pipeline (e.g., "canonicalize,cse,inline").

        Returns:
            Dictionary with transformed MLIR code or errors.
        """
        return transformer.apply_pass_pipeline(config, mlir_code, pipeline)

    @mcp.tool()
    def canonicalize(mlir_code: str) -> dict[str, Any]:
        """Canonicalize MLIR code.

        Args:
            mlir_code: MLIR code as string.

        Returns:
            Dictionary with canonicalized MLIR code or errors.
        """
        return transformer.canonicalize(config, mlir_code)

    # Register MLIR analysis tools
    @mcp.tool()
    def count_operations(mlir_code: str) -> dict[str, Any]:
        """Count operations by type in MLIR code.

        Args:
            mlir_code: MLIR code as string.

        Returns:
            Dictionary with operation counts.
        """
        return analyzer.count_operations(config, mlir_code)

    @mcp.tool()
    def extract_function(mlir_code: str, function_name: str) -> dict[str, Any]:
        """Extract a specific function from MLIR module.

        Args:
            mlir_code: MLIR code as string.
            function_name: Name of function to extract.

        Returns:
            Dictionary with extracted function code.
        """
        return analyzer.extract_function(config, mlir_code, function_name)

    @mcp.tool()
    def get_operation_operands(
        mlir_code: str, op_filter: str | None = None
    ) -> dict[str, Any]:
        """Get operands of operations.

        Args:
            mlir_code: MLIR code as string.
            op_filter: Optional operation name filter.

        Returns:
            Dictionary with operation operand information.
        """
        return analyzer.get_operation_operands(config, mlir_code, op_filter)

    # Register MLIR translation tools
    @mcp.tool()
    def translate_to_llvmir(mlir_code: str) -> dict[str, Any]:
        """Translate MLIR code to LLVM IR.

        Args:
            mlir_code: MLIR code as string (must use LLVM dialect).

        Returns:
            Dictionary with LLVM IR code or errors.
        """
        return translator.translate_to_llvmir(config, mlir_code)

    @mcp.tool()
    def translate_from_llvmir(llvm_ir: str) -> dict[str, Any]:
        """Translate LLVM IR to MLIR (LLVM dialect).

        Args:
            llvm_ir: LLVM IR code as string.

        Returns:
            Dictionary with MLIR code or errors.
        """
        return translator.translate_from_llvmir(config, llvm_ir)

    @mcp.tool()
    def get_translation_info() -> dict[str, Any]:
        """Get information about available translation capabilities.

        Returns:
            Dictionary with translation tool availability and supported formats.
        """
        return translator.get_translation_info(config)

    logger.info("MLIR MCP Server created successfully")
    logger.info(
        "Registered tools: ping, toolchain_info, "
        "parse_mlir, validate_mlir, get_module_info, "
        "create_function, create_operation, generate_from_template, list_templates, "
        "apply_pass, apply_pass_pipeline, canonicalize, "
        "count_operations, extract_function, get_operation_operands, "
        "translate_to_llvmir, translate_from_llvmir, get_translation_info"
    )

    return mcp
