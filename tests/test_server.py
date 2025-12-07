"""Tests for MCP server implementation."""

import pytest

from mlir_mcp_server.config import MLIRConfig
from mlir_mcp_server.server import create_server

from .utils import get_real_mlir_config


@pytest.fixture
def config() -> MLIRConfig:
    """Create configuration using real MLIR installation."""
    return get_real_mlir_config()


class TestServerCreation:
    """Tests for server creation and configuration."""

    def test_create_server(self, config: MLIRConfig) -> None:
        """Test creating the MCP server."""
        server = create_server(config)

        assert server is not None
        assert hasattr(server, "name")

    def test_server_has_mlir_config(self, config: MLIRConfig) -> None:
        """Test that server stores config."""
        server = create_server(config)

        assert hasattr(server, "mlir_config")
        assert server.mlir_config == config  # type: ignore


class TestServerTools:
    """Tests for registered MCP tools."""

    @pytest.mark.asyncio
    async def test_all_tools_registered(self, config: MLIRConfig) -> None:
        """Test that all MCP tools are registered."""
        server = create_server(config)

        tools = await server.list_tools()
        tool_names = [t.name for t in tools]

        expected_tools = [
            "ping",
            "toolchain_info",
            "parse_mlir",
            "validate_mlir",
            "get_module_info",
            "create_function",
            "create_operation",
            "generate_from_template",
            "list_templates",
            "apply_pass",
            "apply_pass_pipeline",
            "canonicalize",
            "count_operations",
            "extract_function",
            "get_operation_operands",
        ]

        for expected in expected_tools:
            assert expected in tool_names, f"Tool {expected} not registered"

        assert len(tools) == 15

