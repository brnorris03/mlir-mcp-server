"""Tests for MLIR generation tools."""

from pathlib import Path

import pytest

from mlir_mcp_server.config import MLIRConfig
from mlir_mcp_server.tools.generator import (
    create_function,
    create_operation,
    generate_from_template,
    list_templates,
)


@pytest.fixture
def config(tmp_path: Path) -> MLIRConfig:
    """Create a test configuration with mock toolchain."""
    toolchain_dir = tmp_path / "bin"
    toolchain_dir.mkdir()
    (toolchain_dir / "mlir-opt").touch()
    (toolchain_dir / "mlir-translate").touch()
    return MLIRConfig(toolchain_path=str(toolchain_dir))


class TestCreateFunction:
    """Tests for create_function tool."""

    def test_create_simple_function(self, config: MLIRConfig) -> None:
        """Test creating a simple function."""
        result = create_function(
            config=config,
            name="test_func",
            arg_types=["i32", "i32"],
            result_types=["i32"],
        )

        assert result["success"] is True
        assert "mlir_code" in result
        assert "test_func" in result["mlir_code"]
        assert result["function_name"] == "test_func"

    def test_create_function_no_args(self, config: MLIRConfig) -> None:
        """Test creating a function with no arguments."""
        result = create_function(
            config=config,
            name="no_args",
            arg_types=[],
            result_types=["i64"],
        )

        assert result["success"] is True
        assert "no_args" in result["mlir_code"]

    def test_create_function_multiple_results(self, config: MLIRConfig) -> None:
        """Test creating a function with multiple results."""
        result = create_function(
            config=config,
            name="multi_result",
            arg_types=["i32"],
            result_types=["i32", "i32"],
        )

        assert result["success"] is True
        assert "multi_result" in result["mlir_code"]


class TestCreateOperation:
    """Tests for create_operation tool."""

    def test_create_simple_operation(self, config: MLIRConfig) -> None:
        """Test creating a simple operation."""
        result = create_operation(
            config=config,
            op_name="arith.addi",
            operands=["%arg0", "%arg1"],
            result_types=["i32"],
        )

        assert result["success"] is True
        assert "mlir_code" in result
        assert "arith.addi" in result["mlir_code"]
        assert result["operation"] == "arith.addi"

    def test_create_operation_with_attributes(self, config: MLIRConfig) -> None:
        """Test creating an operation with attributes."""
        result = create_operation(
            config=config,
            op_name="arith.constant",
            operands=[],
            result_types=["i32"],
            attributes={"value": "42"},
        )

        assert result["success"] is True
        assert "arith.constant" in result["mlir_code"]


class TestGenerateFromTemplate:
    """Tests for generate_from_template tool."""

    def test_generate_add_function(self, config: MLIRConfig) -> None:
        """Test generating from add_function template."""
        result = generate_from_template(
            config=config,
            template_name="add_function",
            params={"type": "i32"},
        )

        assert result["success"] is True
        assert "mlir_code" in result
        assert "arith.addi" in result["mlir_code"]
        assert "func.func @add" in result["mlir_code"]
        assert result["template_used"] == "add_function"

    def test_generate_constant(self, config: MLIRConfig) -> None:
        """Test generating constant function."""
        result = generate_from_template(
            config=config,
            template_name="constant",
            params={"type": "i64", "value": "100"},
        )

        assert result["success"] is True
        assert "arith.constant 100" in result["mlir_code"]

    def test_generate_if_else(self, config: MLIRConfig) -> None:
        """Test generating if-else function."""
        result = generate_from_template(
            config=config,
            template_name="if_else",
            params={"type": "i32"},
        )

        assert result["success"] is True
        assert "scf.if" in result["mlir_code"]
        assert "scf.yield" in result["mlir_code"]

    def test_generate_for_loop(self, config: MLIRConfig) -> None:
        """Test generating for loop function."""
        result = generate_from_template(
            config=config,
            template_name="for_loop",
            params={},
        )

        assert result["success"] is True
        assert "scf.for" in result["mlir_code"]

    def test_generate_unknown_template(self, config: MLIRConfig) -> None:
        """Test generating from unknown template."""
        result = generate_from_template(
            config=config,
            template_name="nonexistent",
            params={},
        )

        assert result["success"] is False
        assert "errors" in result
        assert "Unknown template" in result["errors"][0]["message"]

    def test_generate_missing_params(self, config: MLIRConfig) -> None:
        """Test generating with missing required parameters."""
        result = generate_from_template(
            config=config,
            template_name="add_function",
            params={},  # Missing "type" parameter
        )

        assert result["success"] is False
        assert "errors" in result
        assert "Missing template parameter" in result["errors"][0]["message"]


class TestListTemplates:
    """Tests for list_templates tool."""

    def test_list_templates(self, config: MLIRConfig) -> None:
        """Test listing available templates."""
        result = list_templates(config)

        assert result["success"] is True
        assert "templates" in result
        assert result["count"] > 0

        # Check template structure
        for template in result["templates"]:
            assert "name" in template
            assert "description" in template
            assert "parameters" in template

    def test_list_templates_includes_known_templates(self, config: MLIRConfig) -> None:
        """Test that known templates are in the list."""
        result = list_templates(config)

        template_names = [t["name"] for t in result["templates"]]
        assert "add_function" in template_names
        assert "constant" in template_names
        assert "if_else" in template_names
        assert "for_loop" in template_names
