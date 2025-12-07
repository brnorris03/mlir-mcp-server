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

from .utils import create_mock_toolchain, is_mlir_available, setup_mlir_environment


@pytest.fixture(scope="module", autouse=True)
def setup_mlir() -> None:
    """Initialize MLIR configuration to set up Python paths."""
    setup_mlir_environment()


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MLIRConfig:
    """Create a test configuration with mock toolchain."""
    return create_mock_toolchain(tmp_path, monkeypatch)


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

        if is_mlir_available():
            assert result["success"] is True
            assert "mlir_code" in result
            assert "test_func" in result["mlir_code"]
            assert result["function_name"] == "test_func"
        else:
            # Without MLIR bindings, should return error
            assert result["success"] is False
            assert "errors" in result
            assert "MLIR Python bindings not available" in result["errors"][0]["message"]

    def test_create_function_no_args(self, config: MLIRConfig) -> None:
        """Test creating a function with no arguments."""
        result = create_function(
            config=config,
            name="no_args",
            arg_types=[],
            result_types=["i64"],
        )

        if is_mlir_available():
            assert result["success"] is True
            assert "no_args" in result["mlir_code"]
        else:
            assert result["success"] is False

    def test_create_function_multiple_results(self, config: MLIRConfig) -> None:
        """Test creating a function with multiple results."""
        result = create_function(
            config=config,
            name="multi_result",
            arg_types=["i32"],
            result_types=["i32", "i32"],
        )

        if is_mlir_available():
            assert result["success"] is True
            assert "multi_result" in result["mlir_code"]
        else:
            assert result["success"] is False


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

        if is_mlir_available():
            assert result["success"] is True
            assert "mlir_code" in result
            assert "arith.addi" in result["mlir_code"]
            assert result["operation"] == "arith.addi"
        else:
            assert result["success"] is False

    def test_create_operation_with_attributes(self, config: MLIRConfig) -> None:
        """Test creating an operation with attributes."""
        result = create_operation(
            config=config,
            op_name="arith.constant",
            operands=[],
            result_types=["i32"],
            attributes={"value": "42"},
        )

        if is_mlir_available():
            assert result["success"] is True
            assert "arith.constant" in result["mlir_code"]
        else:
            assert result["success"] is False


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
        assert "Template parameter not provided" in result["errors"][0]["message"]


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


class TestGeneratorEdgeCases:
    """Tests for edge cases and error handling in generator."""

    def test_create_operation_with_none_values(self, config: MLIRConfig) -> None:
        """Test creating operation with None operands and attributes."""
        result = create_operation(
            config=config,
            op_name="arith.constant",
            operands=None,
            result_types=None,
            attributes=None,
        )

        if is_mlir_available():
            assert result["success"] is True
        else:
            assert result["success"] is False

    def test_generate_mul_function_template(self, config: MLIRConfig) -> None:
        """Test generating multiplication function template."""
        result = generate_from_template(
            config=config,
            template_name="mul_function",
            params={"type": "i64"},
        )

        assert result["success"] is True
        assert "arith.muli" in result["mlir_code"]
        assert "func.func @mul" in result["mlir_code"]

    def test_generate_simple_function_template(self, config: MLIRConfig) -> None:
        """Test generating simple_function template."""
        result = generate_from_template(
            config=config,
            template_name="simple_function",
            params={
                "name": "custom",
                "args": "%arg0: i32",
                "results": "i32",
                "return_vals": "%arg0",
            },
        )

        assert result["success"] is True
        assert "func.func @custom" in result["mlir_code"]

    def test_list_templates_has_all_templates(self, config: MLIRConfig) -> None:
        """Test that all six templates are listed."""
        result = list_templates(config)

        assert result["count"] == 6
        template_names = [t["name"] for t in result["templates"]]
        expected = [
            "simple_function",
            "add_function",
            "mul_function",
            "constant",
            "if_else",
            "for_loop",
        ]
        for expected_name in expected:
            assert expected_name in template_names
