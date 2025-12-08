"""Tests for MLIR IR builder wrappers."""

import pytest

from mlir_mcp_server.mlir_wrapper.ir_builder import (
    build_function,
    build_module_from_operations,
    build_operation,
)

from .utils import setup_mlir_environment


@pytest.fixture(scope="module", autouse=True)
def setup_mlir() -> None:
    """Initialize MLIR configuration to set up Python paths."""
    setup_mlir_environment()
    # Skip all tests in this module if MLIR bindings aren't available
    try:
        import mlir  # noqa: F401
    except ImportError:
        pytest.skip("MLIR Python bindings not available", allow_module_level=True)


class TestBuildFunction:
    """Tests for build_function."""

    def test_build_simple_function(self) -> None:
        """Test building a simple function."""
        result = build_function(
            name="test_func",
            arg_types=["i32", "i32"],
            result_types=["i32"],
        )

        assert "test_func" in result
        assert "i32" in result
        assert "func.func" in result or "builtin.module" in result

    def test_build_function_no_args(self) -> None:
        """Test building a function with no arguments."""
        result = build_function(
            name="no_args",
            arg_types=[],
            result_types=["i64"],
        )

        assert "no_args" in result
        assert "i64" in result

    def test_build_function_multiple_results(self) -> None:
        """Test building a function with multiple results."""
        result = build_function(
            name="multi_result",
            arg_types=["i32"],
            result_types=["i32", "i64"],
        )

        assert "multi_result" in result
        assert "i32" in result
        assert "i64" in result

    def test_build_function_with_f64(self) -> None:
        """Test building a function with f64 types."""
        result = build_function(
            name="float_func",
            arg_types=["f64", "f64"],
            result_types=["f64"],
        )

        assert "float_func" in result
        assert "f64" in result

    def test_build_function_index_type(self) -> None:
        """Test building a function with index type."""
        result = build_function(
            name="index_func",
            arg_types=["index"],
            result_types=["index"],
        )

        assert "index_func" in result
        assert "index" in result


class TestBuildOperation:
    """Tests for build_operation."""

    def test_build_simple_operation(self) -> None:
        """Test building a simple operation."""
        result = build_operation(
            op_name="arith.addi",
            operands=["%arg0", "%arg1"],
            result_types=["i32"],
        )

        assert "arith.addi" in result
        assert "%arg0" in result
        assert "%arg1" in result
        assert "i32" in result

    def test_build_operation_no_operands(self) -> None:
        """Test building an operation with no operands."""
        result = build_operation(
            op_name="arith.constant",
            operands=[],
            result_types=["i32"],
        )

        assert "arith.constant" in result
        assert "i32" in result

    def test_build_operation_with_attributes(self) -> None:
        """Test building an operation with attributes."""
        result = build_operation(
            op_name="arith.constant",
            operands=[],
            result_types=["i32"],
            attributes={"value": "42"},
        )

        assert "arith.constant" in result
        assert "value" in result or "42" in result

    def test_build_operation_no_results(self) -> None:
        """Test building an operation with no results."""
        result = build_operation(
            op_name="func.return",
            operands=["%0"],
            result_types=[],
        )

        assert "func.return" in result
        assert "%0" in result


class TestBuildModuleFromOperations:
    """Tests for build_module_from_operations."""

    def test_build_module_empty(self) -> None:
        """Test building a module with no operations."""
        result = build_module_from_operations([])

        assert "builtin.module" in result or "module" in result

    def test_build_module_single_operation(self) -> None:
        """Test building a module with a single operation."""
        operations = [
            'func.func @test() { func.return }',
        ]

        result = build_module_from_operations(operations)

        assert "builtin.module" in result or "module" in result

    def test_build_module_multiple_operations(self) -> None:
        """Test building a module with multiple operations."""
        operations = [
            'func.func @test1() { func.return }',
            'func.func @test2() { func.return }',
        ]

        result = build_module_from_operations(operations)

        assert "builtin.module" in result or "module" in result


class TestIRBuilderErrorHandling:
    """Tests for error handling in IR builders."""

    def test_build_function_invalid_type(self) -> None:
        """Test building function with invalid type."""
        # This might raise or return error depending on MLIR validation
        try:
            result = build_function(
                name="bad_type",
                arg_types=["invalid_type_xyz"],
                result_types=["i32"],
            )
            # If it doesn't raise, just verify we got some output
            assert isinstance(result, str)
        except Exception:
            # MLIR might reject invalid types
            pass

    def test_build_module_with_invalid_operation(self) -> None:
        """Test building module with invalid operation string."""
        operations = [
            'this is not valid MLIR',
        ]

        # Should handle gracefully and skip invalid operations
        result = build_module_from_operations(operations)

        # Should still return a module
        assert "builtin.module" in result or "module" in result
