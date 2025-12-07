"""Tests for MLIR analysis tools."""

from pathlib import Path

import pytest

from mlir_mcp_server.config import MLIRConfig
from mlir_mcp_server.tools.analyzer import (
    count_operations,
    extract_function,
    get_operation_operands,
)

from .utils import get_real_mlir_config

# Get fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def config() -> MLIRConfig:
    """Create configuration using real MLIR installation."""
    return get_real_mlir_config()


@pytest.fixture
def simple_mlir() -> str:
    """Load simple MLIR code."""
    return (FIXTURES_DIR / "valid_simple.mlir").read_text()


@pytest.fixture
def complex_mlir() -> str:
    """Load complex MLIR code."""
    return (FIXTURES_DIR / "valid_complex.mlir").read_text()


class TestCountOperations:
    """Tests for count_operations function."""

    def test_count_operations_simple(self, config: MLIRConfig, simple_mlir: str) -> None:
        """Test counting operations in simple MLIR."""
        result = count_operations(config, simple_mlir)

        assert result["success"] is True
        assert "operation_counts" in result
        assert "total_operations" in result
        assert result["total_operations"] > 0

        # Should have at least func.func and some arithmetic operations
        counts = result["operation_counts"]
        assert len(counts) > 0

    def test_count_operations_complex(
        self, config: MLIRConfig, complex_mlir: str
    ) -> None:
        """Test counting operations in complex MLIR."""
        result = count_operations(config, complex_mlir)

        assert result["success"] is True
        assert result["total_operations"] > 5  # Complex has multiple operations

    def test_count_operations_empty(self, config: MLIRConfig) -> None:
        """Test counting operations in empty module."""
        result = count_operations(config, "module {}")

        assert result["success"] is True
        assert result["total_operations"] >= 0


class TestExtractFunction:
    """Tests for extract_function function."""

    def test_extract_existing_function(
        self, config: MLIRConfig, simple_mlir: str
    ) -> None:
        """Test extracting an existing function."""
        result = extract_function(config, simple_mlir, "add")

        assert result["success"] is True
        assert "function_code" in result
        assert result["function_name"] == "add"
        assert "@add" in result["function_code"]

    def test_extract_nonexistent_function(
        self, config: MLIRConfig, simple_mlir: str
    ) -> None:
        """Test extracting a function that doesn't exist."""
        result = extract_function(config, simple_mlir, "nonexistent")

        assert result["success"] is False
        assert "errors" in result
        assert "not found" in result["errors"][0]["message"]

    def test_extract_from_complex(self, config: MLIRConfig, complex_mlir: str) -> None:
        """Test extracting function from complex MLIR."""
        result = extract_function(config, complex_mlir, "conditional")

        assert result["success"] is True
        assert "@conditional" in result["function_code"]


class TestGetOperationOperands:
    """Tests for get_operation_operands function."""

    def test_get_all_operation_operands(
        self, config: MLIRConfig, simple_mlir: str
    ) -> None:
        """Test getting operands for all operations."""
        result = get_operation_operands(config, simple_mlir)

        assert result["success"] is True
        assert "operations" in result
        assert len(result["operations"]) > 0

        # Check structure of operation info
        if result["operations"]:
            op = result["operations"][0]
            assert "name" in op
            assert "num_operands" in op
            assert "operands" in op
            assert "num_results" in op

    def test_get_filtered_operation_operands(
        self, config: MLIRConfig, simple_mlir: str
    ) -> None:
        """Test getting operands for filtered operations."""
        result = get_operation_operands(config, simple_mlir, "arith.addi")

        assert result["success"] is True
        assert "operations" in result
        assert result["filter"] == "arith.addi"

        # All returned operations should match filter
        for op in result["operations"]:
            assert "arith.addi" in op["name"]

    def test_get_operation_operands_complex(
        self, config: MLIRConfig, complex_mlir: str
    ) -> None:
        """Test getting operands from complex MLIR."""
        result = get_operation_operands(config, complex_mlir)

        assert result["success"] is True
        assert len(result["operations"]) > 5  # Complex has many operations
