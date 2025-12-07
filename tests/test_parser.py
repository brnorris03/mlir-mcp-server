"""Tests for MLIR parsing tools."""

from pathlib import Path

import pytest

from mlir_mcp_server.config import MLIRConfig
from mlir_mcp_server.tools.parser import get_module_info, parse_mlir, validate_mlir

# Get fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def config(tmp_path: Path) -> MLIRConfig:
    """Create a test configuration with mock toolchain."""
    toolchain_dir = tmp_path / "bin"
    toolchain_dir.mkdir()
    (toolchain_dir / "mlir-opt").touch()
    (toolchain_dir / "mlir-translate").touch()
    return MLIRConfig(toolchain_path=str(toolchain_dir))


@pytest.fixture
def valid_simple_mlir() -> str:
    """Load valid simple MLIR fixture."""
    return (FIXTURES_DIR / "valid_simple.mlir").read_text()


@pytest.fixture
def valid_complex_mlir() -> str:
    """Load valid complex MLIR fixture."""
    return (FIXTURES_DIR / "valid_complex.mlir").read_text()


@pytest.fixture
def invalid_syntax_mlir() -> str:
    """Load invalid syntax MLIR fixture."""
    return (FIXTURES_DIR / "invalid_syntax.mlir").read_text()


class TestParseMlir:
    """Tests for parse_mlir function."""

    def test_parse_valid_simple(self, config: MLIRConfig, valid_simple_mlir: str) -> None:
        """Test parsing valid simple MLIR code."""
        result = parse_mlir(config, valid_simple_mlir)

        assert result["success"] is True
        assert "module_str" in result
        assert "operations" in result
        assert result["operation_count"] >= 1

    def test_parse_valid_complex(self, config: MLIRConfig, valid_complex_mlir: str) -> None:
        """Test parsing valid complex MLIR code."""
        result = parse_mlir(config, valid_complex_mlir)

        assert result["success"] is True
        assert result["operation_count"] >= 2  # At least 2 functions

    def test_parse_invalid_syntax(self, config: MLIRConfig, invalid_syntax_mlir: str) -> None:
        """Test parsing invalid MLIR syntax."""
        result = parse_mlir(config, invalid_syntax_mlir)

        assert result["success"] is False
        assert "errors" in result
        assert len(result["errors"]) > 0

    def test_parse_empty_string(self, config: MLIRConfig) -> None:
        """Test parsing empty string.

        Note: MLIR parses empty string as a valid (empty) module.
        """
        result = parse_mlir(config, "")

        # Empty string is valid in MLIR (empty module)
        assert result["success"] is True
        assert result["operation_count"] == 0


class TestValidateMlir:
    """Tests for validate_mlir function."""

    def test_validate_valid_code(self, config: MLIRConfig, valid_simple_mlir: str) -> None:
        """Test validating valid MLIR code."""
        result = validate_mlir(config, valid_simple_mlir)

        assert result["valid"] is True
        assert "message" in result

    def test_validate_invalid_code(self, config: MLIRConfig, invalid_syntax_mlir: str) -> None:
        """Test validating invalid MLIR code."""
        result = validate_mlir(config, invalid_syntax_mlir)

        assert result["valid"] is False
        assert "errors" in result
        assert len(result["errors"]) > 0


class TestGetModuleInfo:
    """Tests for get_module_info function."""

    def test_get_info_simple(self, config: MLIRConfig, valid_simple_mlir: str) -> None:
        """Test extracting info from simple MLIR module."""
        result = get_module_info(config, valid_simple_mlir)

        assert result["success"] is True
        assert "operations" in result
        assert result["operation_count"] >= 1

        # Check that operations have expected fields
        if result["operations"]:
            op = result["operations"][0]
            assert "name" in op
            assert "num_operands" in op
            assert "num_results" in op
            assert "num_regions" in op

    def test_get_info_complex(self, config: MLIRConfig, valid_complex_mlir: str) -> None:
        """Test extracting info from complex MLIR module."""
        result = get_module_info(config, valid_complex_mlir)

        assert result["success"] is True
        assert result["operation_count"] >= 2

        # Verify operation details are extracted
        for op in result["operations"]:
            assert "name" in op
            assert "result_types" in op

    def test_get_info_invalid(self, config: MLIRConfig, invalid_syntax_mlir: str) -> None:
        """Test extracting info from invalid MLIR code."""
        result = get_module_info(config, invalid_syntax_mlir)

        assert result["success"] is False
        assert "errors" in result
