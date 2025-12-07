"""Integration tests using actual MLIR tools.

These tests require a working MLIR installation with mlir-opt and mlir-translate.
They are skipped if MLIR tools are not available.
"""

from pathlib import Path

import pytest

from mlir_mcp_server.config import MLIRConfig
from mlir_mcp_server.tools.parser import get_module_info, parse_mlir, validate_mlir

# Get fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def real_config() -> MLIRConfig:
    """Create configuration using actual MLIR installation.

    This will auto-detect or use configured MLIR installation.
    Tests are skipped if no MLIR installation is found.
    """
    try:
        config = MLIRConfig()
        # Verify core tools exist
        if not config.mlir_opt.exists() or not config.mlir_translate.exists():
            pytest.skip("MLIR tools (mlir-opt, mlir-translate) not found")
        return config
    except RuntimeError:
        pytest.skip("No MLIR installation found")


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


class TestRealMLIRParsing:
    """Integration tests with actual MLIR tools."""

    def test_parse_with_real_mlir(self, real_config: MLIRConfig, valid_simple_mlir: str) -> None:
        """Test parsing using real MLIR Python bindings."""
        result = parse_mlir(real_config, valid_simple_mlir)

        assert result["success"] is True
        assert "module_str" in result
        assert "operations" in result
        assert result["operation_count"] >= 1
        assert "func.func" in str(result["module_str"]) or "@add" in str(result["operations"])

    def test_parse_complex_with_real_mlir(
        self, real_config: MLIRConfig, valid_complex_mlir: str
    ) -> None:
        """Test parsing complex MLIR with control flow."""
        result = parse_mlir(real_config, valid_complex_mlir)

        assert result["success"] is True
        assert result["operation_count"] >= 2  # At least 2 functions

    def test_validate_valid_with_real_mlir(
        self, real_config: MLIRConfig, valid_simple_mlir: str
    ) -> None:
        """Test validation with real MLIR tools."""
        result = validate_mlir(real_config, valid_simple_mlir)

        assert result["valid"] is True
        assert "message" in result

    def test_validate_invalid_with_real_mlir(
        self, real_config: MLIRConfig, invalid_syntax_mlir: str
    ) -> None:
        """Test validation of invalid MLIR."""
        result = validate_mlir(real_config, invalid_syntax_mlir)

        assert result["valid"] is False
        assert "errors" in result
        assert len(result["errors"]) > 0

    def test_get_module_info_with_real_mlir(
        self, real_config: MLIRConfig, valid_simple_mlir: str
    ) -> None:
        """Test extracting module info with real MLIR."""
        result = get_module_info(real_config, valid_simple_mlir)

        assert result["success"] is True
        assert "operations" in result
        assert result["operation_count"] >= 1

        # Verify operation details are extracted
        if result["operations"]:
            op = result["operations"][0]
            assert "name" in op
            assert "num_operands" in op
            assert "num_results" in op
            assert "result_types" in op


class TestRealMLIRToolchain:
    """Integration tests for MLIR toolchain validation."""

    def test_real_toolchain_has_core_tools(self, real_config: MLIRConfig) -> None:
        """Test that real MLIR installation has required core tools."""
        assert real_config.mlir_opt.exists(), f"mlir-opt not found at {real_config.mlir_opt}"
        assert real_config.mlir_translate.exists(), f"mlir-translate not found at {real_config.mlir_translate}"

    def test_validate_real_tools(self, real_config: MLIRConfig) -> None:
        """Test validation of real MLIR toolchain."""
        result = real_config.validate_tools()

        # Core tools must be available
        assert result["mlir_opt"] is True
        assert result["mlir_translate"] is True

        # Optional tools may or may not be available (don't assert)
        # Just verify the keys exist
        assert "mlir_reduce" in result
        assert "mlir_query" in result
        assert "mlir_runner" in result

    def test_config_reports_toolchain_path(self, real_config: MLIRConfig) -> None:
        """Test that config reports the detected toolchain path."""
        assert real_config.toolchain_path is not None
        assert real_config.toolchain_path.exists()
        assert real_config.toolchain_path.is_dir()
