"""Tests for MLIR transformation tools."""

from pathlib import Path

import pytest

from mlir_mcp_server.config import MLIRConfig
from mlir_mcp_server.tools.transformer import (
    apply_pass,
    apply_pass_pipeline,
    canonicalize,
    cse,
    inline_functions,
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
    """Load simple MLIR code for transformation."""
    return (FIXTURES_DIR / "valid_simple.mlir").read_text()


@pytest.fixture
def complex_mlir() -> str:
    """Load complex MLIR code for transformation."""
    return (FIXTURES_DIR / "valid_complex.mlir").read_text()


class TestApplyPass:
    """Tests for apply_pass function."""

    def test_apply_canonicalize_pass(self, config: MLIRConfig, simple_mlir: str) -> None:
        """Test applying canonicalize pass."""
        result = apply_pass(config, simple_mlir, "canonicalize")

        assert result["success"] is True
        assert "mlir_code" in result
        assert result["pass_applied"] == "canonicalize"
        # Canonicalized code should still be valid MLIR
        assert "func.func" in result["mlir_code"] or "module" in result["mlir_code"]

    def test_apply_cse_pass(self, config: MLIRConfig, simple_mlir: str) -> None:
        """Test applying CSE pass."""
        result = apply_pass(config, simple_mlir, "cse")

        assert result["success"] is True
        assert "mlir_code" in result
        assert result["pass_applied"] == "cse"

    def test_apply_invalid_pass(self, config: MLIRConfig, simple_mlir: str) -> None:
        """Test applying non-existent pass."""
        result = apply_pass(config, simple_mlir, "nonexistent-pass-xyz")

        assert result["success"] is False
        assert "errors" in result


class TestApplyPassPipeline:
    """Tests for apply_pass_pipeline function."""

    def test_apply_simple_pipeline(self, config: MLIRConfig, simple_mlir: str) -> None:
        """Test applying a simple pass pipeline."""
        result = apply_pass_pipeline(config, simple_mlir, "canonicalize,cse")

        assert result["success"] is True
        assert "mlir_code" in result
        assert result["pipeline"] == "canonicalize,cse"

    def test_apply_single_pass_pipeline(
        self, config: MLIRConfig, simple_mlir: str
    ) -> None:
        """Test pipeline with single pass."""
        result = apply_pass_pipeline(config, simple_mlir, "canonicalize")

        assert result["success"] is True
        assert "mlir_code" in result

    def test_apply_complex_pipeline(self, config: MLIRConfig, complex_mlir: str) -> None:
        """Test applying pipeline to complex MLIR."""
        result = apply_pass_pipeline(config, complex_mlir, "canonicalize,cse,inline")

        assert result["success"] is True
        assert "mlir_code" in result


class TestCanonicalize:
    """Tests for canonicalize convenience function."""

    def test_canonicalize(self, config: MLIRConfig, simple_mlir: str) -> None:
        """Test canonicalize convenience function."""
        result = canonicalize(config, simple_mlir)

        assert result["success"] is True
        assert "mlir_code" in result
        assert result["pass_applied"] == "canonicalize"


class TestCSE:
    """Tests for cse convenience function."""

    def test_cse(self, config: MLIRConfig, simple_mlir: str) -> None:
        """Test CSE convenience function."""
        result = cse(config, simple_mlir)

        assert result["success"] is True
        assert "mlir_code" in result
        assert result["pass_applied"] == "cse"


class TestInlineFunctions:
    """Tests for inline_functions convenience function."""

    def test_inline_functions(self, config: MLIRConfig, complex_mlir: str) -> None:
        """Test inline convenience function."""
        result = inline_functions(config, complex_mlir)

        assert result["success"] is True
        assert "mlir_code" in result
        assert result["pass_applied"] == "inline"
