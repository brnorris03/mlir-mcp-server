"""Shared test utilities and helpers."""

import os
from pathlib import Path

import pytest

from mlir_mcp_server.config import MLIRConfig


def is_mlir_available() -> bool:
    """Check if MLIR Python bindings are available.

    Returns:
        True if MLIR bindings can be imported, False otherwise.
    """
    try:
        import mlir  # noqa: F401

        return True
    except ImportError:
        return False


def setup_mlir_environment() -> None:
    """Initialize MLIR configuration to set up Python paths.

    This should be called in module-scoped autouse fixtures to ensure
    PYTHONPATH and library paths are configured before tests run.

    In CI environments, this is skipped to avoid requiring MLIR installation.
    """
    if not os.getenv("CI"):  # Only try to load MLIR locally
        try:
            MLIRConfig()
        except RuntimeError:
            pass  # No MLIR installation, tests will handle gracefully


def create_mock_toolchain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MLIRConfig:
    """Create a mock MLIR toolchain for testing.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        MLIRConfig pointing to mock toolchain.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLIR_TOOLCHAIN_PATH", raising=False)

    toolchain_dir = tmp_path / "bin"
    toolchain_dir.mkdir()
    (toolchain_dir / "mlir-opt").touch()
    (toolchain_dir / "mlir-translate").touch()

    return MLIRConfig(toolchain_path=str(toolchain_dir))


def get_real_mlir_config() -> MLIRConfig:
    """Get real MLIR configuration, skip test if not available.

    Returns:
        MLIRConfig for real MLIR installation.

    Raises:
        pytest.skip: If MLIR tools are not found.
    """
    try:
        config = MLIRConfig()
        if not config.mlir_opt.exists() or not config.mlir_translate.exists():
            pytest.skip("Real MLIR tools (mlir-opt, mlir-translate) not found")
        return config
    except RuntimeError:
        pytest.skip("No MLIR installation found")

    raise RuntimeError("Unreachable")
