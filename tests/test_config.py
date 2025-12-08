"""Tests for MLIR configuration module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mlir_mcp_server.config import MLIRConfig, auto_detect_mlir_toolchain


class TestAutoDetection:
    """Tests for MLIR toolchain auto-detection."""

    def test_auto_detect_finds_toolchain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that auto-detection finds mlir-opt in common locations."""
        # Clear environment to avoid .env interference
        monkeypatch.delenv("MLIR_TOOLCHAIN_PATH", raising=False)
        monkeypatch.setattr("mlir_mcp_server.config.MLIRConfig.model_config", {"env_file": None})

        # Create a fake toolchain directory
        toolchain_dir = tmp_path / "bin"
        toolchain_dir.mkdir()
        (toolchain_dir / "mlir-opt").touch()

        # Test with mocked auto-detection
        with patch("mlir_mcp_server.config.auto_detect_mlir_toolchain") as mock_detect:
            mock_detect.return_value = toolchain_dir
            config = MLIRConfig()
            assert config.toolchain_path == toolchain_dir

    def test_auto_detect_returns_none_when_not_found(self) -> None:
        """Test that auto-detection returns None when no toolchain found."""
        with patch("mlir_mcp_server.config.Path.exists", return_value=False):
            result = auto_detect_mlir_toolchain()
            assert result is None


class TestMLIRConfig:
    """Tests for MLIRConfig class."""

    def test_config_with_explicit_path(self, tmp_path: Path) -> None:
        """Test configuration with explicitly provided toolchain path."""
        toolchain_dir = tmp_path / "bin"
        toolchain_dir.mkdir()
        (toolchain_dir / "mlir-opt").touch()
        (toolchain_dir / "mlir-translate").touch()

        config = MLIRConfig(toolchain_path=str(toolchain_dir))
        assert config.toolchain_path == toolchain_dir
        assert config.mlir_opt == toolchain_dir / "mlir-opt"
        assert config.mlir_translate == toolchain_dir / "mlir-translate"

    def test_config_with_env_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test configuration from environment variable."""
        toolchain_dir = tmp_path / "bin"
        toolchain_dir.mkdir()
        (toolchain_dir / "mlir-opt").touch()
        (toolchain_dir / "mlir-translate").touch()

        monkeypatch.setenv("MLIR_TOOLCHAIN_PATH", str(toolchain_dir))
        config = MLIRConfig()
        assert config.toolchain_path == toolchain_dir

    def test_config_with_auto_detection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test configuration with auto-detection."""
        # Clear environment and change directory to avoid .env interference
        monkeypatch.delenv("MLIR_TOOLCHAIN_PATH", raising=False)
        monkeypatch.chdir(tmp_path)

        toolchain_dir = tmp_path / "bin"
        toolchain_dir.mkdir()
        (toolchain_dir / "mlir-opt").touch()
        (toolchain_dir / "mlir-translate").touch()

        with patch("mlir_mcp_server.config.auto_detect_mlir_toolchain") as mock_detect:
            mock_detect.return_value = toolchain_dir
            config = MLIRConfig()
            assert config.toolchain_path == toolchain_dir

    def test_config_raises_when_toolchain_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that configuration raises error when toolchain not found."""
        # Clear environment and change directory to avoid .env interference
        monkeypatch.delenv("MLIR_TOOLCHAIN_PATH", raising=False)
        monkeypatch.chdir(tmp_path)

        with patch("mlir_mcp_server.config.auto_detect_mlir_toolchain", return_value=None):
            with pytest.raises(RuntimeError, match="MLIR toolchain not found"):
                MLIRConfig()

    def test_validate_tools_success(self, tmp_path: Path) -> None:
        """Test tool validation with all tools present."""
        toolchain_dir = tmp_path / "bin"
        toolchain_dir.mkdir()
        (toolchain_dir / "mlir-opt").touch()
        (toolchain_dir / "mlir-translate").touch()
        (toolchain_dir / "mlir-reduce").touch()
        (toolchain_dir / "mlir-query").touch()
        (toolchain_dir / "mlir-runner").touch()

        config = MLIRConfig(toolchain_path=str(toolchain_dir))
        result = config.validate_tools()

        assert result["mlir_opt"] is True
        assert result["mlir_translate"] is True
        assert result["mlir_reduce"] is True
        assert result["mlir_query"] is True
        assert result["mlir_runner"] is True

    def test_validate_tools_missing_core_tool(self, tmp_path: Path) -> None:
        """Test that validation raises error when core tools are missing."""
        toolchain_dir = tmp_path / "bin"
        toolchain_dir.mkdir()
        (toolchain_dir / "mlir-opt").touch()
        # mlir-translate is missing

        config = MLIRConfig(toolchain_path=str(toolchain_dir))
        with pytest.raises(RuntimeError, match="Core MLIR tools not found"):
            config.validate_tools()

    def test_validate_tools_missing_optional_tool(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that validation warns about missing optional tools."""
        toolchain_dir = tmp_path / "bin"
        toolchain_dir.mkdir()
        (toolchain_dir / "mlir-opt").touch()
        (toolchain_dir / "mlir-translate").touch()
        # mlir-reduce, mlir-query, mlir-runner are missing

        config = MLIRConfig(toolchain_path=str(toolchain_dir))
        result = config.validate_tools()

        assert result["mlir_opt"] is True
        assert result["mlir_translate"] is True
        assert result["mlir_reduce"] is False
        assert result["mlir_query"] is False
        assert result["mlir_runner"] is False

        # Check that warnings were logged
        assert "mlir-reduce not found" in caplog.text
        assert "mlir-query not found" in caplog.text
        assert "mlir-runner not found" in caplog.text

    def test_toolchain_path_expansion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that tilde in toolchain path is expanded."""
        toolchain_dir = tmp_path / "bin"
        toolchain_dir.mkdir()
        (toolchain_dir / "mlir-opt").touch()
        (toolchain_dir / "mlir-translate").touch()

        # Mock home directory
        monkeypatch.setenv("HOME", str(tmp_path))

        config = MLIRConfig(toolchain_path="~/bin")
        assert config.toolchain_path == toolchain_dir
