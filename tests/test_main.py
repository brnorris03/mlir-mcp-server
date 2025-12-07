"""Tests for __main__.py server entry point."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from mlir_mcp_server.__main__ import main


class TestMain:
    """Tests for main() entry point."""

    def test_main_with_valid_config(self, tmp_path, monkeypatch):
        """Test main() with valid MLIR configuration."""
        monkeypatch.chdir(tmp_path)

        # Create mock toolchain
        toolchain_dir = tmp_path / "bin"
        toolchain_dir.mkdir()
        (toolchain_dir / "mlir-opt").touch()
        (toolchain_dir / "mlir-translate").touch()

        # Mock the MCP server to avoid actually running it
        mock_server = MagicMock()
        mock_server.run = MagicMock()

        with patch("mlir_mcp_server.__main__.create_server", return_value=mock_server):
            with patch("mlir_mcp_server.__main__.MLIRConfig") as mock_config_cls:
                # Setup mock config
                mock_config = MagicMock()
                mock_config.toolchain_path = toolchain_dir
                mock_config.mlir_opt.exists.return_value = True
                mock_config.mlir_translate.exists.return_value = True
                mock_config.validate_tools.return_value = {
                    "mlir_opt": True,
                    "mlir_translate": True,
                    "mlir_reduce": False,
                    "mlir_query": False,
                    "mlir_runner": False,
                }
                mock_config_cls.return_value = mock_config

                # Run main - should exit normally via mocked run()
                with pytest.raises(SystemExit) as exc_info:
                    with patch.object(mock_server, "run", side_effect=SystemExit(0)):
                        main()

                # Verify it tried to run the server
                assert mock_server.run.called or exc_info.value.code == 0

    def test_main_config_error(self, monkeypatch, capsys):
        """Test main() handles configuration errors."""
        # Mock MLIRConfig to raise RuntimeError
        with patch("mlir_mcp_server.__main__.MLIRConfig") as mock_config:
            mock_config.side_effect = RuntimeError("MLIR toolchain not found")

            # Should exit with error code 1
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

            # Check error message was printed
            captured = capsys.readouterr()
            assert "Fatal error" in captured.err
            assert "MLIR toolchain not found" in captured.err

    def test_main_validation_error(self, tmp_path, monkeypatch, capsys):
        """Test main() handles validation errors."""
        monkeypatch.chdir(tmp_path)

        # Create mock toolchain without required tools
        toolchain_dir = tmp_path / "bin"
        toolchain_dir.mkdir()

        with patch("mlir_mcp_server.__main__.MLIRConfig") as mock_config_cls:
            mock_config = MagicMock()
            mock_config.toolchain_path = toolchain_dir
            # Simulate validation failure
            mock_config.validate_tools.side_effect = RuntimeError(
                "Core MLIR tools not found"
            )
            mock_config_cls.return_value = mock_config

            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Fatal error" in captured.err
