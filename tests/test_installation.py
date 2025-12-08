"""Tests for MLIR installation configuration."""

from pathlib import Path

import pytest

from mlir_mcp_server.config import MLIRConfig, MLIRInstallation


class TestMLIRInstallation:
    """Tests for MLIRInstallation dataclass."""

    def test_create_installation(self, tmp_path: Path) -> None:
        """Test creating an MLIR installation object."""
        installation = MLIRInstallation(
            name="test",
            root=tmp_path,
            tool_prefix="test",
        )

        assert installation.name == "test"
        assert installation.root == tmp_path
        assert installation.tool_prefix == "test"
        assert installation.bin_dir == tmp_path / "bin"
        assert installation.python_dir == tmp_path / "python_packages" / "mlir_core"

    def test_get_tool_path_with_prefix(self, tmp_path: Path) -> None:
        """Test get_tool_path returns correctly prefixed tool names."""
        installation = MLIRInstallation(
            name="tt",
            root=tmp_path,
            tool_prefix="tt",
        )

        assert installation.get_tool_path("opt") == tmp_path / "bin" / "tt-opt"
        assert installation.get_tool_path("translate") == tmp_path / "bin" / "tt-translate"
        assert installation.get_tool_path("reduce") == tmp_path / "bin" / "tt-reduce"

    def test_get_tool_path_mlir_prefix(self, tmp_path: Path) -> None:
        """Test get_tool_path with standard mlir prefix."""
        installation = MLIRInstallation(
            name="llvm",
            root=tmp_path,
            tool_prefix="mlir",
        )

        assert installation.get_tool_path("opt") == tmp_path / "bin" / "mlir-opt"
        assert installation.get_tool_path("translate") == tmp_path / "bin" / "mlir-translate"

    def test_validate_tools(self, tmp_path: Path) -> None:
        """Test validation of tool availability."""
        installation = MLIRInstallation(
            name="test",
            root=tmp_path,
            tool_prefix="test",
        )

        # Create directories
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        # Create some tools
        (bin_dir / "test-opt").touch()
        (bin_dir / "test-translate").touch()

        result = installation.validate()

        assert result["opt"] is True
        assert result["translate"] is True
        assert result["reduce"] is False  # Not created
        assert result["query"] is False  # Not created
        assert result["runner"] is False  # Not created


class TestMultipleInstallations:
    """Tests for multiple MLIR installation configuration."""

    def test_config_with_single_installation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test configuration with single installation via MLIR_INSTALLATIONS."""
        monkeypatch.chdir(tmp_path)

        # Create installation structure
        install_root = tmp_path / "llvm"
        install_root.mkdir()

        bin_dir = install_root / "bin"
        bin_dir.mkdir()
        (bin_dir / "mlir-opt").touch()
        (bin_dir / "mlir-translate").touch()

        lib_dir = install_root / "lib"
        lib_dir.mkdir()

        python_dir = install_root / "python_packages" / "mlir_core"
        python_dir.mkdir(parents=True)

        # Configure with MLIR_INSTALLATIONS
        config = MLIRConfig(installations=str(install_root))

        assert config.toolchain_path == bin_dir
        assert config.mlir_opt.exists()

    def test_config_with_multiple_installations(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test configuration with multiple installations."""
        monkeypatch.chdir(tmp_path)

        # Create first installation (LLVM)
        llvm_root = tmp_path / "llvm"
        llvm_root.mkdir()
        llvm_bin = llvm_root / "bin"
        llvm_bin.mkdir()
        (llvm_bin / "mlir-opt").touch()
        (llvm_bin / "mlir-translate").touch()
        (llvm_root / "lib").mkdir()
        (llvm_root / "python_packages" / "mlir_core").mkdir(parents=True)

        # Create second installation (TT)
        tt_root = tmp_path / "tt"
        tt_root.mkdir()
        tt_bin = tt_root / "bin"
        tt_bin.mkdir()
        (tt_bin / "tt-opt").touch()
        (tt_bin / "tt-translate").touch()
        (tt_root / "lib").mkdir()
        (tt_root / "python_packages" / "mlir_core").mkdir(parents=True)

        # Configure with both installations
        installations_str = f"{llvm_root}:{tt_root}"
        config = MLIRConfig(installations=installations_str)

        # Should use first installation found (LLVM)
        assert config.toolchain_path == llvm_bin
        assert config.mlir_opt.exists()

        # Verify TT tools exist in their directory
        assert (tt_bin / "tt-opt").exists()
        assert (tt_bin / "tt-translate").exists()

    def test_python_paths_added_from_installations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that Python paths are automatically added from installations."""
        import sys

        monkeypatch.chdir(tmp_path)

        # Create installation
        root = tmp_path / "test"
        root.mkdir()

        bin_dir = root / "bin"
        bin_dir.mkdir()
        (bin_dir / "mlir-opt").touch()
        (bin_dir / "mlir-translate").touch()

        python_dir = root / "python_packages" / "mlir_core"
        python_dir.mkdir(parents=True)

        # Configure (side effect: adds to sys.path)
        _ = MLIRConfig(installations=str(root))

        # Verify Python path was added
        assert str(python_dir) in sys.path
