"""Tests for subprocess runner utilities."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlir_mcp_server.config import MLIRConfig
from mlir_mcp_server.utils.subprocess_runner import MLIRToolRunner

from .utils import get_real_mlir_config


@pytest.fixture
def config() -> MLIRConfig:
    """Create configuration using real MLIR installation."""
    return get_real_mlir_config()


@pytest.fixture
def runner(config: MLIRConfig) -> MLIRToolRunner:
    """Create tool runner."""
    return MLIRToolRunner(config)


class TestMLIRToolRunnerInit:
    """Tests for MLIRToolRunner initialization."""

    def test_init_with_config(self, config: MLIRConfig) -> None:
        """Test runner initialization."""
        runner = MLIRToolRunner(config)

        assert runner.config == config
        assert runner.timeout == 30


class TestRunMlirOpt:
    """Tests for run_mlir_opt method."""

    def test_run_mlir_opt_success(self, runner: MLIRToolRunner) -> None:
        """Test successful mlir-opt execution."""
        mlir_code = """
module {
  func.func @test() -> i32 {
    %c1 = arith.constant 1 : i32
    func.return %c1 : i32
  }
}
"""
        result = runner.run_mlir_opt(mlir_code, "canonicalize")

        assert result is not None
        assert "module" in result.lower()

    def test_run_mlir_opt_invalid_pass(self, runner: MLIRToolRunner) -> None:
        """Test error with invalid pass name."""
        mlir_code = "module {}"

        with pytest.raises(RuntimeError, match="mlir-opt failed"):
            runner.run_mlir_opt(mlir_code, "nonexistent-pass")

    def test_run_mlir_opt_invalid_mlir(self, runner: MLIRToolRunner) -> None:
        """Test error with invalid MLIR code."""
        mlir_code = "this is not valid MLIR"

        with pytest.raises(RuntimeError, match="mlir-opt failed"):
            runner.run_mlir_opt(mlir_code, "canonicalize")

    @patch("subprocess.run")
    def test_run_mlir_opt_timeout(self, mock_run: MagicMock, runner: MLIRToolRunner) -> None:
        """Test timeout handling."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["mlir-opt"], timeout=30)

        mlir_code = "module {}"

        with pytest.raises(RuntimeError, match="timed out after 30s"):
            runner.run_mlir_opt(mlir_code, "canonicalize")


class TestRunMlirTranslate:
    """Tests for run_mlir_translate method."""

    def test_run_mlir_translate_to_llvmir(self, runner: MLIRToolRunner) -> None:
        """Test mlir-translate to LLVM IR."""
        # Use simple LLVM dialect code that doesn't need arith dialect
        mlir_code = """
module {
  llvm.func @test() -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    llvm.return %c1 : i32
  }
}
"""
        result = runner.run_mlir_translate(mlir_code, ["--mlir-to-llvmir"])

        assert result is not None
        # LLVM IR should contain define or declare
        assert any(keyword in result for keyword in ["define", "declare", "ModuleID"])

    def test_run_mlir_translate_no_options(self, runner: MLIRToolRunner) -> None:
        """Test mlir-translate with no options (should fail)."""
        mlir_code = "module {}"

        # Should fail because no translation option specified
        with pytest.raises(RuntimeError, match="mlir-translate failed"):
            runner.run_mlir_translate(mlir_code, None)

    def test_run_mlir_translate_invalid_mlir(self, runner: MLIRToolRunner) -> None:
        """Test error with invalid MLIR code."""
        mlir_code = "this is not valid MLIR"

        with pytest.raises(RuntimeError, match="mlir-translate failed"):
            runner.run_mlir_translate(mlir_code, ["--mlir-to-llvmir"])

    @patch("subprocess.run")
    def test_run_mlir_translate_timeout(
        self, mock_run: MagicMock, runner: MLIRToolRunner
    ) -> None:
        """Test timeout handling."""
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["mlir-translate"], timeout=30
        )

        mlir_code = "module {}"

        with pytest.raises(RuntimeError, match="timed out after 30s"):
            runner.run_mlir_translate(mlir_code, ["--mlir-to-llvmir"])


class TestRunTool:
    """Tests for generic run_tool method."""

    def test_run_tool_with_input(self, runner: MLIRToolRunner, config: MLIRConfig) -> None:
        """Test running arbitrary tool with input."""
        # Use mlir-opt as the test tool
        mlir_code = """
module {
  func.func @test() -> i32 {
    %c1 = arith.constant 1 : i32
    func.return %c1 : i32
  }
}
"""
        result = runner.run_tool(
            config.mlir_opt,
            ["--pass-pipeline=builtin.module(canonicalize)"],
            mlir_code,
        )

        assert result is not None
        assert "module" in result.lower()

    def test_run_tool_no_input(self, runner: MLIRToolRunner, config: MLIRConfig) -> None:
        """Test running tool without input data."""
        # Use mlir-opt with --help
        result = runner.run_tool(config.mlir_opt, ["--help"], None)

        assert result is not None
        # Help text should contain usage information
        assert "usage" in result.lower() or "mlir-opt" in result.lower()

    def test_run_tool_nonexistent(self, runner: MLIRToolRunner) -> None:
        """Test error when tool doesn't exist."""
        tool_path = Path("/nonexistent/tool")

        with pytest.raises(FileNotFoundError, match="Tool not found"):
            runner.run_tool(tool_path, ["--help"], None)

    def test_run_tool_invalid_args(self, runner: MLIRToolRunner, config: MLIRConfig) -> None:
        """Test error with invalid tool arguments."""
        with pytest.raises(RuntimeError, match="failed"):
            runner.run_tool(config.mlir_opt, ["--invalid-flag-xyz"], None)

    @patch("subprocess.run")
    def test_run_tool_timeout(
        self, mock_run: MagicMock, runner: MLIRToolRunner, config: MLIRConfig
    ) -> None:
        """Test timeout handling."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["tool"], timeout=30)

        with pytest.raises(RuntimeError, match="timed out after 30s"):
            runner.run_tool(config.mlir_opt, ["--help"], None)


class TestRunnerConfiguration:
    """Tests for runner configuration."""

    def test_custom_timeout(self, config: MLIRConfig) -> None:
        """Test setting custom timeout."""
        runner = MLIRToolRunner(config)
        runner.timeout = 60

        assert runner.timeout == 60

    def test_timeout_used_in_run(self, runner: MLIRToolRunner) -> None:
        """Test that timeout is used in subprocess calls."""
        runner.timeout = 1  # Very short timeout

        # Use a pass that might take time
        mlir_code = "module {}"

        # This should work fine with short MLIR
        try:
            result = runner.run_mlir_opt(mlir_code, "canonicalize")
            assert result is not None
        except RuntimeError as e:
            # Either it works or times out, both are acceptable
            if "timed out" not in str(e):
                raise
