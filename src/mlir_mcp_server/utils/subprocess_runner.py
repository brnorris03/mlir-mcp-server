"""Subprocess runner utilities for executing MLIR tools.

This module provides safe subprocess execution for MLIR command-line tools
like mlir-opt, mlir-translate, mlir-reduce, etc.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any

from ..config import MLIRConfig

logger = logging.getLogger(__name__)


class MLIRToolRunner:
    """Runner for MLIR command-line tools with timeout and error handling."""

    def __init__(self, config: MLIRConfig):
        """Initialize tool runner with configuration.

        Args:
            config: MLIR configuration with tool paths.
        """
        self.config = config
        self.timeout = 30  # Default timeout in seconds

    def run_mlir_opt(self, mlir_code: str, passes: str) -> str:
        """Run mlir-opt with specified pass pipeline.

        Args:
            mlir_code: MLIR code as string.
            passes: Pass pipeline specification (e.g., "canonicalize,cse").

        Returns:
            Transformed MLIR code.

        Raises:
            FileNotFoundError: If mlir-opt is not found.
            RuntimeError: If mlir-opt execution fails or times out.
        """
        mlir_opt = self.config.mlir_opt

        if not mlir_opt.exists():
            raise FileNotFoundError(f"mlir-opt not found at {mlir_opt}")

        try:
            # Wrap passes with anchor operation (builtin.module for module-level passes)
            # Format: --pass-pipeline='builtin.module(pass1,pass2)'
            wrapped_pipeline = f"builtin.module({passes})"

            result = subprocess.run(
                [str(mlir_opt), f"--pass-pipeline={wrapped_pipeline}"],
                input=mlir_code.encode(),
                capture_output=True,
                timeout=self.timeout,
                text=False,
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode()
                raise RuntimeError(f"mlir-opt failed: {error_msg}")

            return result.stdout.decode()

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"mlir-opt timed out after {self.timeout}s") from None

    def run_mlir_translate(self, mlir_code: str, options: list[str] | None = None) -> str:
        """Run mlir-translate with specified options.

        Args:
            mlir_code: MLIR code as string.
            options: List of command-line options (e.g., ["--mlir-to-llvmir"]).

        Returns:
            Translated output.

        Raises:
            FileNotFoundError: If mlir-translate is not found.
            RuntimeError: If execution fails or times out.
        """
        mlir_translate = self.config.mlir_translate

        if not mlir_translate.exists():
            raise FileNotFoundError(f"mlir-translate not found at {mlir_translate}")

        options = options or []

        try:
            result = subprocess.run(
                [str(mlir_translate)] + options,
                input=mlir_code.encode(),
                capture_output=True,
                timeout=self.timeout,
                text=False,
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode()
                raise RuntimeError(f"mlir-translate failed: {error_msg}")

            return result.stdout.decode()

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"mlir-translate timed out after {self.timeout}s"
            ) from None

    def run_tool(self, tool_path: Path, args: list[str], input_data: str | None = None) -> str:
        """Run arbitrary MLIR tool with arguments.

        Args:
            tool_path: Path to the tool executable.
            args: List of command-line arguments.
            input_data: Optional input data to pass via stdin.

        Returns:
            Tool output.

        Raises:
            FileNotFoundError: If tool is not found.
            RuntimeError: If execution fails or times out.
        """
        if not tool_path.exists():
            raise FileNotFoundError(f"Tool not found at {tool_path}")

        try:
            input_bytes = input_data.encode() if input_data else None

            result = subprocess.run(
                [str(tool_path)] + args,
                input=input_bytes,
                capture_output=True,
                timeout=self.timeout,
                text=False,
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode()
                raise RuntimeError(f"{tool_path.name} failed: {error_msg}")

            return result.stdout.decode()

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"{tool_path.name} timed out after {self.timeout}s"
            ) from None
