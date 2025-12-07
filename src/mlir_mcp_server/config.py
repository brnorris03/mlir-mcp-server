"""Configuration management for MLIR MCP Server.

This module provides configuration loading with auto-detection of MLIR toolchain,
support for environment variables, and .env files using Pydantic Settings.
"""

import logging
from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def auto_detect_mlir_toolchain() -> Optional[Path]:
    """Auto-detect MLIR toolchain from common installation locations.

    Searches the following locations in order:
    1. /opt/ttmlir-toolchain/bin
    2. /usr/local/llvm/bin
    3. /usr/local/bin
    4. /opt/llvm/bin
    5. ~/llvm-project/build/bin

    Returns:
        Path to the MLIR toolchain bin directory if found, None otherwise.
    """
    common_locations = [
        Path("/opt/ttmlir-toolchain/bin"),
        Path("/usr/local/llvm/bin"),
        Path("/usr/local/bin"),
        Path("/opt/llvm/bin"),
        Path.home() / "llvm-project" / "build" / "bin",
    ]

    for location in common_locations:
        if location.exists() and (location / "mlir-opt").exists():
            logger.info(f"Auto-detected MLIR toolchain at {location}")
            return location

    return None


class MLIRConfig(BaseSettings):
    """MLIR MCP Server configuration.

    Configuration is loaded with the following priority (highest to lowest):
    1. .env file in the current directory
    2. Environment variables with MLIR_ prefix
    3. Auto-detected toolchain path from common locations

    Attributes:
        toolchain_path: Path to the MLIR toolchain bin directory.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MLIR_",
        case_sensitive=False,
    )

    toolchain_path: Optional[Path] = None

    @field_validator("toolchain_path", mode="before")
    @classmethod
    def validate_toolchain_path(cls, v: Optional[str | Path]) -> Optional[Path]:
        """Validate and convert toolchain path to Path object."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        return v.expanduser().resolve()

    def __init__(self, **kwargs):  # type: ignore
        """Initialize configuration with auto-detection fallback."""
        super().__init__(**kwargs)

        # Auto-detect if not configured
        if self.toolchain_path is None:
            detected = auto_detect_mlir_toolchain()
            if detected:
                self.toolchain_path = detected
            else:
                raise RuntimeError(
                    "MLIR toolchain not found. Please set MLIR_TOOLCHAIN_PATH environment variable. "
                    "Searched locations: /opt/ttmlir-toolchain/bin, /usr/local/llvm/bin, "
                    "/usr/local/bin, /opt/llvm/bin, ~/llvm-project/build/bin"
                )

    @property
    def mlir_opt(self) -> Path:
        """Path to mlir-opt tool."""
        assert self.toolchain_path is not None
        return self.toolchain_path / "mlir-opt"

    @property
    def mlir_translate(self) -> Path:
        """Path to mlir-translate tool."""
        assert self.toolchain_path is not None
        return self.toolchain_path / "mlir-translate"

    @property
    def mlir_reduce(self) -> Path:
        """Path to mlir-reduce tool (optional)."""
        assert self.toolchain_path is not None
        return self.toolchain_path / "mlir-reduce"

    @property
    def mlir_query(self) -> Path:
        """Path to mlir-query tool (optional)."""
        assert self.toolchain_path is not None
        return self.toolchain_path / "mlir-query"

    @property
    def mlir_runner(self) -> Path:
        """Path to mlir-runner tool (optional)."""
        assert self.toolchain_path is not None
        return self.toolchain_path / "mlir-runner"

    def validate_tools(self) -> dict[str, bool]:
        """Validate tool availability.

        Validates that core tools (mlir-opt, mlir-translate) exist and logs
        warnings for missing optional tools.

        Returns:
            Dictionary mapping tool names to availability status.

        Raises:
            RuntimeError: If core tools are missing.
        """
        results = {
            "mlir_opt": self.mlir_opt.exists(),
            "mlir_translate": self.mlir_translate.exists(),
            "mlir_reduce": self.mlir_reduce.exists(),
            "mlir_query": self.mlir_query.exists(),
            "mlir_runner": self.mlir_runner.exists(),
        }

        # Core tools must exist
        if not results["mlir_opt"] or not results["mlir_translate"]:
            raise RuntimeError(
                f"Core MLIR tools not found at {self.toolchain_path}. "
                f"Ensure mlir-opt and mlir-translate are available."
            )

        # Log warnings for missing optional tools
        if not results["mlir_reduce"]:
            logger.warning(
                f"mlir-reduce not found at {self.mlir_reduce} - reduce_testcase tool will be disabled"
            )
        if not results["mlir_query"]:
            logger.warning(
                f"mlir-query not found at {self.mlir_query} - query_operations tool will be disabled"
            )
        if not results["mlir_runner"]:
            logger.warning(
                f"mlir-runner not found at {self.mlir_runner} - execution tools will be disabled"
            )

        return results
