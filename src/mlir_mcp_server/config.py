"""Configuration management for MLIR MCP Server.

This module provides configuration loading with auto-detection of MLIR toolchain,
support for environment variables, and .env files using Pydantic Settings.
Supports multiple MLIR installations with different Python bindings and tool prefixes.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


@dataclass
class MLIRInstallation:
    """Represents a single MLIR installation (e.g., LLVM, custom project).

    Attributes:
        name: Installation name (e.g., "llvm", "tt", "custom").
        root: Root directory of the installation.
        tool_prefix: Prefix for tools (e.g., "mlir" for mlir-opt, "tt" for tt-opt).
        bin_dir: Directory containing tools (default: <root>/bin).
        python_dir: Directory containing Python bindings (default: <root>/python_packages/mlir_core).
    """

    name: str
    root: Path
    tool_prefix: str = "mlir"
    bin_dir: Path | None = None
    python_dir: Path | None = None

    def __post_init__(self) -> None:
        """Initialize derived paths."""
        if self.bin_dir is None:
            self.bin_dir = self.root / "bin"
        if self.python_dir is None:
            self.python_dir = self.root / "python_packages" / "mlir_core"

    def get_tool_path(self, tool_name: str) -> Path:
        """Get path to a tool with this installation's prefix.

        Args:
            tool_name: Base tool name (e.g., "opt", "translate").

        Returns:
            Path to the tool (e.g., /path/to/bin/mlir-opt or /path/to/bin/tt-opt).
        """
        assert self.bin_dir is not None
        return self.bin_dir / f"{self.tool_prefix}-{tool_name}"

    def validate(self) -> dict[str, bool]:
        """Validate that core tools exist.

        Core tools: opt, translate
        Optional tools: reduce, query, runner

        Returns:
            Dictionary of tool availability.
        """
        core_tools = ["opt", "translate"]
        optional_tools = ["reduce", "query", "runner"]

        result = {}
        for tool in core_tools + optional_tools:
            result[tool] = self.get_tool_path(tool).exists()

        return result


def auto_detect_mlir_toolchain(additional_paths: list[Path] | None = None) -> Path | None:
    """Auto-detect MLIR toolchain from common installation locations.

    Searches the following locations in order:
    1. Additional paths (if provided)
    2. /opt/ttmlir-toolchain/bin
    3. /usr/local/llvm/bin
    4. /usr/local/bin
    5. /opt/llvm/bin
    6. ~/llvm-project/build/bin

    Args:
        additional_paths: Optional list of additional paths to search first.

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

    # Prepend additional paths if provided
    search_locations = list(additional_paths) if additional_paths else []
    search_locations.extend(common_locations)

    for location in search_locations:
        if location.exists() and (location / "mlir-opt").exists():
            logger.info(f"Auto-detected MLIR toolchain at {location}")
            return location

    return None


def setup_python_paths(python_paths: list[Path], lib_paths: list[Path]) -> None:
    """Add MLIR Python binding paths to sys.path and library paths to LD_LIBRARY_PATH.

    Args:
        python_paths: List of paths to MLIR Python packages.
        lib_paths: List of paths to MLIR shared libraries.
    """
    import platform
    import sys

    # Add Python bindings to sys.path and PYTHONPATH
    for path in python_paths:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
                logger.info(f"Added to Python path: {path_str}")

            # Also set PYTHONPATH for subprocess compatibility
            current_pythonpath = os.environ.get("PYTHONPATH", "")
            if path_str not in current_pythonpath:
                if current_pythonpath:
                    os.environ["PYTHONPATH"] = f"{path_str}:{current_pythonpath}"
                else:
                    os.environ["PYTHONPATH"] = path_str
        else:
            logger.warning(f"Python bindings path does not exist: {path}")

    # Add library paths for shared libraries
    if platform.system() == "Darwin":
        lib_var = "DYLD_LIBRARY_PATH"
    elif platform.system() == "Windows":
        lib_var = "PATH"
    else:  # Linux and others
        lib_var = "LD_LIBRARY_PATH"

    for lib_path in lib_paths:
        if lib_path.exists():
            lib_path_str = str(lib_path)
            current_lib_path = os.environ.get(lib_var, "")

            if lib_path_str not in current_lib_path:
                separator = ";" if platform.system() == "Windows" else ":"
                if current_lib_path:
                    os.environ[lib_var] = f"{lib_path_str}{separator}{current_lib_path}"
                else:
                    os.environ[lib_var] = lib_path_str
                logger.info(f"Added to {lib_var}: {lib_path_str}")
        else:
            logger.warning(f"Library path does not exist: {lib_path}")


class MLIRConfig(BaseSettings):
    """MLIR MCP Server configuration.

    Configuration is loaded with the following priority (highest to lowest):
    1. .env file in the current directory
    2. Environment variables with MLIR_ prefix
    3. Auto-detected toolchain path from common locations

    Attributes:
        toolchain_path: Path to the MLIR toolchain bin directory.
        installations: Colon-separated root paths for MLIR installations.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MLIR_",
        case_sensitive=False,
        extra="ignore",  # Allow extra fields for flexibility
    )

    toolchain_path: Path | None = None
    installations: str | None = None

    @field_validator("toolchain_path", mode="before")
    @classmethod
    def validate_toolchain_path(cls, v: str | Path | None) -> Path | None:
        """Validate and convert toolchain path to Path object."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        return v.expanduser().resolve()

    def __init__(self, **kwargs):  # type: ignore
        """Initialize configuration with auto-detection fallback."""
        super().__init__(**kwargs)

        # Process multiple MLIR installations if provided
        additional_search_paths: list[Path] | None = None
        python_binding_paths: list[Path] = []
        lib_paths: list[Path] = []

        if self.installations:
            installation_roots = [
                Path(p.strip()).expanduser().resolve()
                for p in self.installations.split(":")
                if p.strip()
            ]

            logger.info(f"Configuring {len(installation_roots)} MLIR installation(s)")

            additional_search_paths = []
            for root in installation_roots:
                # Add bin directory to toolchain search paths
                bin_dir = root / "bin"
                if bin_dir.exists():
                    additional_search_paths.append(bin_dir)
                    logger.info(f"  Toolchain: {bin_dir}")

                # Add Python bindings to path
                python_dir = root / "python_packages" / "mlir_core"
                if python_dir.exists():
                    python_binding_paths.append(python_dir)
                    logger.info(f"  Python bindings: {python_dir}")
                else:
                    logger.warning(f"  Python bindings not found at {python_dir}")

                # Add lib directory for shared libraries
                lib_dir = root / "lib"
                if lib_dir.exists():
                    lib_paths.append(lib_dir)
                    logger.info(f"  Libraries: {lib_dir}")

        # Auto-detect if not configured
        if self.toolchain_path is None:
            detected = auto_detect_mlir_toolchain(additional_search_paths)
            if detected:
                self.toolchain_path = detected
            else:
                raise RuntimeError(
                    "MLIR toolchain not found. Please set MLIR_TOOLCHAIN_PATH environment variable "
                    "or MLIR_INSTALLATIONS. "
                    "Searched locations: /opt/ttmlir-toolchain/bin, /usr/local/llvm/bin, "
                    "/usr/local/bin, /opt/llvm/bin, ~/llvm-project/build/bin"
                )

        # Set up Python paths and library paths for MLIR bindings
        if python_binding_paths or lib_paths:
            setup_python_paths(python_binding_paths, lib_paths)

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
