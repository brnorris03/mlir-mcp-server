"""Tests for MLIR parsing tools."""

from pathlib import Path

import pytest

from mlir_mcp_server.config import MLIRConfig, MLIRInstallation
from mlir_mcp_server.tools.parser import get_module_info, parse_mlir, validate_mlir

# Get fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# Define test installation configurations using MLIRInstallation
def get_test_installations() -> list[MLIRInstallation]:
    """Get test MLIR installation configurations.

    Returns:
        List of MLIRInstallation objects for testing different tool prefixes.
    """
    return [
        MLIRInstallation(name="llvm", root=Path("/test/llvm"), tool_prefix="mlir"),
        MLIRInstallation(name="custom", root=Path("/test/custom"), tool_prefix="custom"),
    ]


@pytest.fixture(params=get_test_installations(), ids=lambda x: x.name)
def installation_config(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[MLIRConfig, MLIRInstallation]:
    """Create test configuration with parameterized MLIR installation.

    This allows testing with different tool prefixes (mlir-opt, custom-opt, etc.).
    """
    template_installation: MLIRInstallation = request.param

    # Clear environment to avoid .env interference
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLIR_TOOLCHAIN_PATH", raising=False)

    # Create installation directory structure in tmp
    root_dir = tmp_path / template_installation.name
    root_dir.mkdir()

    # Create actual MLIRInstallation with tmp paths
    installation = MLIRInstallation(
        name=template_installation.name,
        root=root_dir,
        tool_prefix=template_installation.tool_prefix,
    )

    # Create bin directory and tools
    bin_dir = installation.bin_dir
    assert bin_dir is not None
    bin_dir.mkdir(parents=True)

    # Create mock core tools using installation's get_tool_path method
    for tool in ["opt", "translate"]:
        installation.get_tool_path(tool).touch()

    # Create config with this installation
    config = MLIRConfig(toolchain_path=str(bin_dir))

    return config, installation


@pytest.fixture
def config() -> MLIRConfig:
    """Create configuration using real MLIR installation.

    Uses actual mlir-opt and mlir-translate tools from auto-detected or configured installation.
    """
    try:
        # This will use .env or auto-detect real MLIR installation
        config = MLIRConfig()
        # Verify core tools exist
        if not config.mlir_opt.exists() or not config.mlir_translate.exists():
            pytest.skip("Real MLIR tools (mlir-opt, mlir-translate) not found")
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


class TestParseMlir:
    """Tests for parse_mlir function."""

    def test_parse_valid_simple(self, config: MLIRConfig, valid_simple_mlir: str) -> None:
        """Test parsing valid simple MLIR code."""
        result = parse_mlir(config, valid_simple_mlir)

        assert result["success"] is True
        assert "module_str" in result
        assert "operations" in result
        assert result["operation_count"] >= 1

    def test_parse_valid_complex(self, config: MLIRConfig, valid_complex_mlir: str) -> None:
        """Test parsing valid complex MLIR code."""
        result = parse_mlir(config, valid_complex_mlir)

        assert result["success"] is True
        assert result["operation_count"] >= 2  # At least 2 functions

    def test_parse_invalid_syntax(self, config: MLIRConfig, invalid_syntax_mlir: str) -> None:
        """Test parsing invalid MLIR syntax."""
        result = parse_mlir(config, invalid_syntax_mlir)

        assert result["success"] is False
        assert "errors" in result
        assert len(result["errors"]) > 0

    def test_parse_empty_string(self, config: MLIRConfig) -> None:
        """Test parsing empty string.

        Note: MLIR parses empty string as a valid (empty) module.
        """
        result = parse_mlir(config, "")

        # Empty string is valid in MLIR (empty module)
        assert result["success"] is True
        assert result["operation_count"] == 0


class TestValidateMlir:
    """Tests for validate_mlir function."""

    def test_validate_valid_code(self, config: MLIRConfig, valid_simple_mlir: str) -> None:
        """Test validating valid MLIR code."""
        result = validate_mlir(config, valid_simple_mlir)

        assert result["valid"] is True
        assert "message" in result

    def test_validate_invalid_code(self, config: MLIRConfig, invalid_syntax_mlir: str) -> None:
        """Test validating invalid MLIR code."""
        result = validate_mlir(config, invalid_syntax_mlir)

        assert result["valid"] is False
        assert "errors" in result
        assert len(result["errors"]) > 0


class TestGetModuleInfo:
    """Tests for get_module_info function."""

    def test_get_info_simple(self, config: MLIRConfig, valid_simple_mlir: str) -> None:
        """Test extracting info from simple MLIR module."""
        result = get_module_info(config, valid_simple_mlir)

        assert result["success"] is True
        assert "operations" in result
        assert result["operation_count"] >= 1

        # Check that operations have expected fields
        if result["operations"]:
            op = result["operations"][0]
            assert "name" in op
            assert "num_operands" in op
            assert "num_results" in op
            assert "num_regions" in op

    def test_get_info_complex(self, config: MLIRConfig, valid_complex_mlir: str) -> None:
        """Test extracting info from complex MLIR module."""
        result = get_module_info(config, valid_complex_mlir)

        assert result["success"] is True
        assert result["operation_count"] >= 2

        # Verify operation details are extracted
        for op in result["operations"]:
            assert "name" in op
            assert "result_types" in op

    def test_get_info_invalid(self, config: MLIRConfig, invalid_syntax_mlir: str) -> None:
        """Test extracting info from invalid MLIR code."""
        result = get_module_info(config, invalid_syntax_mlir)

        assert result["success"] is False
        assert "errors" in result
