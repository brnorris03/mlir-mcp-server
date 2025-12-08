"""Tests for MLIR translation tools."""

import pytest

from mlir_mcp_server.config import MLIRConfig
from mlir_mcp_server.tools.translator import (
    get_translation_info,
    translate_from_llvmir,
    translate_to_llvmir,
)

from .utils import get_real_mlir_config


@pytest.fixture
def config() -> MLIRConfig:
    """Create configuration using real MLIR installation."""
    return get_real_mlir_config()


class TestTranslateToLLVMIR:
    """Tests for translate_to_llvmir function."""

    def test_translate_simple_llvm_dialect(self, config: MLIRConfig) -> None:
        """Test translating simple LLVM dialect MLIR to LLVM IR."""
        mlir_code = """
module {
  llvm.func @test() -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    llvm.return %c1 : i32
  }
}
"""
        result = translate_to_llvmir(config, mlir_code)

        assert result["success"] is True
        assert "llvm_ir" in result
        llvm_ir = result["llvm_ir"]
        # LLVM IR should contain function definition
        assert any(keyword in llvm_ir for keyword in ["define", "declare"])
        assert "i32" in llvm_ir

    def test_translate_with_arithmetic(self, config: MLIRConfig) -> None:
        """Test translating LLVM dialect with arithmetic operations."""
        mlir_code = """
module {
  llvm.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %result = llvm.add %arg0, %arg1 : i32
    llvm.return %result : i32
  }
}
"""
        result = translate_to_llvmir(config, mlir_code)

        assert result["success"] is True
        assert "llvm_ir" in result
        llvm_ir = result["llvm_ir"]
        assert "add" in llvm_ir.lower()

    def test_translate_invalid_mlir(self, config: MLIRConfig) -> None:
        """Test translation with invalid MLIR."""
        mlir_code = "this is not valid MLIR"

        result = translate_to_llvmir(config, mlir_code)

        assert result["success"] is False
        assert "errors" in result
        assert len(result["errors"]) > 0

    def test_translate_non_llvm_dialect(self, config: MLIRConfig) -> None:
        """Test translation with non-LLVM dialect (should fail)."""
        mlir_code = """
module {
  func.func @test() -> i32 {
    %c1 = arith.constant 1 : i32
    func.return %c1 : i32
  }
}
"""
        result = translate_to_llvmir(config, mlir_code)

        # Should fail because arith dialect is not LLVM dialect
        assert result["success"] is False
        assert "errors" in result

    def test_translate_empty_module(self, config: MLIRConfig) -> None:
        """Test translating empty module."""
        mlir_code = "module {}"

        result = translate_to_llvmir(config, mlir_code)

        # Empty module should translate successfully
        assert result["success"] is True
        assert "llvm_ir" in result


class TestTranslateFromLLVMIR:
    """Tests for translate_from_llvmir function."""

    def test_translate_simple_llvm_ir(self, config: MLIRConfig) -> None:
        """Test importing simple LLVM IR."""
        llvm_ir = """
define i32 @test() {
  ret i32 1
}
"""
        result = translate_from_llvmir(config, llvm_ir)

        assert result["success"] is True
        assert "mlir_code" in result
        mlir_code = result["mlir_code"]
        # Should contain LLVM dialect operations
        assert "llvm" in mlir_code.lower()

    def test_translate_with_arguments(self, config: MLIRConfig) -> None:
        """Test importing LLVM IR with function arguments."""
        llvm_ir = """
define i32 @add(i32 %a, i32 %b) {
  %result = add i32 %a, %b
  ret i32 %result
}
"""
        result = translate_from_llvmir(config, llvm_ir)

        assert result["success"] is True
        assert "mlir_code" in result
        mlir_code = result["mlir_code"]
        assert "llvm" in mlir_code.lower()

    def test_translate_invalid_llvm_ir(self, config: MLIRConfig) -> None:
        """Test importing invalid LLVM IR."""
        llvm_ir = "this is not valid LLVM IR"

        result = translate_from_llvmir(config, llvm_ir)

        assert result["success"] is False
        assert "errors" in result

    def test_translate_empty_llvm_ir(self, config: MLIRConfig) -> None:
        """Test importing empty LLVM IR."""
        llvm_ir = ""

        result = translate_from_llvmir(config, llvm_ir)

        # Empty IR might succeed or fail depending on mlir-translate behavior
        # Just check that we get a proper response
        assert "success" in result


class TestGetTranslationInfo:
    """Tests for get_translation_info function."""

    def test_get_translation_info(self, config: MLIRConfig) -> None:
        """Test getting translation information."""
        result = get_translation_info(config)

        assert "mlir_translate_available" in result
        assert "mlir_translate_path" in result
        assert "supported_translations" in result

        # Should report mlir-translate as available (we're using real config)
        assert result["mlir_translate_available"] is True

        # Should have supported translations listed
        supported = result["supported_translations"]
        assert len(supported) == 2

        # Check that both directions are listed
        names = [t["name"] for t in supported]
        assert "mlir-to-llvmir" in names
        assert "import-llvm" in names

    def test_translation_info_structure(self, config: MLIRConfig) -> None:
        """Test that translation info has correct structure."""
        result = get_translation_info(config)

        for translation in result["supported_translations"]:
            assert "name" in translation
            assert "description" in translation
            assert "direction" in translation


class TestRoundTrip:
    """Tests for round-trip translation (MLIR -> LLVM IR -> MLIR)."""

    def test_simple_round_trip(self, config: MLIRConfig) -> None:
        """Test round-trip translation of simple function."""
        # Start with LLVM dialect MLIR
        original_mlir = """
module {
  llvm.func @test() -> i32 {
    %c1 = llvm.mlir.constant(1 : i32) : i32
    llvm.return %c1 : i32
  }
}
"""

        # Translate to LLVM IR
        result1 = translate_to_llvmir(config, original_mlir)
        assert result1["success"] is True
        llvm_ir = result1["llvm_ir"]

        # Translate back to MLIR
        result2 = translate_from_llvmir(config, llvm_ir)
        assert result2["success"] is True
        mlir_code = result2["mlir_code"]

        # Should contain LLVM dialect
        assert "llvm" in mlir_code.lower()
        # Should have a function
        assert "func" in mlir_code.lower()


class TestTranslationErrors:
    """Tests for error handling in translation."""

    def test_translate_to_llvmir_with_syntax_error(self, config: MLIRConfig) -> None:
        """Test translation with syntax errors."""
        mlir_code = """
module {
  llvm.func @broken(
}
"""
        result = translate_to_llvmir(config, mlir_code)

        assert result["success"] is False
        assert "errors" in result
        error_msg = result["errors"][0]["message"]
        assert "Translation failed" in error_msg

    def test_translate_from_llvmir_with_syntax_error(
        self, config: MLIRConfig
    ) -> None:
        """Test importing LLVM IR with syntax errors."""
        llvm_ir = "define i32 @broken( {"

        result = translate_from_llvmir(config, llvm_ir)

        assert result["success"] is False
        assert "errors" in result
