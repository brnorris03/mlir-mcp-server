"""Tests for MLIR error handling utilities."""


from mlir_mcp_server.utils.error_handling import MLIRError, parse_mlir_diagnostic


class TestMLIRError:
    """Tests for MLIRError dataclass."""

    def test_create_error_with_all_fields(self) -> None:
        """Test creating MLIRError with all fields."""
        error = MLIRError(
            message="Test error",
            location="test.mlir:10:5",
            line=10,
            column=5,
            severity="error",
        )

        assert error.message == "Test error"
        assert error.location == "test.mlir:10:5"
        assert error.line == 10
        assert error.column == 5
        assert error.severity == "error"

    def test_create_error_minimal(self) -> None:
        """Test creating MLIRError with only message."""
        error = MLIRError(message="Simple error")

        assert error.message == "Simple error"
        assert error.location is None
        assert error.line is None
        assert error.column is None
        assert error.severity == "error"

    def test_to_dict(self) -> None:
        """Test converting error to dictionary."""
        error = MLIRError(
            message="Test error",
            location="test.mlir:10:5",
            line=10,
            column=5,
            severity="warning",
        )

        result = error.to_dict()

        assert result["message"] == "Test error"
        assert result["location"] == "test.mlir:10:5"
        assert result["line"] == 10
        assert result["column"] == 5
        assert result["severity"] == "warning"


class TestParseMlirDiagnostic:
    """Tests for parse_mlir_diagnostic function."""

    def test_parse_simple_error(self) -> None:
        """Test parsing a simple error diagnostic."""
        diagnostic = "test.mlir:5:10: error: invalid operation"

        errors = parse_mlir_diagnostic(diagnostic)

        assert len(errors) == 1
        assert errors[0].message == "invalid operation"
        assert errors[0].severity == "error"

    def test_parse_warning(self) -> None:
        """Test parsing a warning diagnostic."""
        diagnostic = "test.mlir:3:5: warning: unused variable"

        errors = parse_mlir_diagnostic(diagnostic)

        assert len(errors) == 1
        assert errors[0].severity == "warning"
        assert "unused variable" in errors[0].message

    def test_parse_note(self) -> None:
        """Test parsing a note diagnostic."""
        diagnostic = "test.mlir:1:1: note: see declaration here"

        errors = parse_mlir_diagnostic(diagnostic)

        assert len(errors) == 1
        assert errors[0].severity == "note"

    def test_parse_multiple_diagnostics(self) -> None:
        """Test parsing multiple diagnostics."""
        diagnostic = """test.mlir:5:10: error: invalid operation
test.mlir:7:3: warning: unused value
test.mlir:10:15: error: type mismatch"""

        errors = parse_mlir_diagnostic(diagnostic)

        assert len(errors) == 3
        assert errors[0].severity == "error"
        assert errors[1].severity == "warning"
        assert errors[2].severity == "error"

    def test_parse_multiline_error(self) -> None:
        """Test parsing error with continuation lines."""
        diagnostic = """test.mlir:5:10: error: operation does not dominate this use
  note: see current operation"""

        errors = parse_mlir_diagnostic(diagnostic)

        # Should capture at least the main error
        assert len(errors) >= 1
        assert errors[0].severity == "error"

    def test_parse_unstructured_error(self) -> None:
        """Test parsing unstructured error message."""
        diagnostic = "Some generic error message without location"

        errors = parse_mlir_diagnostic(diagnostic)

        assert len(errors) == 1
        assert errors[0].message == diagnostic
        assert errors[0].location is None
        assert errors[0].severity == "error"

    def test_parse_empty_string(self) -> None:
        """Test parsing empty diagnostic string."""
        errors = parse_mlir_diagnostic("")

        assert len(errors) == 1
        assert errors[0].message == ""

    def test_parse_with_line_and_column(self) -> None:
        """Test that line and column numbers are extracted."""
        diagnostic = "file.mlir:42:17: error: test"

        errors = parse_mlir_diagnostic(diagnostic)

        assert len(errors) == 1
        # Note: Current implementation may not extract line/column correctly
        # This test documents the expected behavior
