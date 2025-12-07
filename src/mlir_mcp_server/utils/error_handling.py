"""Error handling utilities for MLIR operations.

This module provides utilities for capturing and formatting MLIR diagnostics
and errors in a structured way suitable for MCP responses.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MLIRError:
    """Structured MLIR error information.

    Attributes:
        message: Error message text.
        location: Source location string (e.g., "file.mlir:10:5").
        line: Line number if available.
        column: Column number if available.
        severity: Error severity (error, warning, note).
    """

    message: str
    location: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    severity: str = "error"

    def to_dict(self) -> dict[str, str | int | None]:
        """Convert error to dictionary format for JSON serialization."""
        return {
            "message": self.message,
            "location": self.location,
            "line": self.line,
            "column": self.column,
            "severity": self.severity,
        }


def parse_mlir_diagnostic(diagnostic_str: str) -> list[MLIRError]:
    """Parse MLIR diagnostic output into structured errors.

    Args:
        diagnostic_str: Raw diagnostic string from MLIR.

    Returns:
        List of structured MLIRError objects.
    """
    errors = []

    # Split by lines and parse each diagnostic
    lines = diagnostic_str.strip().split("\n")
    current_error: Optional[dict[str, str | int | None]] = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to parse location info (format: file.mlir:line:col: severity: message)
        if ":" in line and any(sev in line.lower() for sev in ["error", "warning", "note"]):
            parts = line.split(":", 4)
            if len(parts) >= 4:
                # Save previous error if exists
                if current_error:
                    errors.append(
                        MLIRError(
                            message=str(current_error.get("message", "")),
                            location=str(current_error.get("location")) if current_error.get("location") else None,
                            line=int(current_error["line"]) if current_error.get("line") else None,
                            column=int(current_error["column"]) if current_error.get("column") else None,
                            severity=str(current_error.get("severity", "error")),
                        )
                    )

                # Parse new error
                current_error = {}
                try:
                    current_error["line"] = int(parts[1]) if parts[1].isdigit() else None
                    current_error["column"] = int(parts[2]) if parts[2].isdigit() else None
                    current_error["location"] = f"{parts[0]}:{parts[1]}:{parts[2]}"
                except (ValueError, IndexError):
                    current_error["line"] = None
                    current_error["column"] = None
                    current_error["location"] = None

                # Extract severity and message
                severity_msg = ":".join(parts[3:]) if len(parts) > 3 else ""
                for sev in ["error", "warning", "note"]:
                    if sev in severity_msg.lower():
                        current_error["severity"] = sev
                        current_error["message"] = severity_msg.split(sev, 1)[1].strip(": ")
                        break
        elif current_error:
            # Continuation of previous error message
            current_error["message"] = str(current_error.get("message", "")) + " " + line

    # Add last error
    if current_error:
        errors.append(
            MLIRError(
                message=str(current_error.get("message", "")),
                location=str(current_error.get("location")) if current_error.get("location") else None,
                line=int(current_error["line"]) if current_error.get("line") else None,
                column=int(current_error["column"]) if current_error.get("column") else None,
                severity=str(current_error.get("severity", "error")),
            )
        )

    # If no structured errors found, create a single error from the full message
    if not errors:
        errors.append(MLIRError(message=diagnostic_str))

    return errors
