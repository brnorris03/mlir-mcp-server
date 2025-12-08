#!/bin/bash
# Helper script to run tests
# IMPORTANT: This script requires Bash. Run with: bash run_tests.sh
#
# NOTE: If you set MLIR_INSTALLATIONS in your .env file, the server will
# automatically configure PYTHONPATH and library paths. No manual setup needed!
#
# Example .env:
#   MLIR_INSTALLATIONS=/opt/llvm-build:/path/to/custom-mlir
#
# Or set MLIR_TOOLCHAIN_PATH for simple single installation.

# Activate virtual environment (match Python version to MLIR bindings)
source venv/bin/activate

# Run tests
# PYTHONPATH, DYLD_LIBRARY_PATH/LD_LIBRARY_PATH are configured automatically by MLIRConfig
pytest "$@"
