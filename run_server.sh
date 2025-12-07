#!/bin/bash
# Helper script to run MLIR MCP Server
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

# Run the server
# PYTHONPATH, DYLD_LIBRARY_PATH/LD_LIBRARY_PATH are configured automatically by MLIRConfig
python -m mlir_mcp_server "$@"
