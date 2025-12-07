#!/bin/bash
# Helper script to run tests with correct environment
#
# CUSTOMIZE THESE PATHS to match your MLIR installation:

# Path to MLIR Python bindings
# Example: /usr/local/llvm/python_packages/mlir_core
# Example: ~/llvm-project/build/tools/mlir/python_packages/mlir_core
export PYTHONPATH=/path/to/mlir/python_packages/mlir_core:$PYTHONPATH

# Path to MLIR shared libraries
# macOS:
export DYLD_LIBRARY_PATH=/path/to/mlir/lib:$DYLD_LIBRARY_PATH
# Linux (comment out DYLD and uncomment this):
# export LD_LIBRARY_PATH=/path/to/mlir/lib:$LD_LIBRARY_PATH

# Activate virtual environment (match Python version to MLIR bindings)
source venv/bin/activate

# Run tests
pytest "$@"
