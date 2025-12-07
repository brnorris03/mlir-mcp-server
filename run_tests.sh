#!/bin/bash
# Helper script to run tests with correct environment
#
# CUSTOMIZE THESE PATHS to match your MLIR installation:

# Path to MLIR Python bindings (adjust to your installation)
export PYTHONPATH=/opt/ttmlir-toolchain/python_packages/mlir_core:$PYTHONPATH

# Path to MLIR shared libraries (adjust to your installation)
# macOS:
export DYLD_LIBRARY_PATH=/opt/ttmlir-toolchain/lib:$DYLD_LIBRARY_PATH
# Linux (comment out DYLD and uncomment this):
# export LD_LIBRARY_PATH=/opt/ttmlir-toolchain/lib:$LD_LIBRARY_PATH

# Activate virtual environment (adjust if using different name)
source venv/bin/activate

# Run tests
pytest "$@"
