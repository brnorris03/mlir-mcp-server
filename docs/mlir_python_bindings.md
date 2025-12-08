# MLIR Python Bindings Setup

The MLIR MCP Server requires MLIR Python bindings to parse and manipulate MLIR code. These bindings are part of the LLVM/MLIR project and need to be built or installed separately.

## Option 1: Use Pre-built MLIR Python Bindings

If you have a pre-built MLIR installation with Python bindings:

```bash
# Set PYTHONPATH to include MLIR Python bindings
export PYTHONPATH=/opt/ttmlir-toolchain/python:$PYTHONPATH

# Or add to your virtual environment
source venv/bin/activate
echo "/opt/ttmlir-toolchain/python" > venv/lib/python*/site-packages/mlir.pth
```

## Option 2: Build MLIR with Python Bindings

### Prerequisites

- CMake 3.20 or higher
- Python 3.10 or higher with development headers
- C++ compiler (GCC 7+, Clang 5+, or MSVC 2019+)
- Ninja build system (recommended)

### Building LLVM/MLIR

```bash
# Clone LLVM project
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Create build directory
mkdir build && cd build

# Configure with Python bindings enabled
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=$(which python3)

# Build (this takes a while - 30+ minutes on most machines)
ninja

# Install (optional)
ninja install
```

### Adding to Python Path

After building:

```bash
# Add to PYTHONPATH
export PYTHONPATH=/path/to/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH

# Or create a .pth file in your virtual environment
echo "/path/to/llvm-project/build/tools/mlir/python_packages/mlir_core" > venv/lib/python*/site-packages/mlir.pth
```

## Option 3: Use pip install (if available)

Some LLVM distributions provide pip-installable packages:

```bash
pip install mlir
```

**Note**: This may not be available for all platforms or LLVM versions.

## Verifying Installation

Test that MLIR Python bindings are available:

```python
from mlir import ir

# Create a context and parse simple MLIR
with ir.Context() as ctx:
    module = ir.Module.parse('''
        module {
          func.func @test() {
            func.return
          }
        }
    ''')
    print(f"Parsed module: {module}")
```

## Testing the MLIR MCP Server

Once MLIR Python bindings are available, test the server:

```bash
# Run parsing tool tests
pytest tests/test_parser.py -v

# Test the server can start
python -m mlir_mcp_server
```

## Troubleshooting

### Import Error: No module named 'mlir'

The MLIR Python bindings are not in your Python path. Ensure:
1. MLIR was built with `-DMLIR_ENABLE_BINDINGS_PYTHON=ON`
2. `PYTHONPATH` includes the MLIR Python packages directory
3. Python version matches the one used to build MLIR

### Import Error: Symbol not found / DLL load failed

The MLIR C++ libraries are not found. Ensure:
1. `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) includes MLIR lib directory
2. On Windows, add the MLIR bin directory to `PATH`

Example:
```bash
export LD_LIBRARY_PATH=/path/to/llvm-project/build/lib:$LD_LIBRARY_PATH
```

### Type mismatch or version errors

MLIR Python bindings version must match your MLIR toolchain version. Ensure you're using bindings from the same LLVM build as your toolchain.

## For Development

When developing the MLIR MCP Server without MLIR Python bindings, the parsing tools will gracefully return an error message indicating that bindings are not available. This allows you to develop other parts of the server (configuration, infrastructure, non-parsing tools) without requiring MLIR Python bindings.
