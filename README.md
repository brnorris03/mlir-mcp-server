# MLIR MCP Server

MCP (Model Context Protocol) server for MLIR (Multi-Level Intermediate
Representation) manipulation. This server enables AI assistants like Claude to
parse, generate, transform, and analyze MLIR code through a standardized
protocol.

## Features

### MCP Tools

The server provides 9 MCP tools for MLIR manipulation:

**Server Management:**
1. `ping` - Health check endpoint
2. `toolchain_info` - Get toolchain configuration and tool availability

**Parsing & Validation:**
3. `parse_mlir` - Parse MLIR code and extract module information
4. `validate_mlir` - Validate MLIR syntax and semantics
5. `get_module_info` - Extract detailed structural information (operations,
   regions, blocks)

**Code Generation:**
6. `create_function` - Generate MLIR functions with custom signatures
7. `create_operation` - Generate specific MLIR operations
8. `generate_from_template` - Generate MLIR from predefined templates
9. `list_templates` - List available generation templates

### Configuration

- **Zero-configuration setup**: Auto-detects MLIR toolchain from common
  installation locations
- **Flexible configuration**: Supports `.env` files and environment variables
  for custom toolchain paths
- **Simple and extensible**: Minimal configuration with room for future
  expansion
- **Relaxed validation**: Works with minimal MLIR installations (core tools
  only), with graceful fallback for optional tools

## Installation

### Prerequisites

- bash or compatible shell
- Python 3.10 or higher
- MLIR toolchain installed (from LLVM project or custom build)
- MLIR Python bindings (optional, but required for parsing tools)

**Important**: Use the same Python version as your MLIR Python bindings. Check
by examining the extension modules:

```bash
ls /path/to/mlir/python_packages/mlir_core/mlir/_mlir_libs/
# Look for: _mlir.cpython-311-darwin.so (Python 3.11)
#       or: _mlir.cpython-310-linux.so (Python 3.10)
#                  ^^^^^^^^
```

### From Source

```bash
git clone https://github.com/brnorris03/mlir-mcp-server.git
cd mlir-mcp-server

# Create virtual environment with Python version matching your MLIR bindings
# Example: if MLIR was built with Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### For Development

```bash
pip install -e ".[dev]"
```

### MLIR Python Bindings Setup

**Automatic Setup (Recommended)**: Use the `MLIR_INSTALLATIONS` configuration
option (see Configuration section below). The server will automatically
configure PYTHONPATH and library paths for you!

**Manual Setup** (if needed):

```bash
# Set MLIR Python bindings path
export PYTHONPATH=/path/to/mlir/python_packages/mlir_core:$PYTHONPATH

# Set library path for MLIR shared libraries
# macOS:
export DYLD_LIBRARY_PATH=/path/to/mlir/lib:$DYLD_LIBRARY_PATH
# Linux:
export LD_LIBRARY_PATH=/path/to/mlir/lib:$LD_LIBRARY_PATH
# Windows:
set PATH=C:\path\to\mlir\bin;%PATH%
```

**Helper Scripts**: Use these to run the server/tests:
- `./run_server.sh` - Run the server
- `./run_tests.sh` - Run tests

## Configuration

The MLIR MCP Server uses a simple, flexible configuration system with
auto-detection.

### Auto-Detection

The server automatically searches for MLIR toolchain in these locations (in
order):
1. `/opt/ttmlir-toolchain/bin`
2. `/usr/local/llvm/bin`
3. `/usr/local/bin`
4. `/opt/llvm/bin`
5. `~/llvm-project/build/bin`

If your MLIR toolchain is in one of these locations with `mlir-opt` present, no
configuration is needed!

### Simple Configuration

For a single MLIR installation, use `MLIR_TOOLCHAIN_PATH`:

```bash
export MLIR_TOOLCHAIN_PATH=/path/to/llvm/build/bin
python -m mlir_mcp_server
```

### Multiple MLIR Installations

For multiple MLIR projects (e.g., main LLVM + custom dialects), use
`MLIR_INSTALLATIONS`:

```bash
# Automatically configures PYTHONPATH, library paths, and tools
export MLIR_INSTALLATIONS=/opt/llvm-build:/opt/custom-project
python -m mlir_mcp_server
```

Each installation root should contain:
- `bin/` - MLIR tools (mlir-opt, custom-opt, etc.)
- `lib/` - Shared libraries
- `python_packages/mlir_core/` - Python bindings

The server automatically:
- Searches all bin directories for tools
- Adds all Python bindings to sys.path
- Sets platform-specific library paths (DYLD_LIBRARY_PATH on macOS,
  LD_LIBRARY_PATH on Linux)

### Configuration File (.env)

Create a `.env` file in your project root for persistent configuration.

**Option 1: Simple Single Installation**

```env
MLIR_TOOLCHAIN_PATH=/path/to/llvm/build/bin
```

**Option 2: Multiple MLIR Installations (Recommended)**

Automatically configures tools, Python bindings, and library paths:

```env
# Colon-separated list of MLIR installation roots
# Each should have: bin/, lib/, python_packages/mlir_core/
MLIR_INSTALLATIONS=/path/to/llvm-build:/path/to/custom-mlir-project

# Example with main LLVM + custom "TT" project:
# MLIR_INSTALLATIONS=/opt/llvm:/opt/tt-mlir
```

This automatically:
- Searches both `/opt/llvm/bin` and `/opt/tt-mlir/bin` for tools
- Adds both Python bindings to PYTHONPATH
- Adds both lib directories to LD_LIBRARY_PATH/DYLD_LIBRARY_PATH
- Makes tools like `mlir-opt` and `tt-opt` available

**No manual PYTHONPATH or LD_LIBRARY_PATH setup needed!**

### Configuration Priority

The server resolves the toolchain path using this priority (highest to lowest):
1. `.env` configuration file
2. Environment variables
3. Auto-detected from common locations

## Usage with Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mlir": {
      "command": "python",
      "args": ["-m", "mlir_mcp_server"],
      "env": {
        "MLIR_TOOLCHAIN_PATH": "/usr/local/llvm/bin"
      }
    }
  }
}
```

**Note**: On Windows, use forward slashes or escaped backslashes in paths (e.g.,
`"C:/LLVM/bin"` or `"C:\\LLVM\\bin"`).

If you omit `MLIR_TOOLCHAIN_PATH`, the server will auto-detect the toolchain.

## Running the Server

### Standalone

```bash
python -m mlir_mcp_server
```

The server uses stdio transport and is ready to receive MCP requests.

### Testing the Installation

After starting the server, you can test it using an MCP client or Claude
Desktop. The server provides two basic tools:

- `ping`: Health check that returns server status
- `toolchain_info`: Information about the configured MLIR toolchain and
  available tools

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src tests
ruff check src tests
```

### Type Checking

```bash
mypy src
```

## Toolchain Validation

The server validates the MLIR toolchain at startup:

- Core tools (required): `mlir-opt`, `mlir-translate`
- Optional tools: `mlir-reduce`, `mlir-query`, `mlir-runner`

If core tools are missing, the server exits with an error. If optional tools are
missing, warnings are logged and corresponding MCP tools will be disabled when
implemented.

## Project Status

This project is in active development. Currently implemented:

- Configuration system with auto-detection
- Multiple MLIR installations support
- Automatic Python bindings and library path setup
- MCP server with 9 tools
- MLIR parsing tools (parse, validate, get module info)
- MLIR generation tools (create functions, operations, templates)
- Comprehensive test suite (47 tests, 71% coverage)

Coming soon:
- Transformation and optimization tools
- Analysis and query tools
- Resource providers for dialect documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Boyana Norris (brnorris03@gmail.com)
