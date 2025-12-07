# Contributing to MLIR MCP Server

Thank you for your interest in contributing to the MLIR MCP Server!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/brnorris03/mlir-mcp-server.git
cd mlir-mcp-server
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

For coverage report:
```bash
pytest --cov=mlir_mcp_server --cov-report=html
```

## Code Style

We use:
- **black** for code formatting
- **ruff** for linting
- **mypy** for type checking

Run all checks:
```bash
black src tests
ruff check src tests
mypy src
```

## Development Workflow

1. Create a new branch for your feature/fix
2. Write tests for your changes
3. Implement your changes
4. Run tests and style checks
5. Submit a pull request

## Project Structure

```
src/mlir_mcp_server/
├── config.py           # Configuration management
├── server.py           # Main MCP server
├── tools/              # MCP tools implementation
├── resources/          # MCP resources
├── prompts/            # MCP prompts
├── mlir_wrapper/       # MLIR Python API wrappers
└── utils/              # Utility modules
```

## Commit Message Guidelines

- Use clear, descriptive commit messages
- Reference issue numbers when applicable
- Follow the format: `[category] brief description`

Example:
```
[config] Add support for custom timeout values
[tools] Implement parse_mlir tool
[tests] Add tests for transformer module
```

## Questions?

Feel free to open an issue for any questions or concerns!
