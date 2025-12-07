"""Entry point for MLIR MCP Server.

This module initializes the MCP server with configuration, validates the MLIR
toolchain, and starts the server using stdio transport.
"""

import logging
import sys

from .config import MLIRConfig
from .server import create_server


def main() -> None:
    """Initialize and run the MLIR MCP server."""
    try:
        # Configure logging to stderr (avoid corrupting JSON-RPC on stdout)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stderr,
        )
        logger = logging.getLogger(__name__)

        # Load configuration with priority: .env > env > auto-detect
        logger.info("Loading MLIR MCP Server configuration...")
        config = MLIRConfig()
        logger.info(f"MLIR toolchain path: {config.toolchain_path}")

        # Validate toolchain (relaxed - only core tools required)
        logger.info("Validating MLIR toolchain...")
        tool_status = config.validate_tools()
        logger.info(f"Tool availability: {tool_status}")

        # Create and run server
        logger.info("Starting MLIR MCP Server...")
        mcp = create_server(config)
        mcp.run(transport="stdio")

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
