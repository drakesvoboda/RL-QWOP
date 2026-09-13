"""Game server management for QWOP environment."""
import os
import subprocess
import time
import socket
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)


class GameServerManager:
    """Manage QWOP game server lifecycle."""

    def __init__(
        self,
        server_path: Optional[str] = None,
        port: int = 3000,
        auto_start: bool = True,
        timeout: int = 10,
    ):
        """
        Initialize game server manager.

        Args:
            server_path: Path to game server directory (auto-detect if None)
            port: Port to run server on
            auto_start: Automatically start server if not running
            timeout: Timeout for server startup in seconds
        """
        self.server_path = server_path or self._find_server_path()
        self.port = port
        self.auto_start = auto_start
        self.timeout = timeout
        self.process = None
        self.url = f"http://localhost:{port}"

        if self.server_path is None:
            raise RuntimeError(
                "Could not find QWOP game server. "
                "Set QWOP_SERVER_PATH environment variable or pass server_path."
            )

        if self.auto_start:
            self.start()

    def _find_server_path(self) -> Optional[str]:
        """Auto-detect game server path."""
        # Check environment variable
        if "QWOP_SERVER_PATH" in os.environ:
            return os.environ["QWOP_SERVER_PATH"]

        # Try to find relative to this file
        base_paths = [
            # From gym-qwop package
            os.path.join(os.path.dirname(__file__), "../../../game"),
            # From project root
            os.path.join(os.getcwd(), "game"),
            # Absolute paths
            "/Users/drakesvoboda/Documents/Projects/RL-QWOP/game",
        ]

        for path in base_paths:
            if os.path.exists(os.path.join(path, "server.js")):
                return os.path.abspath(path)

        return None

    def is_running(self) -> bool:
        """Check if server is already running on port."""
        try:
            response = requests.get(self.url, timeout=2)
            return response.status_code < 500
        except (requests.ConnectionError, requests.Timeout):
            return False

    def is_server_ready(self) -> bool:
        """Check if server is fully initialized and responding."""
        try:
            response = requests.get(self.url, timeout=2)
            # Server should return HTML with game
            return response.status_code == 200 and len(response.text) > 100
        except (requests.ConnectionError, requests.Timeout):
            return False

    def start(self) -> bool:
        """Start the game server."""
        # Check if already running
        if self.is_server_ready():
            logger.info(f"Game server already running at {self.url}")
            return True

        # Check if Node.js is installed
        if not self._check_nodejs():
            raise RuntimeError(
                "Node.js not found. Install with: brew install node\n"
                "Or download: https://nodejs.org/"
            )

        logger.info(f"Starting QWOP game server at {self.url}...")

        try:
            # Install dependencies if needed
            package_json = os.path.join(self.server_path, "package.json")
            if os.path.exists(package_json):
                self._ensure_dependencies()

            # Start server
            env = os.environ.copy()
            env["PORT"] = str(self.port)
            logger.debug(f"Starting Node server with PORT={self.port} from {self.server_path}")
            self.process = subprocess.Popen(
                ["node", "server.js"],
                cwd=self.server_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
            )

            # Wait for server to be ready
            return self._wait_for_server()

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise RuntimeError(f"Failed to start QWOP server: {e}")

    def _check_nodejs(self) -> bool:
        """Check if Node.js is installed."""
        try:
            subprocess.run(
                ["node", "--version"],
                capture_output=True,
                timeout=5,
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _ensure_dependencies(self) -> bool:
        """Install npm dependencies if needed."""
        try:
            logger.info("Checking npm dependencies...")
            node_modules = os.path.join(self.server_path, "node_modules")

            if not os.path.exists(node_modules):
                logger.info("Installing npm dependencies...")
                result = subprocess.run(
                    ["npm", "install"],
                    cwd=self.server_path,
                    capture_output=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    logger.warning(f"npm install output: {result.stderr.decode()}")
                    return False

            return True

        except subprocess.TimeoutExpired:
            logger.error("npm install timed out")
            return False
        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
            return False

    def _wait_for_server(self, max_retries: int = 60) -> bool:
        """Wait for server to be ready."""
        for attempt in range(max_retries):
            if self.is_server_ready():
                logger.info(f"✓ Game server ready at {self.url}")
                return True

            time.sleep(0.5)

            # Check if process crashed
            if self.process:
                exit_code = self.process.poll()
                if exit_code is not None:
                    stdout, stderr = self.process.communicate()
                    logger.error(f"Server process crashed (exit code {exit_code})")
                    logger.error(f"  stdout: {stdout[:200] if stdout else '(empty)'}")
                    logger.error(f"  stderr: {stderr[:200] if stderr else '(empty)'}")
                    raise RuntimeError(f"Server failed to start: {stderr}")

            if (attempt + 1) % 10 == 0:
                logger.info(f"Waiting for server... ({attempt + 1}s)")

        raise RuntimeError(f"Server did not respond within {self.timeout}s")

    def stop(self):
        """Stop the game server."""
        if self.process and self.process.poll() is None:
            logger.info("Stopping QWOP game server...")
            self.process.terminate()

            try:
                self.process.wait(timeout=5)
                logger.info("Game server stopped")
            except subprocess.TimeoutExpired:
                logger.warning("Server did not stop gracefully, killing...")
                self.process.kill()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
