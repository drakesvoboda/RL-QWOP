"""WebSocket-based QWOP environment (10-20x faster than Selenium)."""
import json
import time
import logging
from typing import Dict, Tuple, Any, Optional
import numpy as np
import websocket
import threading

from .base import BaseQWOPEnv
from .server import GameServerManager

logger = logging.getLogger(__name__)


class WebSocketQWOPEnv(BaseQWOPEnv):
    """
    QWOP environment using WebSocket for direct communication with game.

    This is 10-20x faster than Selenium because it eliminates:
    - WebDriver protocol overhead (100-200ms per call)
    - DOM querying and parsing
    - Browser automation latency

    Instead, it communicates directly with the game via WebSocket messages.
    """

    def __init__(
        self,
        url: str = "ws://localhost:3000/api/env",
        headless: bool = True,
        timeout: int = 10,
        auto_start_server: bool = True,
        server_path: Optional[str] = None,
        server_port: int = 3000,
    ):
        """
        Initialize WebSocket QWOP environment.

        Args:
            url: WebSocket URL to game server
            headless: Run browser in headless mode (for Selenium fallback)
            timeout: Connection timeout
            auto_start_server: Automatically start game server
            server_path: Path to game server
            server_port: Port for game server
        """
        super().__init__()

        self.url = url
        self.timeout = timeout
        self.server_manager = None
        self.ws = None
        self.ws_thread = None

        # Auto-start game server
        if auto_start_server:
            try:
                self.server_manager = GameServerManager(
                    server_path=server_path,
                    port=server_port,
                    auto_start=True,
                    timeout=timeout,
                )
            except RuntimeError as e:
                logger.warning(f"Could not start game server: {e}")

        # Connect to WebSocket
        self._connect_websocket()

        # For velocity tracking
        self.last_joint_angles = {}
        self.joint_names = ["head", "torso", "leftarm", "rightarm", "leftthigh", "rightthigh", "leftcalf", "rightcalf"]
        self.max_fps = 60
        self.last_frame_time = time.time()

    def _connect_websocket(self):
        """Connect to game server via WebSocket."""
        try:
            logger.info(f"Connecting to QWOP via WebSocket: {self.url}")
            self.ws = websocket.WebSocketApp(
                self.url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self.ws.on_open = self._on_open

            # Run WebSocket in background thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.ws_thread.start()

            # Wait for connection
            time.sleep(0.5)
            if not self.ws or not self.ws.keep_running:
                raise RuntimeError("WebSocket connection failed")

            logger.info("WebSocket connected successfully")
            self.last_state = None

        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            raise RuntimeError(f"Failed to connect to QWOP: {e}")

    def _on_open(self, ws):
        """Handle WebSocket connection open."""
        logger.info("WebSocket connection opened")

    def _on_message(self, ws, message):
        """Handle WebSocket message."""
        try:
            self.last_state = json.loads(message)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message}")

    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        logger.info("WebSocket connection closed")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step via WebSocket."""
        self._throttle()

        # Send action to game
        message = {"action": action.tolist()}
        try:
            self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send action: {e}")
            raise

        # Wait for response (should arrive within 16ms at 60 FPS)
        timeout = time.time() + 0.1
        while self.last_state is None and time.time() < timeout:
            time.sleep(0.001)  # Small sleep to avoid busy-wait

        if self.last_state is None:
            logger.error("No response from game server")
            raise RuntimeError("Game server not responding")

        state = self.last_state
        self.last_state = None

        obs = self.obs(state)
        reward = state.get("reward", 0.0)
        terminated = state.get("terminated", False)
        truncated = state.get("truncated", False)

        return obs, float(reward), terminated, truncated, {"state": state}

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset game to initial state."""
        try:
            message = {"action": "reset"}
            self.ws.send(json.dumps(message))

            # Wait for reset response
            timeout = time.time() + 1.0
            while self.last_state is None and time.time() < timeout:
                time.sleep(0.01)

            if self.last_state is None:
                raise RuntimeError("Reset response timeout")

            state = self.last_state
            self.last_state = None

            self.last_frame_time = time.time()
            self.last_joint_angles = {}

            return self.obs(state), {}

        except Exception as e:
            logger.error(f"Error during reset: {e}")
            raise RuntimeError(f"Failed to reset game: {e}")

    def _throttle(self):
        """Maintain frame rate."""
        current_time = time.time()
        delta = current_time - self.last_frame_time
        min_delta = 1.0 / self.max_fps
        if delta < min_delta:
            time.sleep(min_delta - delta)
        self.last_frame_time = time.time()

    def close(self):
        """Clean up resources."""
        if self.ws:
            try:
                self.ws.close()
                logger.info("WebSocket closed successfully")
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self.ws = None

        if self.server_manager:
            try:
                self.server_manager.stop()
            except Exception as e:
                logger.warning(f"Error stopping server: {e}")
            finally:
                self.server_manager = None
