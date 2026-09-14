"""WebSocket-based QWOP environment (10-20x faster than Selenium)."""
import atexit
import json
import time
import logging
from typing import Dict, Tuple, Any, Optional
import numpy as np
import websocket
import threading

from .base import BaseQWOPEnv

logger = logging.getLogger(__name__)


class WebSocketQWOPEnv(BaseQWOPEnv):
    """
    QWOP environment using WebSocket for direct communication with game.

    This is 10-20x faster than Selenium-driven stepping because per-step interaction
    (actions in, state out) goes over a WebSocket instead of the WebDriver protocol.
    A browser tab is still launched once at startup (see `BaseQWOPEnv._launch_browser`)
    to actually run the game's JS/Box2D physics - headless unless render_mode="human".

    server.js relays messages between this client and the browser tab running the game
    (see game/public/ws.js), so both must be connected to the same server instance.
    """

    def __init__(
        self,
        render_mode=None,
        render_fps=60,
        url: Optional[str] = None,
        timeout: int = 10,
        auto_start_server: bool = True,
        server_path: Optional[str] = None,
        server_port: Optional[int] = None,
        launch_browser: bool = False,
    ):
        """
        Initialize WebSocket QWOP environment.

        Args:
            url: WebSocket URL to game server (derived from server_port if not given)
            timeout: Connection/startup timeout in seconds
            auto_start_server: Automatically start game server
            server_path: Path to game server
            server_port: Port for game server (falls back to QWOP_SERVER_PORT env var,
                then 3000). Each parallel instance needs its own port.
            launch_browser: Automatically launch a browser tab to run the game. The tab
                is visible with render_mode="human", headless otherwise.
        """
        super().__init__(render_mode, render_fps)

        if server_port is None:
            server_port = self._resolve_server_port(3100)

        self.server_port = server_port
        self.http_url = f"http://localhost:{server_port}"
        self.url = url or f"ws://localhost:{server_port}"
        self.timeout = timeout
        self.headless = self.render_mode not in ("human", "realtime")
        self.ws = None
        self.ws_thread = None
        self.last_state = None

        self.server_manager = self._start_game_server(server_path, server_port, timeout) if auto_start_server else None

        # Connect to WebSocket
        self._connect_websocket()

        if launch_browser:
            self.driver = self._launch_browser(f"{self.http_url}/?webgl=true", self.headless)

        atexit.register(self.close)

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

    def get_state(self) -> Dict[str, float]:
        """Get current state (from last server response)."""
        return self.last_state or {
            "posx": 0.0,
            "posy": 0.0,
            "velx": 0.0,
            "terminal": False,
        }

    def compute_reward(self, state: Dict[str, float] = None) -> float:
        """Reward is computed by the browser client and returned in the response."""
        state = state or self.get_state()
        return float(state.get("reward", 0.0))

    def _request(self, message: Dict[str, Any], timeout: float, retry: bool = False) -> Dict[str, Any]:
        """Send a message and wait for the matching state response.

        With `retry`, keeps retrying (e.g. while the game browser is still starting up)
        until `timeout` seconds have elapsed rather than failing on the first attempt.
        """
        deadline = time.time() + timeout

        while True:
            self.last_state = None
            try:
                self.ws.send(json.dumps(message))
            except Exception as e:
                raise RuntimeError(f"Failed to send message to game server: {e}")

            wait_until = time.time() + (0.5 if retry else 0.1)
            while self.last_state is None and time.time() < wait_until:
                time.sleep(0.001)

            if self.last_state is not None and "error" not in self.last_state:
                return self.last_state

            if not retry or time.time() >= deadline:
                reason = self.last_state.get("error") if self.last_state else "no response"
                raise RuntimeError(f"Game server did not return a valid state ({reason})")

            time.sleep(0.2)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step via WebSocket."""
        self._throttle()

        # Decode the Discrete(16) action index into the [q, w, o, p] array the browser's
        # actionToKeys() expects (see QWOPEnv._decode_action for why the action space is
        # one Categorical over key combinations rather than 4 independent Bernoullis).
        key_states = [(int(action) >> i) & 1 for i in range(4)]
        state = self._request({"type": "step", "action": key_states}, timeout=0.5)

        obs = self.obs(state)
        reward = state.get("reward", 0.0)
        terminated = state.get("terminated", False)
        truncated = state.get("truncated", False)

        return obs, float(reward), terminated, truncated, {"state": state}

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset game to initial state."""
        # Long timeout + retry: the game browser may still be loading on the very first reset.
        state = self._request({"type": "reset"}, timeout=15.0, retry=True)

        self.last_frame_time = time.time()

        return self.obs(state), {}

    def render(self, mode='human', rank=None, state=None):
        """Print human-readable state. Pass `state` (e.g. the info['state'] a just-completed
        step() already returned) to avoid an extra round trip re-fetching it."""
        state = state if state is not None else self.get_state()
        rank_str = f"[Agent {rank}] " if rank is not None else ""
        print(
            f"\r{rank_str}[QWOP-WS] Pos: {state.get('posx', 0):6.2f}  "
            f"Height: {state.get('posy', 0):5.2f}  "
            f"Vel: {state.get('velx', 0):6.2f}  "
            f"Terminal: {state.get('terminal', False)}",
            end="", flush=True
        )

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

        super().close()
