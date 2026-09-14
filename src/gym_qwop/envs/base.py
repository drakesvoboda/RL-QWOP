"""Base environment interface for QWOP.

Also holds the infrastructure shared by both the Selenium (qwop_env.py) and WebSocket
(qwop_websocket.py) environments: starting the Node game server, launching a Firefox
tab to run it, frame-rate throttling, and resource cleanup.
"""
import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import numpy as np
import gymnasium
from gymnasium import spaces
from selenium import webdriver
from selenium.common.exceptions import WebDriverException

from .server import GameServerManager

logger = logging.getLogger(__name__)


class BaseQWOPEnv(gymnasium.Env, ABC):
    """Abstract base class for QWOP environments."""

    metadata = {'render_modes': ['human', 'realtime'], 'render_fps': 60}

    def __init__(self, render_mode=None, render_fps=60):
        """Initialize observation and action spaces."""
        self.render_mode = render_mode
        self.render_fps = render_fps
        # 16 = one action per combination of the 4 keys (Q, W, O, P) being up/down,
        # modeled as a single Categorical rather than 4 independent Bernoullis - see
        # MLPActor. Bit i of the action index is key i's state; QWOPEnv._decode_action
        # turns it back into the [q, w, o, p] array the game expects.
        self.action_space = spaces.Discrete(16)

        # State vector: [posy, 8 joint positions, 8 joint velocities, velx]
        # Total: 1 + 8 + 8 + 1 = 18 dimensions
        high = np.array([np.inf] * 18, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.last_reward = 0.0
        self.last_state_time = 0.0
        self.old_posx = 0.0
        self.old_velx = 0.0

        self.driver = None
        self.server_manager = None
        self.max_fps = render_fps
        self.last_frame_time = time.time()

    def _resolve_server_port(self, default: int) -> int:
        """Resolve this instance's game server port: explicit arg < QWOP_SERVER_PORT env
        var < `default`. Each parallel training worker needs a distinct port (rlqwop.PPO
        sets QWOP_SERVER_PORT per worker when running with num_procs > 1)."""
        return int(os.getenv("QWOP_SERVER_PORT", str(default)))

    def _start_game_server(self, server_path: Optional[str], server_port: int, timeout: int) -> Optional[GameServerManager]:
        """Start (or attach to) the Node game server, or None if it couldn't be started."""
        try:
            return GameServerManager(
                server_path=server_path,
                port=server_port,
                auto_start=True,
                timeout=timeout,
            )
        except RuntimeError as e:
            logger.warning(f"Could not start game server: {e}")
            return None

    def _launch_browser(self, url: str, headless: bool, enable_sound: bool = False):
        """Launch a Firefox tab pointed at `url` and return the WebDriver."""
        try:
            options = webdriver.FirefoxOptions()
            if headless:
                options.add_argument("--headless")
            if not enable_sound:
                options.set_preference("media.volume_scale", "0.0")

            driver = webdriver.Firefox(options=options)
            driver.get(url)
            return driver

        except WebDriverException as e:
            logger.error(f"Failed to launch browser: {e}")
            raise RuntimeError(
                "Failed to start Firefox. Make sure Firefox is installed.\n"
                "Download: https://www.mozilla.org/en-US/firefox/new/\n"
                f"Error: {e}"
            )

    def _throttle(self, target_dt: Optional[float] = None):
        """Sleep as needed to pace stepping to `target_dt` seconds per call (default
        1/self.max_fps) - only when actually rendering ("human" or "realtime" mode).
        Training (render_mode=None) doesn't need real-time pacing, and capping it would
        limit rollout throughput regardless of how fast the sim itself can run.

        render_mode="human" always uses the default 1/max_fps regardless of `target_dt` -
        it's for watching at a fun, fixed frame rate, not literal real-time (see
        render_mode="realtime" for that, which passes the actual amount of simulated time
        each step covers, e.g. ticks_per_step * PHYSICS_DT on QWOPEnv)."""
        if self.render_mode != "realtime":
            return
        if target_dt is None:
            target_dt = 1.0 / self.max_fps
        current_time = time.time()
        delta = current_time - self.last_frame_time
        if delta < target_dt:
            time.sleep(target_dt - delta)
        self.last_frame_time = time.time()

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Returns:
            obs: observation
            reward: reward
            terminated: episode finished (e.g., fell over)
            truncated: episode timeout (max steps reached)
            info: additional info dict
        """
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            obs: initial observation
            info: additional info dict
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, float]:
        """Get current state from the environment."""
        pass

    @abstractmethod
    def compute_reward(self, state: Dict[str, float] = None) -> float:
        """Compute reward based on state."""
        pass

    def obs(self, state: Dict[str, float] = None) -> np.ndarray:
        """Convert state to observation vector (positions + velocities)."""
        state = state or self.get_state()
        joint_names = ["head", "torso", "leftarm", "rightarm", "leftthigh", "rightthigh", "leftcalf", "rightcalf"]

        # Positions
        obs = [state.get("posy", 0)]
        obs.extend([state.get(joint, 0) for joint in joint_names])

        # Velocities (from state if available)
        obs.extend([state.get(f"{joint}_vel", 0) for joint in joint_names])

        # Horizontal velocity
        obs.append(state.get("velx", 0))

        return np.array(obs, dtype=np.float32)

    def render(self, mode='human') -> None:
        """Render environment (optional)."""
        pass

    def get_render_state(self) -> str:
        """Get a human-readable representation of current state."""
        state = self.get_state()
        return (
            f"Position: {state.get('posx', 0):.2f}  "
            f"Height: {state.get('posy', 0):.2f}  "
            f"Velocity: {state.get('velx', 0):.2f}  "
            f"Terminal: {state.get('terminal', False)}"
        )

    def close(self) -> None:
        """Clean up the browser and game server. Subclasses with extra resources (e.g. a
        websocket) should close those first, then call super().close()."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Browser closed successfully")
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            finally:
                self.driver = None

        if self.server_manager:
            try:
                self.server_manager.stop()
            except Exception as e:
                logger.warning(f"Error stopping server: {e}")
            finally:
                self.server_manager = None
