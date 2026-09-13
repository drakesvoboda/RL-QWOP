"""Real QWOP environment using modern Selenium API."""
import os
import json
import time
import signal
import atexit
import logging
from typing import Dict, Tuple, Any, Optional
import numpy as np

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False

from .base import BaseQWOPEnv
from .server import GameServerManager

logger = logging.getLogger(__name__)


class RealQWOPEnv(BaseQWOPEnv):
    """
    Real QWOP environment using Selenium browser automation.

    Configuration via environment variables:
    - QWOP_URL: Game server URL (default: http://localhost:3000)
    - QWOP_HEADLESS: Run browser in headless mode (default: true)
    - QWOP_TIMEOUT: Page load timeout in seconds (default: 10)
    """

    def __init__(
        self,
        url: str = None,
        headless: bool = True,
        timeout: int = 10,
        enable_sound: bool = False,
        auto_start_server: bool = False,
        server_path: Optional[str] = None,
        server_port: int = 3100,
    ):
        """
        Initialize real QWOP environment.

        Args:
            url: Game server URL (uses QWOP_URL env var if not provided)
            headless: Run browser in headless mode
            timeout: Page load timeout
            enable_sound: Enable browser audio
            auto_start_server: Automatically start game server if not running
            server_path: Path to game server (auto-detect if None)
            server_port: Port for game server

        Environment variables:
            QWOP_URL: Override server URL
            QWOP_SERVER_PATH: Path to game server
            QWOP_HEADLESS: Run headless (true/false)
            QWOP_TIMEOUT: Page load timeout in seconds
        """
        if not HAS_SELENIUM:
            raise ImportError(
                "Selenium not installed. Install with: pip install selenium\n"
                "Or use MockQWOPEnv for testing without browser dependency."
            )

        super().__init__()

        self.server_manager = None
        self.url = url or os.getenv("QWOP_URL", f"http://localhost:{server_port}")
        self.headless = headless or os.getenv("QWOP_HEADLESS", "true").lower() == "true"
        self.timeout = timeout or int(os.getenv("QWOP_TIMEOUT", "10"))
        self.enable_sound = enable_sound

        self.driver = None
        self.canvas = None
        self.terminal_div = None
        self.state_elem = None
        self.max_fps = 8
        self.last_frame_time = time.time()

        # Auto-start game server if requested
        if auto_start_server:
            try:
                # Try to start server, but don't fail if another process beat us to it
                self.server_manager = GameServerManager(
                    server_path=server_path,
                    port=server_port,
                    auto_start=True,
                    timeout=self.timeout,
                )
            except RuntimeError as e:
                # Check if server is already running (maybe another process started it)
                import requests
                try:
                    requests.get(self.url, timeout=2)
                    logger.info(f"Game server already running at {self.url}")
                    # Server is running, that's fine
                except Exception:
                    logger.warning(f"Could not start game server: {e}")
                    logger.info("Make sure game server is running manually, or set auto_start_server=False")

        self._init_driver()
        self._connect_to_game()

        # Setup signal handlers for graceful shutdown
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGTSTP]:
            signal.signal(sig, self._signal_handler)
        atexit.register(self.close)

    def _init_driver(self):
        """Initialize Selenium WebDriver with modern API."""
        try:
            options = webdriver.FirefoxOptions()

            if self.headless:
                options.add_argument("--headless")

            if not self.enable_sound:
                options.set_preference("media.volume_scale", "0.0")

            # Try to find geckodriver automatically first
            geckodriver_path = self._find_geckodriver()

            print('geckodriver_path ', geckodriver_path)

            # Use service parameter for modern Selenium 4+
            if geckodriver_path:
                from selenium.webdriver.firefox.service import Service
                service = Service(geckodriver_path)
                self.driver = webdriver.Firefox(options=options, service=service)
            else:
                # Let Selenium find geckodriver in PATH
                self.driver = webdriver.Firefox(options=options)

            logger.info("Firefox WebDriver initialized successfully")

        except WebDriverException as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise RuntimeError(
                "Failed to start Firefox. Make sure Firefox is installed.\n"
                "Download: https://www.mozilla.org/en-US/firefox/new/\n"
                f"Error: {e}"
            )

    def _find_geckodriver(self) -> str:
        """Find geckodriver executable from environment variable or system PATH."""
        # Check env variable first
        env_path = os.getenv("GECKODRIVER_PATH")
        if env_path and os.path.exists(env_path):
            logger.info(f"Using geckodriver from GECKODRIVER_PATH: {env_path}")
            return env_path

        # Always use system geckodriver from PATH (via Selenium auto-discovery)
        # Don't look for bundled geckodriver - rely on system installation
        logger.info("Using system geckodriver from PATH")
        return None

    def _connect_to_game(self):
        """Connect to QWOP game server."""
        try:
            logger.info(f"Connecting to QWOP at {self.url}")

            # Ensure server is actually responding before opening in browser
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    import urllib.request
                    urllib.request.urlopen(self.url, timeout=2)
                    logger.info("Game server is responding, loading in browser...")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Game server not responding at {self.url}: {e}")
                    time.sleep(0.5)

            self.driver.get(f"{self.url}?webgl=true")
            logger.info("Page loaded in Firefox, waiting for game elements...")

            # Wait for game to load - wait for terminal element to appear and have content
            wait = WebDriverWait(self.driver, self.timeout + 5)
            # First wait for the static elements
            logger.debug("Waiting for terminal element...")
            wait.until(EC.presence_of_element_located((By.ID, "terminal")))
            logger.debug("Terminal element found, waiting for state element...")
            wait.until(EC.presence_of_element_located((By.ID, "state")))
            logger.debug("State element found, waiting for game canvas...")
            # Wait for the game canvas element (created by JavaScript)
            wait.until(EC.presence_of_element_located((By.ID, "window1")))
            logger.debug("Game canvas loaded, waiting for game state...")

            # Wait for game state to be populated (non-empty)
            # The game client initializes asynchronously, so we need to wait
            max_state_wait = 15
            state_elem_cached = None
            for attempt in range(max_state_wait * 2):
                try:
                    if not state_elem_cached:
                        state_elem_cached = self.driver.find_element(By.ID, "state")

                    text = state_elem_cached.get_attribute("innerHTML")
                    if text and len(text.strip()) > 2:
                        logger.debug(f"Game state initialized after {attempt * 0.5:.1f}s")
                        break

                    time.sleep(0.5)
                except Exception as e:
                    logger.debug(f"Error checking state: {e}")
                    time.sleep(0.5)
            else:
                # Timeout - let's check what's in the console
                try:
                    logs = self.driver.get_log("browser")
                    errors = [log for log in logs if log["level"] in ["SEVERE", "WARNING"]]
                    if errors:
                        logger.warning(f"Browser console errors: {errors[:3]}")
                except:
                    pass
                logger.warning("Game state not initialized within timeout, attempting to continue...")

            logger.debug("Game initialization complete")

            # Get game elements (modern API)
            self.canvas = self.driver.find_element(By.ID, "window1")
            self.terminal_div = self.driver.find_element(By.ID, "terminal")
            self.state_elem = self.driver.find_element(By.ID, "state")

            logger.info("QWOP game loaded successfully")
            # Initialize state trackers - will be populated on first step
            self.last_state_time = time.time()
            self.old_posx = 0
            self.old_velx = 0
            self.last_frame_time = time.time()

        except TimeoutException:
            logger.error(
                f"Game server not responding at {self.url}. "
                "Make sure the QWOP server is running.\n"
                "See: https://github.com/drakesvoboda/QWOP for setup instructions."
            )
            self.close()
            raise RuntimeError(f"QWOP server not responding at {self.url}")

    def get_state(self) -> Dict[str, float]:
        """Get current game state."""
        try:
            state_str = self.state_elem.get_attribute("innerHTML")

            # If state is empty, return a default state
            if not state_str or len(state_str.strip()) < 2:
                logger.debug("State element is empty, using default state")
                return {
                    "posx": 0,
                    "quads": [0, 0, 0, 0],
                    "terminal": False,
                    "diffx": 0,
                    "velx": 0,
                    "accelx": 0,
                }

            state = json.loads(state_str)
            state["terminal"] = (
                self.terminal_div.get_attribute("innerHTML").lower() == "true"
            )

            current_time = time.time()
            delta = current_time - self.last_state_time
            self.last_state_time = current_time

            state["diffx"] = state.get("posx", 0) - self.old_posx
            state["velx"] = state["diffx"] / max(delta, 0.001)
            state["accelx"] = (state["velx"] - self.old_velx) / max(delta, 0.001)

            self.old_posx = state.get("posx", 0)
            self.old_velx = state["velx"]

            return state

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing state JSON: {e}")
            raise RuntimeError(f"Failed to get game state: {e}")
        except Exception as e:
            logger.error(f"Error getting state: {e}")
            raise RuntimeError(f"Failed to get game state: {e}")

    def compute_reward(self, state: Dict[str, float] = None) -> float:
        """Compute reward based on forward progress."""
        state = state or self.get_state()

        current_time = time.time()
        delta = current_time - self.last_reward if hasattr(self, "last_reward") else 0.1
        self.last_reward = current_time

        reward = state.get("diffx", 0) / max(delta, 0.001)

        if state.get("terminal", False):
            reward = -1.0

        return float(reward)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        self._throttle()

        state = self.get_state()
        obs = self.obs(state)
        reward = self.compute_reward(state)

        # Apply action to game
        self._apply_action(action)

        terminated = state.get("terminal", False)
        truncated = False  # No timeout in real environment

        return obs, reward, terminated, truncated, {"state": state}

    def _apply_action(self, action: np.ndarray):
        """Apply action (key presses) to game."""
        q_press, w_press, o_press, p_press = action

        action_chain = ActionChains(self.driver)

        if q_press:
            action_chain.key_down("q")
        else:
            action_chain.key_up("q")

        if w_press:
            action_chain.key_down("w")
        else:
            action_chain.key_up("w")

        if o_press:
            action_chain.key_down("o")
        else:
            action_chain.key_up("o")

        if p_press:
            action_chain.key_down("p")
        else:
            action_chain.key_up("p")

        action_chain.perform()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset game to initial state."""
        try:
            self.canvas.click()
            ActionChains(self.driver).key_up("p").key_up("o").key_up("w").key_up(
                "q"
            ).key_down("r").key_up("r").perform()

            time.sleep(0.03)

            self.last_reward = time.time()
            state = self.get_state()
            self.last_frame_time = time.time()
            self.old_velx = 0.0

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

    def _signal_handler(self, sig, frame):
        """Handle system signals."""
        logger.info(f"Received signal {sig}, cleaning up...")
        self.close()

    def close(self):
        """Clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("WebDriver closed successfully")
            except Exception as e:
                logger.warning(f"Error closing WebDriver: {e}")
            finally:
                self.driver = None

        if self.server_manager:
            try:
                self.server_manager.stop()
            except Exception as e:
                logger.warning(f"Error stopping server: {e}")
            finally:
                self.server_manager = None
