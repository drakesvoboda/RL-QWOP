"""Real QWOP environment using modern Selenium API."""
import os
import time
import math
import signal
import atexit
import logging
from typing import Dict, Tuple, Any, Optional
import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from .base import BaseQWOPEnv

logger = logging.getLogger(__name__)


class QWOPEnv(BaseQWOPEnv):
    """
    Real QWOP environment using Selenium browser automation.

    Configuration via environment variables:
    - QWOP_URL: Game server URL (default: http://localhost:3000)
    - QWOP_HEADLESS: Run browser in headless mode (default: true)
    - QWOP_TIMEOUT: Page load timeout in seconds (default: 10)
    """

    # Fixed simulated seconds per physics tick (matches world.step(0.04, ...) in QWOP.min.js).
    PHYSICS_DT = 0.04

    # Deadzone half-width (radians) around stride≈0 that counts as "ambiguous" rather
    # than clearly left- or right-leading - avoids chattering swap detection right at
    # the midline. Held keys apply continuous torque (thigh angles aren't bounded to a
    # small range - they can spin past +/-pi), so the lead side is read off
    # wrap_to_pi(leftthigh - rightthigh), not the raw angle difference.
    STRIDE_SWAP_THRESHOLD = 0.5
    # Consecutive steps the new side must hold past the deadzone before a swap counts,
    # so a brief twitch across the midline doesn't register as a real stride.
    STRIDE_MIN_DWELL = 5

    def __init__(
        self,
        render_mode=None,
        render_fps=60,
        url: str = None,
        timeout: int = 10,
        enable_sound: bool = False,
        auto_start_server: bool = True,
        server_path: Optional[str] = None,
        server_port: Optional[int] = None,
        ticks_per_step: int = 1,
        speed_reward_coef: float = 0.0,
        posy_reward_coef: float = 0.0,
        stride_reward_coef: float = 1.0,
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
            server_port: Port for game server (falls back to QWOP_SERVER_PORT env
                var, then 3100). Each parallel instance needs its own port.
            ticks_per_step: Number of physics ticks (each PHYSICS_DT simulated seconds)
                to run per env.step()/reset() call. Physics is lock-stepped to the
                agent - it only advances when explicitly driven (see
                window.__qwopAdvance in ws.js), not on a real-time timer - so this is an
                exact, deterministic action-repeat/frame-skip rather than depending on
                Selenium round-trip timing. Raise it to cover more simulated time per RL
                step (and, since ticks run synchronously in one script call, to train
                faster).
            posy_reward_coef: Weight on a standing-height *bonus*, added on top of the
                forward-velocity reward: coef * max(0, -posy). The engine's world y-axis
                points *down* (gravity is (0, +10), posy is the runner's world-center y),
                so upright posture is negative posy (measured empirically: ~-0.18
                standing, ~-0.03 settled in a crouch, ~+0.5 at a face-plant's terminal
                tick) - max(0, -posy) is positive while upright and floors at exactly 0
                once posy crosses into "on the ground" territory, rather than going
                negative. Deliberately a bonus, not a penalty: a per-step penalty for low
                posture creates a flow cost the agent can escape by ending the episode
                (falling) rather than recovering, since falling zeroes out all future
                reward instead of accumulating more of the penalty. A floored bonus never
                produces that incentive - the worst outcome is forfeiting the bonus, not
                paying anything. Keep this well below the scale of velx*abs(velx) (raw
                height_bonus tops out around 0.18 while standing still; velx^2 already
                reaches ~0.1-0.2 from a brief forward stumble alone) - too large and
                standing motionless becomes a risk-free reward source that beats actually
                moving, since moving lowers posture and forfeits the bonus. Set to 0 to
                disable.
            stride_reward_coef: Weight on the *only* forward-progress reward in this
                env: coef * velx, paid once, at the instant the "lead leg" (whichever
                thigh is more forward, from wrap_to_pi(leftthigh - rightthigh)) actually
                swaps sides - see _stride_bonus. There's no dense per-tick velocity
                reward anymore; all speed credit is gated behind completing a genuine
                stride, so a gait that never swaps (e.g. knee-sliding) earns nothing for
                its speed no matter how fast it's going. Swap detection uses a deadzone
                (STRIDE_SWAP_THRESHOLD) around stride≈0 plus a minimum dwell time on the
                new side (STRIDE_MIN_DWELL steps) before it counts, so briefly twitching
                across the midline doesn't register as a swap. The very first side the
                policy settles into after reset isn't rewarded (there's nothing to have
                swapped from yet). Set to 0 to disable (in which case forward speed is
                never rewarded at all - only posture would be).

                Being scaled by velx rather than flat also closes the "farm swaps in
                place" exploit that a flat per-swap bonus would allow: swapping with
                ~0 velx at that instant pays ~0, regardless of how fast the dwell timer
                lets swaps repeat - so farming requires actually moving. Set fairly
                large (default 10.0) since real swaps essentially never showed up in
                rollouts under the old flat-bonus scheme's more conservative values -
                worth erring toward "too attractive" until the agent actually discovers
                the behavior at all. The tradeoff: reward is now much sparser overall
                (near-zero except at swap instants, whereas before every tick paid out
                velx^2), which can make initial discovery slower since there's no dense
                per-tick signal teaching "moving forward is good" - watch for whether
                training stalls with EpAvgVelx never leaving ~0, which would suggest the
                agent isn't finding its way to a first swap often enough to learn from.

        Environment variables:
            QWOP_URL: Override server URL
            QWOP_SERVER_PATH: Path to game server
            QWOP_HEADLESS: Run headless (true/false)
            QWOP_TIMEOUT: Page load timeout in seconds
            QWOP_SERVER_PORT: Default port for game server
        """

        super().__init__(render_mode, render_fps)

        self.ticks_per_step = ticks_per_step
        self.posy_reward_coef = posy_reward_coef
        self.stride_reward_coef = stride_reward_coef
        self.speed_reward_coef = speed_reward_coef

        # Lead-leg swap tracking (see _stride_bonus / stride_reward_coef docstring).
        # _lead_side is 0 (undetermined) until the policy first settles clearly to one
        # side; _pending_side/_pending_ticks track a candidate swap-in-progress before
        # it's held long enough (STRIDE_MIN_DWELL) to count.
        self._lead_side = 0
        self._pending_side = 0
        self._pending_ticks = 0
        self._episode_stride_count = 0

        if server_port is None:
            server_port = self._resolve_server_port(3100)

        self.url = url or os.getenv("QWOP_URL", f"http://localhost:{server_port}")
        self.headless = self.render_mode not in ("human", "realtime")
        self.timeout = timeout or int(os.getenv("QWOP_TIMEOUT", "10"))
        self.enable_sound = enable_sound

        self.server_manager = self._start_game_server(server_path, server_port, self.timeout) if auto_start_server else None

        self.canvas = None

        # For velocity tracking
        self.last_joint_angles = {}
        self.joint_names = ["head", "torso", "leftarm", "rightarm", "leftthigh", "rightthigh", "leftcalf", "rightcalf"]

        self._connect_to_game()

        # Setup signal handlers for graceful shutdown
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGTSTP]:
            signal.signal(sig, self._signal_handler)
        atexit.register(self.close)

    def _connect_to_game(self):
        """Connect to QWOP game server."""
        try:
            logger.info(f"Connecting to QWOP at {self.url}")

            self.driver = self._launch_browser(f"{self.url}?webgl=true", self.headless, self.enable_sound)
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
            logger.debug("Game canvas loaded, pumping physics loop until ready...")

            # Get game canvas element (modern API). State/terminal are read directly from
            # window.state / window.terminal in get_state() rather than the #state/#terminal
            # DOM elements, since QWOP.min.js sets both directly on window as well.
            self.canvas = self.driver.find_element(By.ID, "window1")

            # Physics is lock-stepped to the agent (see window.__qwopAdvance in ws.js) - it
            # doesn't run on its own timer, so nothing happens until we explicitly drive it.
            # Element presence above isn't enough on its own either way: until the update
            # loop has run at least once, `window.state` is still the browser's automatic
            # named-element binding for <div id="state"> (elements with an id are
            # auto-exposed as window.<id>), not the real data object the game later assigns
            # there. Use the plain one-shot pump here, not __qwopAdvance: physics can't
            # actually tick yet (no click-to-start has happened), and __qwopAdvance would
            # burn its whole retry budget waiting for a tick that can't happen.
            deadline = time.time() + self.timeout + 5
            while True:
                self.driver.execute_script("window.__qwopPumpFrame();")
                if self.driver.execute_script(
                    "return !!(window.state && typeof window.state.posx !== 'undefined');"
                ):
                    break
                if time.time() >= deadline:
                    raise TimeoutException("Game physics loop did not start in time")

            logger.debug("Game initialization complete")
            logger.info("QWOP game loaded successfully")
            # Initialize state trackers - will be populated on first step
            self.last_tick = self.driver.execute_script("return window.physicsTick || 0;")
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

    # Combined read of window.state / window.terminal / window.physicsTick in one round
    # trip. Optionally applies an action (window.__qwopApplyAction, JS-dispatched key
    # events - no separate ActionChains round trip) and/or advances physics by `ticks`
    # first, still all in the same round trip - so step()/reset() each need only one
    # execute_script call total instead of a key-press call, an advance call, and three
    # separate reads.
    _READ_RAW_STATE_JS = "return {state: window.state || null, terminal: !!window.terminal, tick: window.physicsTick || 0};"
    _ADVANCE_AND_READ_JS = "window.__qwopAdvance(arguments[0]);" + _READ_RAW_STATE_JS
    _STEP_JS = "window.__qwopApplyAction(arguments[0]); window.__qwopAdvance(arguments[1]);" + _READ_RAW_STATE_JS

    def _compute_state(self, raw: Dict[str, Any]) -> Dict[str, float]:
        """Turn a raw {state, terminal, tick} read from the browser into the full state
        dict, with diffx/velx/accelx derived from window.physicsTick (incremented once per
        simulated PHYSICS_DT-second world.step()) rather than wall-clock time. Since
        physics is lock-stepped to ticks_per_step (see step()/reset()), this is exact
        regardless of Selenium round-trip latency, GC pauses, or CPU contention from other
        parallel workers.
        """
        state = raw.get("state")

        # If state isn't available yet, return a default state
        if not state:
            logger.debug("window.state not yet available, using default state")
            return {
                "posx": 0,
                "quads": [0, 0, 0, 0],
                "terminal": False,
                "diffx": 0,
                "velx": 0,
                "accelx": 0,
            }

        state["terminal"] = bool(raw.get("terminal"))

        tick = raw.get("tick", 0)
        sim_dt = (tick - self.last_tick) * self.PHYSICS_DT
        self.last_tick = tick

        state["diffx"] = state.get("posx", 0) - self.old_posx
        state["velx"] = state["diffx"] / sim_dt if sim_dt > 0 else 0.0
        state["accelx"] = (state["velx"] - self.old_velx) / sim_dt if sim_dt > 0 else 0.0

        self.old_posx = state.get("posx", 0)
        self.old_velx = state["velx"]

        return state

    def get_state(self) -> Dict[str, float]:
        """Get current game state, read directly from the window.state / window.terminal
        globals that QWOP.min.js sets on every physics tick, without advancing physics."""
        try:
            return self._compute_state(self.driver.execute_script(self._READ_RAW_STATE_JS))
        except Exception as e:
            logger.error(f"Error getting state: {e}")
            raise RuntimeError(f"Failed to get game state: {e}")

    def _advance_and_get_state(self, ticks: int, action: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Advance physics by `ticks` and read the resulting state, optionally applying an
        action first - all in a single round trip rather than a separate action-application
        call, a separate __qwopAdvance call, and three separate reads."""
        try:
            if action is not None:
                raw = self.driver.execute_script(self._STEP_JS, np.asarray(action).tolist(), ticks)
            else:
                raw = self.driver.execute_script(self._ADVANCE_AND_READ_JS, ticks)
            return self._compute_state(raw)
        except Exception as e:
            logger.error(f"Error advancing/getting game state: {e}")
            raise RuntimeError(f"Failed to advance game state: {e}")

    def _stride_bonus(self, state: Dict[str, float], velx: float) -> float:
        """Pay stride_reward_coef * velx the moment the lead leg (whichever thigh is
        more forward) actually swaps sides - see stride_reward_coef docstring in
        __init__. Stateful: call exactly once per real step (compute_reward's only
        caller), since it advances the dwell-time tracking each time it runs."""
        stride = state.get("leftthigh", 0.0) - state.get("rightthigh", 0.0)
        stride = (stride + math.pi) % (2 * math.pi) - math.pi  # wrap to (-pi, pi]

        if stride > self.STRIDE_SWAP_THRESHOLD:
            side = 1
        elif stride < -self.STRIDE_SWAP_THRESHOLD:
            side = -1
        else:
            side = 0  # in the deadzone - too ambiguous to count either way

        if side == 0 or side == self._lead_side:
            # Neutral zone, or still on the current lead side - nothing swap-worthy is
            # in progress.
            self._pending_side = 0
            self._pending_ticks = 0
            return 0.0

        if side == self._pending_side:
            self._pending_ticks += 1
        else:
            self._pending_side = side
            self._pending_ticks = 1

        if self._pending_ticks < self.STRIDE_MIN_DWELL:
            return 0.0

        # The new side has held for long enough to count as a real swap. The very
        # first side settled into after reset (_lead_side still 0) establishes the
        # baseline rather than counting/rewarding a "swap" from nothing.
        is_real_swap = self._lead_side != 0
        if is_real_swap:
            self._episode_stride_count += 1
        bonus = self.stride_reward_coef * velx if is_real_swap else 0.0
        self._lead_side = side
        self._pending_side, self._pending_ticks = 0, 0
        return bonus

    def compute_reward(self, state: Dict[str, float] = None) -> float:
        """Compute shaped reward to guide exploration and learning."""
        state = state or self.get_state()

        velx = state.get("velx", 0.0)
        # No dense per-tick velocity reward - forward progress only pays out at the
        # moment a stride actually completes (see _stride_bonus/stride_reward_coef),
        # scaled by velx at that instant. This ties earning any speed credit at all to
        # genuinely alternating legs: gaits that never swap (e.g. knee-sliding) get
        # zero reward for their speed, no matter how fast they're moving, since there's
        # no swap event to pay it out on. It also closes off farming swaps in place for
        # free reward - a swap with ~0 velx at that instant pays ~0 regardless of how
        # fast the dwell timer lets swaps repeat. The height term is a floored bonus
        # (see posy_reward_coef docstring above), not a penalty, and stays dense every
        # tick - posture and forward-progress credit are separate concerns.
        height_bonus = max(0.0, -state.get("posy", 0.0))
        speed_bonus = velx * abs(velx)
        reward = self.speed_reward_coef * speed_bonus + self.posy_reward_coef * height_bonus + self._stride_bonus(state, velx)

        return float(reward)

    @staticmethod
    def _decode_action(action: int) -> list:
        """Decode a Discrete(16) action index into the [q, w, o, p] key-state array the
        game expects: bit i of the index is whether key i is held down. Modeling the
        action space as one Categorical over all 16 key combinations (rather than 4
        independent Bernoullis, one per key) lets the policy represent correlations
        between keys - e.g. always pressing two specific keys together - which
        independent per-key probabilities structurally can't express."""
        action = int(action)
        return [(action >> i) & 1 for i in range(4)]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step: apply the action, advance physics by exactly
        ticks_per_step (see PHYSICS_DT / __qwopAdvance), then observe the result - all in
        one round trip (see _advance_and_get_state)."""
        state = self._advance_and_get_state(self.ticks_per_step, action=self._decode_action(action))
        obs = self.obs(state)
        reward = self.compute_reward(state)

        # No-op unless actually rendering (render_mode "human" or "realtime"); see
        # BaseQWOPEnv._throttle. Passing the actual simulated time this step covered makes
        # "realtime" mode's playback speed correct regardless of ticks_per_step.
        self._throttle(self.ticks_per_step * self.PHYSICS_DT)

        terminated = state.get("terminal", False)
        truncated = False  # No timeout in real environment

        # "posx" is also exposed as a plain top-level scalar (in addition to the full
        # "state" dict) because gymnasium's vector-env info merging recursively decomposes
        # nested dicts field-by-field - a scalar is far simpler to read back out for a
        # specific env index than re-navigating that recursive structure.
        return obs, reward, terminated, truncated, {
            "state": state,
            "posx": float(state.get("posx", 0)),
            "velx": float(state.get("velx", 0)),
            "stride_count": self._episode_stride_count,
        }

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset game to initial state."""
        try:
            self._lead_side = 0
            self._pending_side = 0
            self._pending_ticks = 0
            self._episode_stride_count = 0


            self.canvas.click()
            ActionChains(self.driver) \
                .key_up("q") \
                .key_up("w") \
                .key_up("o") \
                .key_up("p") \
                .key_down("r") \
                .key_up("r") \
                .perform()

            self.old_posx = 0.0
            self.old_velx = 0.0
            state = self._advance_and_get_state(self.ticks_per_step)
            self.last_frame_time = time.time()

            return self.obs(state), {}

        except Exception as e:
            logger.error(f"Error during reset: {e}")
            raise RuntimeError(f"Failed to reset game: {e}")

    def render(self, mode='human', rank=None, state=None):
        """Print human-readable state. Pass `state` (e.g. the info['state'] a just-completed
        step() already returned) to avoid an extra round trip re-fetching it."""
        state = state if state is not None else self.get_state()
        rank_str = f"[Agent {rank}] " if rank is not None else ""
        print(
            f"\r{rank_str}[QWOP] Pos: {state.get('posx', 0):6.2f}  "
            f"Height: {state.get('posy', 0):5.2f}  "
            f"Vel: {state.get('velx', 0):6.2f}  "
            f"Terminal: {state.get('terminal', False)}",
            end="", flush=True
        )

    def _signal_handler(self, sig, frame):
        """Handle system signals."""
        logger.info(f"Received signal {sig}, cleaning up...")
        self.close()
