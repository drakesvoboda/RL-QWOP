"""QWOP Gymnasium environment package."""
from gymnasium.envs.registration import register

# Real environments (requires Selenium + game server)
register(
    id='qwop-v0',
    entry_point='gym_qwop.envs.qwop_env:QWOPEnv',
)