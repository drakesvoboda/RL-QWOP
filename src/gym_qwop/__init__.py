"""QWOP Gymnasium environment package."""
from gymnasium.envs.registration import register

# Mock environments (lightweight, no browser overhead)
register(
    id='qwop-mock-v0',
    entry_point='gym_qwop.envs.mock_qwop:MockQWOPEnv',
)

# Real environments (requires Selenium + game server)
register(
    id='qwop-v0',
    entry_point='gym_qwop.envs.qwop_real:RealQWOPEnv',
)

# Legacy aliases
register(
    id='frame-qwop-v0',
    entry_point='gym_qwop.envs.qwop_real:RealQWOPEnv',
)

register(
    id='multi-frame-qwop-v0',
    entry_point='gym_qwop.envs.qwop_real:RealQWOPEnv',
)

# Mock aliases
register(
    id='frame-qwop-mock-v0',
    entry_point='gym_qwop.envs.mock_qwop:MockQWOPEnv',
)

register(
    id='multi-frame-qwop-mock-v0',
    entry_point='gym_qwop.envs.mock_qwop:MockQWOPEnv',
)