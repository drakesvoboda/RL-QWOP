"""QWOP Gymnasium environments with real and mock implementations."""
from gymnasium.envs.registration import register

# Mock QWOP (lightweight, no browser - great for testing/development)
register(
    id='qwop-mock-v0',
    entry_point='gym_qwop.envs.mock_qwop:MockQWOPEnv',
)

# Real QWOP (requires game server + Selenium + Firefox)
register(
    id='qwop-v0',
    entry_point='gym_qwop.envs.qwop_real:RealQWOPEnv',
)

# WebSocket QWOP (fast alternative - 10-20x speedup, requires game server with WebSocket)
register(
    id='qwop-websocket-v0',
    entry_point='gym_qwop.envs.qwop_websocket:WebSocketQWOPEnv',
)

# Aliases for backward compatibility
try:
    from gym_qwop.envs.qwop_real import RealQWOPEnv as QWOPEnv
    FrameQWOPEnv = QWOPEnv  # TODO: Implement frame observation
    MultiFrameQWOPEnv = QWOPEnv  # TODO: Implement multi-frame observation
except ImportError:
    pass

__all__ = ['register', 'QWOPEnv']