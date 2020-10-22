from gym.envs.registration import register

register(
    id='qwop-v0',
    entry_point='gym_qwop.envs:QWOPEnv',
)

register(
    id='frame-qwop-v0',
    entry_point='gym_qwop.envs:FrameQWOPEnv',
)

register(
    id='multi-frame-qwop-v0',
    entry_point='gym_qwop.envs:MultiFrameQWOPEnv',
)