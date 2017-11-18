from .boolean_multiplexer import BooleanMultiplexer

from gym.envs.registration import register

name = "boolean-multiplexer"
max_episode_steps = 1

register(
    id='{}-3bit-v0'.format(name),
    entry_point='gym_multiplexer:BooleanMultiplexer',
    max_episode_steps=max_episode_steps,
    kwargs={'control_bits': 1}
)

register(
    id='{}-6bit-v0'.format(name),
    entry_point='gym_multiplexer:BooleanMultiplexer',
    max_episode_steps=max_episode_steps,
    kwargs={'control_bits': 2}
)

register(
    id='{}-11bit-v0'.format(name),
    entry_point='gym_multiplexer:BooleanMultiplexer',
    max_episode_steps=max_episode_steps,
    kwargs={'control_bits': 3}
)