from gym.envs.registration import register

from .boolean_multiplexer import BooleanMultiplexer
from .real_multiplexer import RealMultiplexer

bool_mpx_name = "boolean-multiplexer"
real_mpx_name = "real-multiplexer"
max_episode_steps = 1

# Length of a multiplexer is calculated
# using l = k + 2^k

register(
    id='{}-3bit-v0'.format(bool_mpx_name),
    entry_point='gym_multiplexer:BooleanMultiplexer',
    max_episode_steps=max_episode_steps,
    kwargs={'control_bits': 1}
)

register(
    id='{}-6bit-v0'.format(bool_mpx_name),
    entry_point='gym_multiplexer:BooleanMultiplexer',
    max_episode_steps=max_episode_steps,
    kwargs={'control_bits': 2}
)

register(
    id='{}-11bit-v0'.format(bool_mpx_name),
    entry_point='gym_multiplexer:BooleanMultiplexer',
    max_episode_steps=max_episode_steps,
    kwargs={'control_bits': 3}
)

register(
    id='{}-20bit-v0'.format(bool_mpx_name),
    entry_point='gym_multiplexer:BooleanMultiplexer',
    max_episode_steps=max_episode_steps,
    kwargs={'control_bits': 4}
)

register(
    id='{}-37bit-v0'.format(bool_mpx_name),
    entry_point='gym_multiplexer:BooleanMultiplexer',
    max_episode_steps=max_episode_steps,
    kwargs={'control_bits': 5}
)

register(
    id='{}-3bit-v0'.format(real_mpx_name),
    entry_point='gym_multiplexer:RealMultiplexer',
    max_episode_steps=max_episode_steps,
    kwargs={'control_bits': 1}
)

register(
    id='{}-6bit-v0'.format(real_mpx_name),
    entry_point='gym_multiplexer:RealMultiplexer',
    max_episode_steps=max_episode_steps,
    kwargs={'control_bits': 2}
)