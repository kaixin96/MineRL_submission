"""
Customized Environment
"""
from copy import deepcopy

import numpy as np
import gym
from minerl.env.spaces import Box, Dict
from rlpyt.envs.base import EnvSpaces, EnvStep
from rlpyt.envs.gym import build_info_tuples, info_to_nt
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper


class MoveAxisWrapper(gym.ObservationWrapper):
    """Move axes of observation ndarrays."""
    def __init__(self, env, source, destination):
        super().__init__(env)

        self.source = source
        self.destination = destination

        space_dict = deepcopy(self.observation_space.spaces)
        low = self.moveaxis(space_dict['pov'].low)
        high = self.moveaxis(space_dict['pov'].high)
        space_dict['pov'] = Box(low=low, high=high, dtype=space_dict['pov'].dtype)
        self.observation_space = Dict(space_dict)

    def moveaxis(self, frame):
        return np.moveaxis(frame, self.source, self.destination)

    def observation(self, observation):
        observation['pov'] = self.moveaxis(observation['pov']).copy()
        return observation


class InventoryWrapper(gym.ObservationWrapper):
    """
    Remove unnecessary items in inventory
    """
    def __init__(self, env):
        super().__init__(env)

        space_dict = deepcopy(self.observation_space.spaces)
        self.keep_items = ['planks', 'wooden_pickaxe', 'cobblestone', 'furnace', 'stone_pickaxe']
        space_dict['inventory'] = Box(low=0, high=2304, shape=(len(self.keep_items),))
        space_dict.pop('equipped_items')
        self.observation_space = Dict(space_dict)

    def observation(self, observation):
        inventory = np.array([observation['inventory'][item] for item in self.keep_items])
        observation['inventory'] = inventory
        observation.pop('equipped_items')
        return observation


class RLPytWrapper(gym.Wrapper):
    """
    Wrap the gym environment with namedtuple
    """
    def __init__(self, env,
                 act_null_value=0, obs_null_value=0, force_float32=True):
        super().__init__(env)
        o = self.env.reset()
        o, r, d, info = self.env.dummy_action()
        self.action_space = GymSpaceWrapper(
            space=self.env.action_space,
            name="act",
            null_value=act_null_value,
            force_float32=force_float32,
        )
        self.observation_space = GymSpaceWrapper(
            space=self.env.observation_space,
            name="obs",
            null_value=obs_null_value,
            force_float32=force_float32,
        )
        build_info_tuples(info)

    def step(self, action):
        a = self.action_space.revert(action)
        o, r, d, info = self.env.step(a)
        obs = self.observation_space.convert(o)
        info = info_to_nt(info)
        return EnvStep(obs, r, d, info)

    def reset(self):
        return self.observation_space.convert(self.env.reset())

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )
