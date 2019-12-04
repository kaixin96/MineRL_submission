"""
TreeChop Env
"""
import random
from copy import deepcopy

import numpy as np
import gym
from minerl.env.spaces import Discrete

from envs import MoveAxisWrapper, InventoryWrapper, RLPytWrapper


class MacroActionTreechopWrapperBaseline(gym.ActionWrapper):
    """
    Wrap actions into discrete macro-actions

    > Turn left
    > Turn right
    > Jump
    > Attack front
    """
    def __init__(self, env, to_plank=False):
        super().__init__(env)

        self.noop = self.action_space.no_op()

        ###### Compose macro-actions ######
        self._actions = []
        self._action_names = []
        def add_action(name, action):
            self._action_names.append(name)
            self._actions.append(action)

        ### Camera ###
        def camera(pitch, yaw):
            macro_action = []
            action = deepcopy(self.noop)
            action['attack'] = 1
            action['camera'] = np.array([pitch, yaw])
            macro_action.append(action)
            return macro_action
        add_action('turn_left', camera(0, -40))
        add_action('turn_right', camera(0, 40))

        ### Jump ###
        macro_action = []
        action = deepcopy(self.noop)
        action['attack'] = 1
        action['jump'] = 1
        action['forward'] = 1
        macro_action.extend([action for _ in range(5)])
        add_action('jump', macro_action)

        ### Attack ###
        def attack(n_frames, delta_pitch, to_plank=False):
            """
            Attack blocks

            Args:
                n_frames: number of consecutive frames of attacking
                delta_pitch: change of pitch value before attacking
            """
            macro_action = []
            if delta_pitch != 0:
                action = deepcopy(self.noop)
                action['camera'] = np.array([delta_pitch, 0])
                macro_action.append(action)
            action = deepcopy(self.noop)
            action['attack'] = 1
            macro_action.extend([action for _ in range(n_frames)])
            if to_plank and 'craft' in self.noop:
                macro_action[-1]['craft'] = 3
            if delta_pitch != 0:
                action = deepcopy(self.noop)
                action['camera'] = np.array([-delta_pitch, 0])
                macro_action.append(action)
            return macro_action
        add_action('attack_forward', attack(n_frames=35, delta_pitch=0, to_plank=to_plank))

        ###### Register action space ######
        self.action_space = Discrete(len(self._actions))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = self._actions[action]
        return original_space_action

    def step(self, action):
        macro_action = self.action(action)
        total_reward = 0.0
        for action in macro_action:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        action = deepcopy(self.noop)
        action['camera'] = np.array([30, 0])
        obs, _, _, _ = self.env.step(action)
        return obs

    def nearby_craft(self, item_idx, equip=True, make_tools=False):
        """
        Return a nearby craft action
        """
        macro_action = []
        macro_action.extend([self.noop for _ in range(7)])
        # carft table / stick
        if make_tools:
            tools = (2, 4) if random.random() > 0.5 else (4, 2)
            for tool in tools:
                action = deepcopy(self.noop)
                action['craft'] = tool
                macro_action.extend([action for _ in range(1)])
        macro_action.extend([self.noop for _ in range(3)])
        # place
        action = deepcopy(self.noop)
        action['place'] = 4
        macro_action.extend([action for _ in range(1)])
        macro_action.extend([self.noop for _ in range(3)])
        # nearby craft
        action = deepcopy(self.noop)
        action['nearbyCraft'] = item_idx
        macro_action.append(action)
        macro_action.extend([self.noop for _ in range(3)])
        # attack
        action = deepcopy(self.noop)
        action['attack'] = 1
        macro_action.extend([action for _ in range(85)])
        # equip
        if equip:
            action = deepcopy(self.noop)
            action['equip'] = 3
            macro_action.append(action)
            action = deepcopy(self.noop)
            action['equip'] = 5
            macro_action.append(action)
        # collect
        action = deepcopy(self.noop)
        action['forward'] = 1
        action['jump'] = 1
        macro_action.extend([action for _ in range(7)])
        # noop
        macro_action.extend([self.noop for _ in range(5)])
        return macro_action

    def wooden_pickaxe(self):
        """
        Craft a wooden pickaxe
        """
        macro_action = self.nearby_craft(2, make_tools=True)
        total_reward = 0.0
        for action in macro_action:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def stone_pickaxe(self):
        """
        Craft a stone pickaxe
        """
        macro_action = self.nearby_craft(4)
        total_reward = 0.0
        for action in macro_action:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def furnace(self):
        """
        Craft a furnace
        """
        macro_action = self.nearby_craft(7, equip=False)
        total_reward = 0.0
        for action in macro_action:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def minedown(self, n_steps=5, attack_down_frames=200):
        """
        Mine down and then mine around for certain steps
        """
        macro_action = []
        # Look down
        action = deepcopy(self.noop)
        action['camera'] = np.array([90, 0])
        macro_action.append(action)
        # Attack down
        for i in range(attack_down_frames):
            action = deepcopy(self.noop)
            action['attack'] = 1
            if (i + 1) % 40 == 0:
                action['forward'] = 1
            macro_action.append(action)
        # Look up
        action = deepcopy(self.noop)
        action['camera'] = np.array([-60, 0])
        macro_action.append(action)
        # Mine round
        for _ in range(n_steps):
            action = deepcopy(self.noop)
            action['attack'] = 1
            action['camera'] = np.array([0, 0.2])
            macro_action.extend([action for _ in range(50)])

        # Perform the macro action
        total_reward = 0.0
        for action in macro_action:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def dummy_action(self):
        return self.env.step(self.noop)

class DropInventoryWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        space_dict = deepcopy(self.observation_space.spaces)
        space_dict.pop('inventory')
        space_dict.pop('equipped_items')
        self.observation_space = gym.spaces.Dict(space_dict)

    def observation(self, observation):
        observation.pop('inventory')
        observation.pop('equipped_items')
        return observation

def treechop_env_creator(env_id):
    env = gym.make(env_id)
    mcenv = MoveAxisWrapper(env, source=-1, destination=0)
    mcenv = DropInventoryWrapper(mcenv)
    mcenv = MacroActionTreechopWrapperBaseline(mcenv, to_plank=False)
    mcenv = RLPytWrapper(mcenv)
    return mcenv
