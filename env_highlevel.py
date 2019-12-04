"""
Env for training meta-controller
"""
import gym
from minerl.env.spaces import Discrete

from envs import MoveAxisWrapper, InventoryWrapper, RLPytWrapper
from env_lowlevel import MacroActionTreechopWrapperBaseline

class OptionDiamondEnv(gym.Wrapper):
    """
    Environment for training the meta-controller
    """
    def __init__(self, env, treechop_model, treechop_steps=100, minedown_steps=5):
        super().__init__(env)
        self.treechop_model = treechop_model
        self.treechop_steps = treechop_steps
        self.minedown_steps = minedown_steps
        self.num_meta_action = 5
        self.action_space = Discrete(self.num_meta_action)

    def step(self, action):
        total_reward = 0.0
        if action == 0:
            # Perform treechop
            obs, rew, done, info = self.dummy_action()
            total_reward += rew
            cnt = 0
            while (not done) and (cnt < self.treechop_steps):
                macro_action = self.treechop_model.get_action(obs['pov'])
                obs, rew, done, info = self.env.step(macro_action)
                total_reward += rew
                cnt += 1
        elif action == 1:
            # Perform minedown and minearound
            obs, rew, done, info = self.env.minedown(self.minedown_steps)
            total_reward += rew
        elif action == 2:
            # Perform crafting wooden pickaxe
            obs, rew, done, info = self.env.wooden_pickaxe()
            total_reward += rew
        elif action == 3:
            # Perform crafting stone pickaxe
            obs, rew, done, info = self.env.stone_pickaxe()
            total_reward += rew
        elif action == 4:
            # Perform crafting furnace
            obs, rew, done, info = self.env.furnace()
            total_reward += rew
        else:
            raise ValueError(f"Wrong action value ({action}).")

        return obs, total_reward, done, info

    def dummy_action(self):
        return self.env.dummy_action()

def meta_env_creator(env_id, treechop_model):
    env = gym.make(env_id)
    mcenv = MoveAxisWrapper(env, source=-1, destination=0)
    mcenv = InventoryWrapper(mcenv)
    mcenv = MacroActionTreechopWrapperBaseline(mcenv, to_plank=True)
    mcenv = OptionDiamondEnv(mcenv, treechop_model)
    mcenv = RLPytWrapper(mcenv)
    return mcenv
