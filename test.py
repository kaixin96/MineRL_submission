"""
Testing Script
"""
import os

import torch

from env_highlevel import meta_env_creator
from minerl_model import TreechopModel, MetaController

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

############ Load treechop model ############
treechop_state_dict = torch.load(
    f'./train/treechop/run_1/itr_149.pkl')['agent_state_dict']
treechop_model = TreechopModel(
    image_shape=(3, 64, 64),
    output_size=4,
    fc_sizes=64,
    use_maxpool=False,
    channels=[32, 64, 64, 64],
    kernel_sizes=[8, 4, 3, 4],
    strides=[4, 2, 1, 1],
    paddings=[0, 0, 0, 0],
)
treechop_model.load_state_dict(treechop_state_dict)
treechop_model.eval()

############ Make the environment ############
env_id = 'MineRLObtainDiamond-v0'
env = meta_env_creator(env_id=env_id, treechop_model=treechop_model)

############ Load meta-controller ############
meta_state_dict = torch.load(f'./train/meta/run_1/itr_49.pkl')['agent_state_dict']
# meta_state_dict = torch.load(f'./train/meta_finetune/run_1/itr_49.pkl')['agent_state_dict']
meta_model = MetaController(
    inventory_shape=env.spaces.observation.shape.inventory,
    output_size=env.action_space.n,
    hidden_sizes=[64, 64],
    hidden_nonlinearity=torch.nn.Tanh)
meta_model.load_state_dict(meta_state_dict)
meta_model.cuda()
meta_model.eval()

############ Test ############
n_episodes = 100
total_reward = 0
for i in range(n_episodes):
    prev_obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = meta_model.get_action(prev_obs.inventory)
        obs, reward, done, _ = env.step(action)
        prev_obs = obs
        episode_reward += reward
    print(f"Episode {i}: reward {episode_reward}.")
    total_reward += episode_reward
print(f'Average reward: {total_reward / n_episodes}')
