"""
Train meta-controller using Behavioral Cloning
"""
import os
import itertools

import minerl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from minerl_model import MetaController


#####################################################################
###### Process Demonstration Data (convert actions to options) ######
#####################################################################

keep_items = ['planks', 'wooden_pickaxe', 'cobblestone', 'furnace', 'stone_pickaxe']

all_options = []
all_obs_inv = []
for env in ['MineRLObtainDiamond-v0', 'MineRLObtainIronPickaxe-v0']:
    data = minerl.data.make(env)
    loader = data.sarsd_iter(num_epochs=1, max_sequence_len=-1)
    for current_state, action, reward, next_state, done in loader:
        new_options = []
        obs_inv = []
        nbCraft = action['nearbyCraft']
        inventory = current_state['inventory']
        n_steps = len(nbCraft)

        wp_idx = np.where(nbCraft == 2)[0]
        sp_idx = np.where(nbCraft == 4)[0]
        fs_idx = np.where(nbCraft == 7)[0]
        idx = np.sort(np.hstack([wp_idx, sp_idx, fs_idx]))

        for i in range(max(int(round(idx.min(initial=n_steps) / 500)), 1)):
            obs_inv.append(
                np.array([inventory[item][i * 500] for item in keep_items]))
            new_options.append(0)  # treechop

        last_idx = idx.min() if idx.size else 0
        for i in idx:
            if i in wp_idx:
                new_options.append(2)
            elif i in sp_idx:
                new_options.append(3)
            elif i in fs_idx:
                new_options.append(4)
            else:
                raise ValueError(f'Wrong index {i}')
            obs_inv.append(
                np.array([inventory[item][i] for item in keep_items]))
            interval = i - last_idx
            for j in range(int(round(interval / 400))):
                obs_inv.append(
                    np.array([inventory[item][last_idx + 1 + j*400]
                              for item in keep_items]))
                new_options.append(1)
            last_idx = i

        if idx.max(initial=0):
            for i in range(int(round((n_steps - idx.max(initial=0)) / 400))):
                obs_inv.append(
                    np.array([inventory[item][idx.max(initial=0) + 1 + i*400]
                              for item in keep_items]))
                new_options.append(1)

        all_options.append(new_options)
        all_obs_inv.append(obs_inv)

#####################################################
###### Build a PyTorch Dataset for BC training ######
#####################################################

obs_data = np.vstack(list(itertools.chain.from_iterable(all_obs_inv))).copy()
action_data = np.vstack(list(itertools.chain.from_iterable(all_options))).copy()

class DemoDataset(Dataset):
    """Demostration Dataset"""
    def __init__(self, obs, action):
        assert len(obs) == len(action)
        self.obs = obs
        self.action = action

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.action[idx]

demodata = DemoDataset(obs_data, action_data)
loader = DataLoader(demodata, batch_size=1024, shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=True)

meta_model = MetaController(
    inventory_shape=(5,),
    output_size=5,
    hidden_sizes=[64, 64],
    hidden_nonlinearity=torch.nn.Tanh)
meta_model.cuda()
meta_model.train()

optimizer = torch.optim.SGD(meta_model.mlp.parameters(), lr=2.5e-4, weight_decay=0.05)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 10.0, 10.0, 10.0]).cuda())

for epoch in range(100):
    for obs, action in loader:
        obs = obs.cuda().float()
        action = action.cuda().long()[:, 0]
        pred = meta_model.pi(meta_model.mlp(obs))
        loss = criterion(pred, action)
        loss.backward()
        optimizer.step()

os.makedirs('./train/meta_BC', exist_ok=True)
torch.save(meta_model.state_dict(), './train/meta_BC/100itr.pth')
