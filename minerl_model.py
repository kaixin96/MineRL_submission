"""
Torch Model for MineRL environment

Megre the features from image and discrete inventory vector
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.mlp import MlpModel


class TreechopModel(torch.nn.Module):
    """MineRL Treechop Model"""
    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=512,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
        ):
        super().__init__()
        self.conv = Conv2dHeadModel(
            image_shape=image_shape,
            channels=channels or [32, 64, 64],
            kernel_sizes=kernel_sizes or [8, 4, 3],
            strides=strides or [4, 2, 1],
            paddings=paddings or [0, 0, 0],
            use_maxpool=use_maxpool,
            hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
        )
        self.pi = torch.nn.Linear(self.conv.output_size, output_size)
        self.value = torch.nn.Linear(self.conv.output_size, 1)

    def forward(self, observation, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        image = observation.pov
        img = image.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        fc_out = self.conv(img.view(T * B, *img_shape))

        pi = F.softmax(self.pi(fc_out), dim=-1)
        v = self.value(fc_out).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v

    def get_action(self, pov_obs):
        with torch.no_grad():
            device = self.conv.conv.conv[0].weight.device
            obs = torch.from_numpy(pov_obs).to(device).float()[None, ...]
            obs = obs.mul_(1. / 255)
            pi = F.softmax(self.pi(self.conv(obs)), dim=-1)
            action = Categorical(probs=pi).sample()
        return action.item()

class MetaController(torch.nn.Module):
    """
    MineRL meta-contorller to switch between options

    options:
        -- chop trees
        -- mine down and mine around
        -- craft wooden pickaxe
        -- craft stone pickaxe and furnace
    """
    def __init__(
            self,
            inventory_shape,
            output_size,
            hidden_sizes=None,  # None for default (see below).
            hidden_nonlinearity=torch.nn.Tanh,  # Module form.
            init_path=None,
        ):
        super().__init__()
        self._obs_ndim = len(inventory_shape)
        input_size = int(np.prod(inventory_shape))
        hidden_sizes = hidden_sizes or [64, 64]
        self.mlp = MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            nonlinearity=hidden_nonlinearity,
        )
        self.pi = torch.nn.Linear(self.mlp.output_size, output_size)
        self.value = torch.nn.Linear(self.mlp.output_size, 1)
        if init_path:
            self.load_state_dict(torch.load(init_path, map_location='cpu'))

    def forward(self, observation, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        inv = observation.inventory
        inv = inv.type(torch.float)

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(inv, self._obs_ndim)

        obs_flat = inv.view(T * B, -1)
        fc_out = self.mlp(obs_flat)

        pi = F.softmax(self.pi(fc_out), dim=-1)
        v = self.value(fc_out).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v

    def get_action(self, inv_obs):
        with torch.no_grad():
            device = self.pi.weight.device
            obs = torch.from_numpy(inv_obs).to(device).float()[None, ...]
            pi = F.softmax(self.pi(self.mlp(obs)), dim=-1)
            action = Categorical(probs=pi).sample()
        return action.item()
