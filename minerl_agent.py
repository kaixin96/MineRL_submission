"""
MineRL agent
"""
from rlpyt.agents.pg.categorical import CategoricalPgAgent

from minerl_model import TreechopModel, MetaController


class MinerlTreeChopMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        model_kwargs = {
            'image_shape': env_spaces.observation.shape.pov,
            'output_size': env_spaces.action.n,
        }
        return model_kwargs

class MinerlMetaMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        model_kwargs = {
            'inventory_shape': env_spaces.observation.shape.inventory,
            'output_size': env_spaces.action.n,
        }
        return model_kwargs


class MinerlTreeChopAgent(MinerlTreeChopMixin, CategoricalPgAgent):

    def __init__(self, ModelCls=TreechopModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class MinerlMetaAgent(MinerlMetaMixin, CategoricalPgAgent):

    def __init__(self, ModelCls=MetaController, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
