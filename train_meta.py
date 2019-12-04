"""
Train meta-controller in the environment
"""
import torch
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.algos.pg.ppo import PPO
from rlpyt.runners.minibatch_rl import MinibatchRl

from env_highlevel import meta_env_creator
from minerl_agent import MinerlMetaAgent
from minerl_model import TreechopModel
from minerl_logging import logger_context

env_id = 'MineRLObtainDiamond-v0'

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

init_path = './train/meta_BC/100itr.pth'
meta_agent = MinerlMetaAgent(
    model_kwargs={
        'hidden_sizes': [64, 64],
        'hidden_nonlinearity': torch.nn.Tanh,
        'init_path': init_path
    }
)

sampler = CpuSampler(
    EnvCls=meta_env_creator,
    env_kwargs=dict(env_id=env_id, treechop_model=treechop_model),
    batch_T=32,  # One time-step per sampler iteration.
    batch_B=2,  # One environment (i.e. sampler Batch dimension).
    max_decorrelation_steps=0,
)

algo = PPO(
    discount=0.99,
    learning_rate=1e-5,
    value_loss_coeff=0.5,
    entropy_loss_coeff=0.01,
    clip_grad_norm=0.5,
    initial_optim_state_dict=None,
    gae_lambda=0.95,
    minibatches=4,
    epochs=10,
    ratio_clip=0.1,
    linear_lr_schedule=False,
    normalize_advantage=False,
)

runner = MinibatchRl(
    algo=algo,
    agent=meta_agent,
    sampler=sampler,
    n_steps=50,
    log_interval_steps=1,
    affinity=dict(cuda_idx=None,
                  workers_cpus=[i for i in range(12)],
                  set_affinity=False),
)

config = dict(env_id=env_id)
name = "ppo_" + env_id
log_dir = "./train/meta_finetune" if init_path else "./train/meta"
with logger_context(log_dir=log_dir, run_ID=1, name=name, log_params=config,
                    snapshot_mode='gap', snapshot_gap=10):
    runner.train()
