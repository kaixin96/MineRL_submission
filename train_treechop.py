"""
Training Script
"""
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.algos.pg.ppo import PPO
from rlpyt.runners.minibatch_rl import MinibatchRl

from env_lowlevel import treechop_env_creator
from minerl_agent import MinerlTreeChopAgent
from minerl_logging import logger_context

env_id = 'MineRLObtainDiamondDense-v0'

agent = MinerlTreeChopAgent(model_kwargs={
    'fc_sizes': 64,
    'use_maxpool': False,
    'channels': [32, 64, 64, 64],
    'kernel_sizes': [8, 4, 3, 4],
    'strides': [4, 2, 1, 1],
    'paddings': [0, 0, 0, 0],
})

sampler = CpuSampler(
    EnvCls=treechop_env_creator,
    env_kwargs=dict(env_id=env_id),
    batch_T=2048,  # One time-step per sampler iteration.
    batch_B=2,  # One environment (i.e. sampler Batch dimension).
    max_decorrelation_steps=0,
)

algo = PPO(
    discount=0.99,
    learning_rate=2.5e-4,
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
    agent=agent,
    sampler=sampler,
    n_steps=150,
    log_interval_steps=1,
    affinity=dict(cuda_idx=None,
                  workers_cpus=[i for i in range(12)],
                  set_affinity=False),
)

config = dict(env_id=env_id)
name = "ppo_" + env_id
log_dir = "./train/treechop"
with logger_context(log_dir=log_dir, run_ID=1, name=name, log_params=config,
                    snapshot_mode='gap', snapshot_gap=10):
    runner.train()
