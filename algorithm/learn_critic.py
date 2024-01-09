import os
import numpy as np
import torch
import gym
import d4rl
import utils
import bellman_wasserstein as Distance
import wandb

from config.base import WandbConfig
from dataclasses import dataclass, field

import pyrallis
import uuid

@dataclass
class RunConfig:
    device: str = 'cuda'
    device_num: int = 1

    task: str = 'SARSA-learn-critic'
    checkpoint_path: str = './checkpoint'
    env: str = 'halfcheetah-medium-expert-v2'
    seed: int = 0
    eval_freq: int = 10000
    timesteps: int = 1000000

    distance_freq: int = 10000    
    
    expl_noise: float = 0.1
    batch_size: int = 256
    hidden_dim: int = 1024
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2

    alpha: float = 1.

    normalize: bool = True

@dataclass
class TrainConfig:
    run: RunConfig = field(default_factory=RunConfig)
    wandb_cfg: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        self.wandb_cfg.name = f"{self.run.task}-{self.run.env}-{str(uuid.uuid4())[:8]}"
        if self.run.checkpoint_path is not None:
            self.run.checkpoint_path = os.path.join(self.run.checkpoint_path, self.wandb_cfg.name)




def make_env(cfg: RunConfig):
    env = gym.make(cfg.env)
    env.seed(cfg.seed)

    return env


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def wandb_init(cfg):
    wandb.init(
        project=cfg.project,
        name=cfg.name
    )


def run(cfg: RunConfig): 

    env = make_env(cfg)
    set_seed(cfg.seed)



    models_path = os.path.join(
        cfg.run.checkpoint_path,
        cfg.run.env
        )

    if not os.path.exists(models_path):
        print(f"Didn't find specified directory. Creating {models_path}")
        os.makedirs(models_path)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": cfg.run.hidden_dim,
        "batch_size": cfg.run.batch_size,
        "max_action": max_action,
        "max_steps": cfg.run.timesteps,
        "discount": cfg.run.discount,
        "tau": cfg.run.tau,
        "alpha": cfg.run.alpha
    }

    method = Distance.BellmanWasserstein(**kwargs)

    replay_buffer = utils.SarsaReplayBuffer(state_dim, action_dim, max_size=int(10000000))
    sarsa_dataset = utils.qlearning_dataset(env)
    replay_buffer.convert_D4RL(sarsa_dataset)

    if cfg.normalize:
        mean,std = replay_buffer.normalize_states() 
    else:
        mean,std = 0,1

    checkpoint_count = 0
    checkpoint_step = 2
    
    log_step = True

    for t in range(int(run.timesteps) + 1):
        log_dict = method.update_critic(replay_buffer)
        wandb.log(log_dict, step=t)
        log_dict = method.update_value(replay_buffer)
        wandb.log(log_dict, step=t)
        torch.cuda.empty_cache()

        if log_step and t == 10 ** ((checkpoint_step + 1) * checkpoint_count):
            method.save_checkpoints(cfg.checkpoint_path, t)
        elif not log_step and t == (checkpoint_count + 1) * checkpoint_step:
            method.save_checkpoints(cfg.checkpoint_path, t)

    method.save_checkpoints(cfg.checkpoint_path)




if __name__ == "__main__":
    cfg = pyrallis.parse(TrainConfig)

    wandb_init(cfg.wandb_config)
    run(cfg=cfg)    
    
    torch.cuda.empty_cache()
        

