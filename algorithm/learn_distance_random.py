import os
import numpy as np
import torch
import gym
import d4rl
import d4rl.gym_mujoco
import utils
import bellman_wasserstein as Distance
import wandb
import uuid

import pyrallis

from config.base import WandbConfig
from dataclasses import dataclass, field

@dataclass
class RunConfig:
    device: str = 'cuda'
    device_num: int = 1

    task: str = 'Learn-random-BWD'
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
    W: float = 1.

    normalize: bool = True

@dataclass
class TrainConfig:
    run: RunConfig = field(default_factory=RunConfig)
    wandb_cfg: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        self.wandb_cfg.group = self.run.task
        self.wandb_cfg.name = f"{self.run.env}_{str(uuid.uuid4())[:8]}"

def make_env(cfg: RunConfig):
    env = gym.make(cfg.env)
    env.seed(cfg.seed)

    return env


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def wandb_init(cfg: WandbConfig):
    wandb.init(
        project=cfg.project,
        group=cfg.group,
        name=cfg.name
    )

def run(cfg: RunConfig):
    
    env = make_env(cfg)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": cfg.hidden_dim,
        "batch_size": cfg.batch_size,
        "max_action": max_action,
        "max_steps": cfg.timesteps,
        "discount": cfg.discount,
        "tau": cfg.tau,
        "alpha": cfg.alpha,
        "W": cfg.W
    }


    method = Distance.BellmanWasserstein(**kwargs)

    method.load(cfg.checkpoint_path)

    replay_buffer = utils.SarsaReplayBuffer(state_dim, action_dim, max_size=int(10000000))
    sarsa_dataset = utils.qlearning_dataset(env)
    replay_buffer.convert_D4RL(sarsa_dataset)
    if cfg.normalize:
        mean,std = replay_buffer.normalize_states() 
    else:
        mean,std = 0,1

    for t in range(int(cfg.timesteps)):
        log_dict = method.update_distance_cond_R(replay_buffer, advantage=True)
        wandb.log(log_dict, step=t)
        torch.cuda.empty_cache()
        
        if t % cfg.eval_freq==0:
            log_dict = method.estimate_distance_cond_R(replay_buffer,advantage=True)
            wandb.log(log_dict, step=t)
    
    torch.cuda.empty_cache()


if __name__ == "__main__":

    cfg: TrainConfig = pyrallis.parse(TrainConfig)

    wandb_init(cfg.wandb_cfg)
    run(cfg.run)  
      

        

