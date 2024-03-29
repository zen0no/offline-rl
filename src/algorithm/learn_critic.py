import os
import numpy as np
import torch
import gym
import d4rl
import utils
import bellman_wasserstein as Distance
import wandb

from dataclasses import dataclass, field

import pyrallis

@dataclass
class TrainConfig:
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


    #Wandb
    project: str = 'Bellman-Wasserstein-distance(ICLR)'
    name: str = 'project_name'
    group: str = 'Learn-random-BWD'

    def __post_init__(self):
        self.name = f"{self.task}_{self.env}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.checkpoint_path is not None:
            self.checkpoint_path = os.path.join(self.checkpoint_path, self.env)



    
def make_env(cfg: TrainConfig):
    env = gym.make(cfg.env)
    env.seed(cfg.seed)

    return env


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def wandb_init(cfg):
    wandb.init(
        project=cfg.project,
        name=cfg.env
    )


def run(cfg: TrainConfig): 

    env = make_env(cfg)
    set_seed(cfg.seed)



    models_path = os.path.join(
        cfg.checkpoint_path,
        cfg.env
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
        "hidden_dim": cfg.hidden_dim,
        "batch_size": cfg.batch_size,
        "max_action": max_action,
        "max_steps": cfg.timesteps,
        "discount": cfg.discount,
        "tau": cfg.tau,
        "alpha": cfg.alpha,
        "device": cfg.device
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
    

    for t in range(int(cfg.timesteps) + 1):
        log_dict = method.update_critic(replay_buffer)
        wandb.log(log_dict, step=t)
        log_dict = method.update_value(replay_buffer)
        wandb.log(log_dict, step=t)
        torch.cuda.empty_cache()

        if t in [1000, 10000, 20000, 50000, 100000, 500000, 1000000]:
            method.save_checkpoints(cfg.checkpoint_path, t)
            checkpoint_count += 1

    method.save_checkpoints(cfg.checkpoint_path)
    torch.cuda.empty_cache()




if __name__ == "__main__":
    cfg = pyrallis.parse(TrainConfig)

    wandb_init(cfg)
    run(cfg=cfg)    
    
        

