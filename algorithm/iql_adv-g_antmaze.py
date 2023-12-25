# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import uuid

import d4rl
import gym
import numpy as np
import argparse
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from iql_utils import *
from networks import Potential as PotentialFunction, PotentialCond
import wandb

TensorBatch = List[torch.Tensor]


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        f_network: nn.Module,
        f_optimizer: torch.optim.Optimizer,
        beta_actor: nn.Module,
        beta_actor_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.f = f_network
        self.actor = actor
        self.beta_actor = beta_actor
        self.q_optimizer = q_optimizer
        self.v_optimizer = v_optimizer
        self.f_optimizer = f_optimizer
        self.actor_optimizer = actor_optimizer
        self.beta_actor_optimizer = beta_actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau
        self.W = 3
        self.temp = 1
        self.exp_adv_max = 100.0

        self.total_it = 0
        self.device = device

        
    def _update_v_f(self, observations, actions, log_dict) -> torch.Tensor:
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        f = self.f(observations, actions)
        adv = (target_q + torch.abs(f)) - v 
        #v_loss = asymmetric_l2_loss_original(adv, self.iql_tau)
        v_loss = torch.mean(adv**2)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    
    def _update_q(self,next_v, observations, actions, rewards, terminals, log_dict):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        soft_update(self.q_target, self.qf, self.tau)
        
        
    def _update_policy(self, adv, observations, actions, log_dict):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.exp_adv_max)
        policy_out = self.actor(observations)
        bc_losses = -policy_out.log_prob(actions)
            
        policy_loss = torch.mean(exp_adv * bc_losses)#self.f(observations, policy_out).mean()
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

        
    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v_f(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)
        # Update potential
        #if (self.total_it %10)==0:

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "f": self.f.state_dict(),
            "f_optimizer": self.f_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])
        
        self.f.load_state_dict(state_dict["f"])
        self.f_optimizer.load_state_dict(state_dict["f_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--task', default="IQL_(-Q-l2)-8g(s,a)_10000", type=str)
    parser.add_argument('--env', default="halfcheetah-medium-expert-v2", type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eval_freq', default=int(30e3), type=int)
    parser.add_argument('--n_episodes', default=100, type=int)
    parser.add_argument('--max_timesteps', default=int(1e6)+10, type=int)
    parser.add_argument('--checkpoints_path', default=None, type=str)
    parser.add_argument('--load_model', default="", type=str)

    # IQL
    parser.add_argument('--buffer_size', default=10000000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--beta', default=10.0, type=float)
    parser.add_argument('--iql_tau', default=0.9, type=float)
    parser.add_argument('--iql_deterministic', default = False, type=bool)
    parser.add_argument('--normalize', default = True, type=bool)
    parser.add_argument('--normalize_reward', default = True, type=bool)
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--qf_lr', default=3e-4, type=float)
    parser.add_argument('--vf_lr', default=3e-4, type=float)

    args = parser.parse_args()
    
    env = gym.make(args.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    #models_path = f"./saved_models/{args.env}/"
    models_path = f"../../NeurIPS2023/OT_RL/TD3_BC/saved_models/{args.env}/"

    dataset = d4rl.qlearning_dataset(env)

    if args.normalize_reward:
        modify_reward(dataset, args.env)

    if args.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        args.buffer_size,
        args.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    # if args.checkpoints_path is not None:
    #     print(f"Checkpoints path: {args.checkpoints_path}")
    #     os.makedirs(args.checkpoints_path, exist_ok=True)
    #     with open(os.path.join(args.checkpoints_path, "config.yaml"), "w") as f:
    #         pyrallis.dump(args, f)

    # Set seeds
    seed = args.seed
    set_seed(seed, env)

    q_network = TwinQ(state_dim, action_dim).to(args.device)
    v_network = ValueFunction(state_dim).to(args.device)
    #f_network = CondPotentialFunction(state_dim, action_dim).to(args.device) 
    #f_network = PotentialFunction(action_dim, hidden_dim = 1024).to(args.device)
    f_network = PotentialCond(action_dim, state_dim, hidden_dim=1024).to(args.device)
    actor = GaussianPolicy(state_dim, action_dim, max_action).to(args.device)
    beta_actor = GaussianPolicy(state_dim, action_dim, max_action).to(args.device)
    
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=3e-4)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=3e-4)
    f_optimizer = None#torch.optim.Adam(f_network.parameters(), lr=1e-4)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    beta_actor_optimizer = torch.optim.Adam(beta_actor.parameters(), lr=1e-4)
    
    #q_network.load_state_dict(torch.load(models_path + "qf199999.pt"))
    #v_network.load_state_dict(torch.load(models_path + "vf199999.pt"))
    #actor.load_state_dict(torch.load(models_path + "beta199999.pt"))
    #beta_actor.load_state_dict(torch.load(models_path + "beta999999.pt"))
    #f_network.load_state_dict(torch.load(models_path + "Extreme_A(s,a)_1000-8g(a).pt"))
    f_network.load_state_dict(torch.load(models_path + "-Q(s,rand(s))-l2_10000-8f(s,a).pt"))

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "beta_actor": beta_actor,
        "beta_actor_optimizer": beta_actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "f_network": f_network,
        "f_optimizer": f_optimizer,
        "discount": args.discount,
        "tau": args.tau,
        "device": args.device,
        # IQL
        "beta": args.beta,
        "iql_tau": args.iql_tau,
        "max_steps": args.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {args.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if args.load_model != "":
        policy_file = Path(args.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb.init(project='Bellman-Wasserstein(ICLR)',
        name=f"{args.task}_{args.env}",
        entity='rock-and-roll',
        reinit=True,
    )
    wandb.run.save()

    evaluations = []
    for t in range(int(args.max_timesteps)):
        batch = replay_buffer.sample(args.batch_size)
        batch = [b.to(args.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if t % args.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=args.device,
                n_episodes=args.n_episodes,
                seed=args.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {args.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            # if args.checkpoints_path is not None:
            #     torch.save(
            #         trainer.state_dict(),
            #         os.path.join(args.checkpoints_path, f"checkpoint.pt"),
            #     )
            wandb.log(
                {"d4rl_score": normalized_eval_score}, step=trainer.total_it
            )
