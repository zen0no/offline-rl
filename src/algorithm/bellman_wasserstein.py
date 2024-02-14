import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from networks import GaussActor, Actor, Critic, Value, Potential, PotentialCond
from utils import log_prob_func, NegAbs, SampledValueBaseline
from torch.optim.lr_scheduler import CosineAnnealingLR

class BellmanWasserstein(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        batch_size,
        max_action,
        actor_lr = 1e-4,
        qf_lr = 1e-4,
        vf_lr = 1e-4,
        max_steps=1e6,
        dist_policy= False,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        policy_freq=2,
        alpha=2.5,
        W = 1,
        device='cpu',
    ):
        
        #self.beta = GaussActor(state_dim, hidden_dim, action_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.value = Value(state_dim, hidden_dim).to(device)
        # self.potential_1 = Potential(action_dim, hidden_dim).to(device)
        # self.potential_2 = Potential(state_dim, hidden_dim).to(device)
        
        self.potential_1 = PotentialCond(action_dim, state_dim, hidden_dim).to(device)
        self.potential_2 = PotentialCond(action_dim, state_dim, hidden_dim).to(device)
            
        #self.beta_optimizer = torch.optim.Adam(self.beta.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=qf_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=vf_lr)
        self.potential_1_optimizer = torch.optim.Adam(self.potential_1.parameters(), lr=1e-4)
        self.potential_2_optimizer = torch.optim.Adam(self.potential_2.parameters(), lr=1e-4)
        
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size
        self.W = W

    
    def update_critic(self, replay_buffer):
        log_dict = {}
        state, action, reward, next_state, next_action, not_done = replay_buffer.sample(self.batch_size)
                    
        with torch.no_grad():
            #next_action = self.beta(next_state).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        log_dict["critic_loss"] = critic_loss.item()
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return log_dict
    
    def update_value(self, replay_buffer):
        log_dict = {}
        state, action, reward, next_state, next_action, not_done = replay_buffer.sample(self.batch_size)
                    
        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_V = self.value(state)
        value_loss = F.mse_loss(current_V, target_Q)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        log_dict["value_loss"] = value_loss.item()
    
        return log_dict
    
    
#     def update_distance(self, replay_buffer, epsilon=1, advantage=False):
#         log_dict = {}
#         state, action, reward, next_state, next_action, not_done = replay_buffer.sample(self.batch_size)

#         self.potential_1_optimizer.zero_grad()
#         self.potential_2_optimizer.zero_grad()

#         if advantage:
#             cost = (self.critic(state, action)[0] - self.value(state))
#         else:
#             cost = self.critic(state, action)[0]
#         reg = -torch.mean(1/(4*epsilon)*(self.potential_1(action).flatten() + (self.W*self.potential_2(state)).flatten() + cost).relu()**2)#mistake here, W unnecessary

#         d_loss = -(self.potential_1(action).mean() + (self.W*self.potential_2(state)).mean() + reg)
#         d_loss.backward()

#         self.potential_1_optimizer.step()
#         self.potential_2_optimizer.step()

#         log_dict["batch_bw"] = d_loss.item()
        
#         return log_dict
    
    
#     def estimate_distance(self, replay_buffer,  epsilon=1, advantage=False):
#         log_dict = {}
#         state, action, reward, next_state, next_action, not_done = replay_buffer.sample(10000)

#         if advantage:
#             cost = -(self.critic(state, action)[0] - self.value(state))
#         else:
#             cost = -self.critic(state, action)[0]
#         reg = -torch.mean(1/(4*epsilon)*(self.potential_1(action).flatten() + (self.W*self.potential_2(state)).flatten() - cost).relu()**2)#mistake here, W unnecessary

#         distance = self.potential_1(action).mean() + (self.W*self.potential_2(state)).mean() + reg
#         if advantage:
#             log_dict["A-bellman-wasserstein"] = distance.item()
#         else:
#             log_dict["bellman-wasserstein"] = distance.item()
#         log_dict["reward"] = reward.mean().item()
#         log_dict["Q mean"] = self.critic(state, action)[0].mean().item()
#         log_dict["A mean"] = (self.critic(state, action)[0] - self.value(state)).mean().item()
        
#         return log_dict
    
    
    def update_distance_cond(self, replay_buffer, epsilon=1, advantage=False):
        log_dict = {}
        state, action, reward, next_state, next_action, not_done = replay_buffer.sample(self.batch_size)

        self.potential_1_optimizer.zero_grad()
        self.potential_2_optimizer.zero_grad()

        random_action = torch.randn_like(action).clamp(-self.max_action, self.max_action)
        
        cost = -self.critic(state, random_action)[0] - F.mse_loss(random_action, action)
        reg = torch.mean(1/(4*epsilon)*(self.potential_1(state, random_action).flatten() + self.potential_2(state, action).flatten() - cost).relu()**2)
        d_loss = -(self.potential_1(state, random_action).mean() + (self.W*self.potential_2(state, action)).mean()) + reg
        
        d_loss.backward()

        self.potential_1_optimizer.step()
        self.potential_2_optimizer.step()

        log_dict["batch_bw"] = d_loss.item()
        
        return log_dict
    
    
    
    def estimate_distance_cond(self, replay_buffer,  epsilon=1, advantage=False):
        log_dict = {}
        state, action, reward, next_state, next_action, not_done = replay_buffer.sample(20000)

        random_action = torch.randn_like(action).clamp(-self.max_action, self.max_action)
        
        cost = -self.critic(state, random_action)[0] - F.mse_loss(random_action, action)
        reg = torch.mean(1/(4*epsilon)*(self.potential_1(state, random_action).flatten() + self.potential_2(state, action).flatten() - cost).relu()**2)
        distance = -(self.potential_1(state, random_action).mean() + (self.W*self.potential_2(state, action)).mean()) + reg

        log_dict["BWD"] = distance.item()
        log_dict["reward"] = reward.mean().item()
        log_dict["Q mean"] = self.critic(state, action)[0].mean().item()
        log_dict["A mean"] = (self.critic(state, action)[0] - self.value(state)).mean().item()
        log_dict["A-performance difference"] = (self.critic(state, random_action)[0] - self.value(state)).mean().item()
            
        return log_dict
    
    
    def update_distance_cond_R(self, replay_buffer, epsilon=1, advantage=False):
        log_dict = {}
        state, action, reward, next_state, next_action, not_done = replay_buffer.sample(self.batch_size)

        self.potential_1_optimizer.zero_grad()
        self.potential_2_optimizer.zero_grad()

        random_action = torch.randn_like(action).clamp(-self.max_action, self.max_action)
        
        cost = -self.critic(state, action)[0] - F.mse_loss(random_action, action)
        reg = torch.mean(1/(4*epsilon)*(self.potential_1(state, random_action).flatten() + self.potential_2(state, action).flatten() - cost).relu()**2)
        d_loss = -(self.potential_1(state, random_action).mean() + (self.W*self.potential_2(state, action)).mean()) + reg
        
        d_loss.backward()

        self.potential_1_optimizer.step()
        self.potential_2_optimizer.step()

        log_dict["batch_bw"] = d_loss.item()
        
        return log_dict
    
    
    
    def estimate_distance_cond_R(self, replay_buffer,  epsilon=1, advantage=False):
        log_dict = {}
        state, action, reward, next_state, next_action, not_done = replay_buffer.sample(20000)

        random_action = torch.randn_like(action).clamp(-self.max_action, self.max_action)
        
        cost = -self.critic(state, action)[0] - F.mse_loss(random_action, action)
        reg = torch.mean(1/(4*epsilon)*(self.potential_1(state, random_action).flatten() + self.potential_2(state, action).flatten() - cost).relu()**2)
        distance = -(self.potential_1(state, random_action).mean() + (self.W*self.potential_2(state, action)).mean()) + reg

        log_dict["BWD"] = distance.item()
        log_dict["reward"] = reward.mean().item()
        log_dict["Q mean"] = self.critic(state, action)[0].mean().item()
        log_dict["A mean"] = (self.critic(state, action)[0] - self.value(state)).mean().item()
        log_dict["A-performance difference"] = (self.critic(state, action)[0] - self.value(state)).mean().item()
        
        return log_dict
    

    def save_checkpoints(self, checkpoint_path: str, timestep: int = -1):
        assert os.path.exists(checkpoint_path)

        checkp_name = str(timestep) if timestep != -1 else 'latest'
        
        critic_path = os.path.join(checkpoint_path, f'Sarsa_Critic_{checkp_name}.pt')
        value_path = os.path.join(checkpoint_path, f'Sarsa_Value_{checkp_name}.pt')

        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.value.state_dict(), value_path)

    def save_optimizers(self, checkpoint_path: str, timestep: int = -1 ):
        assert os.path.exists(checkpoint_path)

        checkp_name = str(timestep) if timestep != -1 else 'latest'


        critic_path = os.path.join(checkpoint_path, f'Sarsa_Critic_{checkp_name}.pt')
        value_path = os.path.join(checkpoint_path, f'Sarsa_Value_{checkp_name}.pt')

        assert os.path.exists(critic_path) and os.path.exists(value_path)

        torch.save(self.critic_optimizer.state_dict(), critic_path)
        torch.save(self.value_optimizer.state_dict(), value_path)

    def load(self, checkpoint_path: str, timestep: int = -1):

        checkp_name = str(timestep) if timestep != -1 else 'latest'
        critic_path = os.path.join(checkpoint_path, f'Sarsa_Critic_{checkp_name}.pt')
        value_path = os.path.join(checkpoint_path, f'Sarsa_Value_{checkp_name}.pt')
        
        assert os.path.exists(critic_path) and os.path.exists(value_path)


        self.critic.load_state_dict(torch.load(critic_path))
        self.value.load_state_dict(torch.load(value_path))
        print("Models uploaded")