import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from utils import NegAbs, soft_clamp

class MLP(nn.Module):
    def __init__(self, input_shape, output_dim,
                        hidden_dim, depth):
        super().__init__()
        input_dim = torch.tensor(input_shape).prod()
        layers = [nn.Linear(input_dim, hidden_dim),
                    nn.ReLU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(nn.Flatten(), *layers)

    def forward(self, x):
        return self.net(x)    
    
class GaussActor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, depth=2, dist_type='normal'):
        super().__init__()
        self.net = MLP(input_shape=(state_dim), output_dim=2*action_dim,
                        hidden_dim=hidden_dim, depth=depth)
        self.log_std_bounds = (-5., 0.)
        self.mu_bounds = (-1., 1.)
        self.dist_type = dist_type

    def get_dist(self, s):
        s = torch.flatten(s, start_dim=1)
        mu, log_std = self.net(s).chunk(2, dim=-1)

        mu = soft_clamp(mu, *self.mu_bounds)
        log_std = soft_clamp(log_std, *self.log_std_bounds)

        std = log_std.exp()
        dist = D.Normal(mu, std)
        return dist
    
    def forward(self, state):
        dist = self.get_dist(state)
        action = dist.sample()
        action = action.clamp(-1., 1.)
        return action
        

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        #return a #alternative
        return self.max_action * torch.tanh(a)

    
class Potential(nn.Module):
    def __init__(self, action_dim, hidden_dim):
        super(Potential, self).__init__()

        self.l1 = nn.Linear(action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.activation = NegAbs()

    def forward(self, action):
        f = F.relu(self.l1(action))
        f = F.relu(self.l2(f))
        f = self.l3(f)
        return self.activation(f)

class PotentialCond(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PotentialCond, self).__init__()

        self.l1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.activation = NegAbs()

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        f = F.relu(self.l1(sa))
        f = F.relu(self.l2(f))
        f = self.l3(f)
        return self.activation(f)
    

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
        

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
    
class Value(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Value, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        

    def forward(self, state):
        v1 = F.relu(self.l1(state))
        v2 = F.relu(self.l2(v1))
        v3 = self.l3(v2)
        return v3