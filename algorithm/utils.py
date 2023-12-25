import numpy as np
import torch
import torch.nn as nn
import gym


def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    next_action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N - 1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i + 1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        new_action = dataset['actions'][i + 1].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        action_.append(action)
        reward_.append(reward)
        next_obs_.append(new_obs)
        next_action_.append(new_action)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'rewards': np.array(reward_),
        'next_observations': np.array(next_obs_),
        'next_actions': np.array(next_action_),
        'terminals': np.array(done_),
    }

class SarsaReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.next_action[self.ptr] = next_action
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.next_action[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.reward = dataset['rewards'].reshape(-1,1)
        self.next_state = dataset['next_observations']
        self.next_action = dataset['next_actions']
        self.not_done = 1. - dataset['terminals'].reshape(-1,1)
        self.size = self.state.shape[0]


    def normalize_states(self, eps = 1e-3):
        mean = self.state.mean(0,keepdims=True)
        std = self.state.std(0,keepdims=True) + eps
        self.state = (self.state - mean)/std
        self.next_state = (self.next_state - mean)/std
        return mean, std




class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1,1)
        self.not_done = 1. - dataset['terminals'].reshape(-1,1)
        self.size = self.state.shape[0]


    def normalize_states(self, eps = 1e-3):
        mean = self.state.mean(0,keepdims=True)
        std = self.state.std(0,keepdims=True) + eps
        self.state = (self.state - mean)/std
        self.next_state = (self.next_state - mean)/std
        return mean, std

class BaselineLearner(object):
    def __init__(self, *args, **kwargs):
        return

    def set_logger(self, logger):
        self.logger = logger

    def loss(self, transitions, q, pi):
        raise NotImplementedError

    def predict(self, state, q, pi, batch=True):
        raise NotImplementedError

    def train_step(self, replay, q, pi):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError



class SampledValueBaseline(BaselineLearner):
    def __init__(self, device, n_samples, *args, **kwargs):
        self.device = torch.device(device)
        self.n_samples = n_samples

    def _get_sampled_qvals(self, critic, policy, state, n_samples):
        batch_size = state.shape[0]
        state_shape = tuple(state.shape[1:])

        # sample actions
        dist = policy.get_dist(state)
        actions = dist.sample((n_samples,))

        # reshape
        action_shape = tuple(actions.shape[2:])
        actions = actions.reshape((n_samples * batch_size, *action_shape))

        # calculate q values
        ones = tuple([1 for i in range(len(state.shape))])
        repeated_states = state.unsqueeze(0).repeat((n_samples, *ones))
        repeated_states = repeated_states.reshape((n_samples * 
                                                batch_size, *state_shape))
        qvals = critic(repeated_states, actions)
        qvals = qvals.reshape((n_samples, batch_size, 1))

        return qvals

    def predict(self, state, critic, policy):
        qvals = self._get_sampled_qvals(critic, policy, state, self.n_samples)

        # compute mean
        value = qvals.mean(dim=0)

        return value

def select_action(state, policy, device):
    state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    return policy(state).cpu().data.numpy().flatten()

def eval_policy(policy, env_name, seed, mean, std, device='cuda', seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1,-1) - mean)/std
            action = select_action(state, policy, device)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score

def log_prob_func(dist, sample):
    log_prob = dist.log_prob(sample)
    if len(log_prob.shape) == 1:
        return log_prob
    else:
        return log_prob.sum(-1, keepdim=True)

def soft_clamp(x, low, high):
    x = torch.tanh(x)
    x = low + 0.5 * (high - low) * (x + 1)
    return x

class NegAbs(nn.Module):
    def __init__(self):
        super(NegAbs, self).__init__()

    def forward(self, input):
        return -torch.abs(input)