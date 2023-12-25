import os
import numpy as np
import torch
import gym
import argparse
import d4rl
import d4rl.gym_mujoco
import utils
import bellman_wasserstein as Distance
import wandb

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", default="1")                # Cuda device
    parser.add_argument("--task", default="-Q(s,rand(s)-l2)_10000-1f(s,a)_") 
    parser.add_argument("--env", default="halfcheetah-medium-v2")   # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1000, type=int)     # How often (time steps) we evaluates
    parser.add_argument("--timesteps", default=3001, type=int)   # Num of time steps to update
    parser.add_argument("--distance_freq", default = 1e4, type=int)

    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--hidden_dim", default=1024, type=int)     # Hidden layer size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates

    parser.add_argument("--alpha", default=1)
    parser.add_argument("--W", default=1)
    parser.add_argument("--normalize", default=True)
    args = parser.parse_args()

    seed = int(args.seed)
    
    wandb.init(project='Bellman-Wasserstein(ICLR)',
        name=f"{args.task}_{args.env}",
        entity='rock-and-roll',
        reinit=True,
    )
    wandb.run.save()

    models_path = f"../../NeurIPS2023/OT_RL/TD3_BC/saved_models/{args.env}/"

    env = gym.make(args.env)
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "max_action": max_action,
        "max_steps": args.timesteps,
        "discount": args.discount,
        "tau": args.tau,
        "alpha": args.alpha,
        "W": args.W
    }
    
    method = Distance.BellmanWasserstein(**kwargs)

    method.critic.load_state_dict(torch.load(models_path + "Sarsa_Critic_10000.pt"))
    method.value.load_state_dict(torch.load(models_path + "Sarsa_Value_10000.pt"))
    print("Models uploaded")

    replay_buffer = utils.SarsaReplayBuffer(state_dim, action_dim, max_size=int(10000000))
    sarsa_dataset = utils.qlearning_dataset(env)
    replay_buffer.convert_D4RL(sarsa_dataset)
    if args.normalize:
        mean,std = replay_buffer.normalize_states() 
    else:
        mean,std = 0,1

    for t in range(int(args.timesteps)):
        log_dict = method.update_distance_cond(replay_buffer, advantage=True)
        wandb.log(log_dict, step=t)
        torch.cuda.empty_cache()
        
        if t % args.eval_freq==0:
            log_dict = method.estimate_distance_cond(replay_buffer,advantage=True)
            wandb.log(log_dict, step=t)
    
    torch.save(method.potential_1.state_dict(), models_path + "-Q(s,rand(s)-l2)_10000-1f(s,a).pt")
    torch.save(method.potential_2.state_dict(), models_path + "-Q(s,rand(s)-l2)_10000-1f(s,a).pt")
    torch.cuda.empty_cache()
        

