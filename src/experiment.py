import numpy as np
import torch
import os
import functools

import matplotlib.pyplot as plt
from tqdm import trange

from live import live
from agents import RandomAgent, mountain_car_reward_function, DQNAgent, TabularLsviAgent
from rlsvi import RLSVIIncrementalTDAgent, TabularRlsviAgent
from feature import MountainCarIdentityFeature

import gym

if __name__ == '__main__':
    print(os.listdir('.'))
    reward_path = './results/'
    agent_path = './agents/'


    ## Start with noise variance of 1 because we incur a penalty of 1 at each timestep (in Deep Sea, we used
    ## a noise variance of 0.01, which was the same as the move penalty at each).
    ## Start with a prior variance of 10
    prior_var = 10.0
    noise_var = 1.0
    dims = [25, 25]
    episodes = 10000

    env = gym.make('MountainCar-v0') 

    # train dqn agents
    number_seeds = 10
    for seed in trange(number_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        agent = RLSVIIncrementalTDAgent(
            action_set=[0, 1, 2],
            reward_function=mountain_car_reward_function,
            prior_variance=prior_var,
            noise_variance=noise_var,
            feature_extractor=MountainCarIdentityFeature(),
            prior_network=True,
            num_ensemble=10,
            hidden_dims=dims,
            learning_rate=5e-4,
            buffer_size=50000,
            batch_size=64,
            num_batches=20,
            starts_learning=5000,
            discount=0.99,
            target_freq=10,
            verbose=True,
            print_every=10)

        _, _, rewards = live(
            agent=agent,
            environment=env,
            num_episodes=episodes, 
            max_timesteps=500,
            verbose=True,
            print_every=50)

        file_name = '|'.join(['dqn_RLSVI_mountaincar', 'noise_var' + str(noise_var), 'prior_var' + str(prior_var), 'dims' + str(dims), str(seed)])
        np.save(os.path.join(reward_path, file_name), rewards)
        agent.save(path=os.path.join(agent_path, file_name+'.pt'))
