import numpy as np
import torch
import os
import functools

import matplotlib.pyplot as plt
from tqdm import trange

from live import live
from agents import RandomAgent
from agents import DQNAgent
from agents import mountain_car_reward_function
from feature import MountainCarIdentityFeature

import gym

if __name__ == '__main__':
    print(os.listdir('.'))
    reward_path = './results/'
    agent_path = './agents/'

    env = gym.make('MountainCar-v0') 

    # train dqn agents
    number_seeds = 3
    for seed in trange(number_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        agent = DQNAgent(
            action_set=[0, 1, 2],
            reward_function=mountain_car_reward_function,
            feature_extractor=MountainCarIdentityFeature(),
            hidden_dims=[50, 50],
            learning_rate=5e-4,
            buffer_size=50000,
            batch_size=64,
            num_batches=100,
            starts_learning=5000,
            final_epsilon=0.02,
            discount=0.99,
            target_freq=10,
            verbose=False, 
            print_every=10)

        _, _, rewards = live(
            agent=agent,
            environment=env,
            num_episodes=10, 
            max_timesteps=500,
            verbose=True,
            print_every=50)

        file_name = '|'.join(['dqn_mountaincar', str(seed)])
        np.save(os.path.join(reward_path, file_name), rewards)
        agent.save(path=os.path.join(agent_path, file_name+'.pt'))
