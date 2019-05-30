'''
Agents for cartpole
'''
import numpy as np
import torch
import torch.nn as nn
import math


class Buffer(object):
    """
    A finite-memory buffer that rewrites oldest data when buffer is full.
    Stores tuples of the form (feature, action, reward, next feature). 
    """
    def __init__(self, size=50000):
        self.size = size
        self.buffer = []
        self.next_idx = 0

    def add(self, x, a, r, x_next):
        if self.next_idx == len(self.buffer):
            self.buffer.append((x, a, r, x_next))
        else:
            self.buffer[self.next_idx] = (x, a, r, x_next)
        self.next_idx = (self.next_idx + 1) % self.size

    def sample(self, batch_size=1):
        idxs = np.random.randint(len(self.buffer), size=batch_size)
        return [self.buffer[i] for i in idxs]


class Agent(object):
    """
    generic class for agents
    """
    def __init__(self, action_set, reward_function):
        self.action_set = action_set
        self.reward_function = reward_function
        self.cummulative_reward = 0

    def __str__(self):
        pass

    def reset_cumulative_reward(self):
        self.cummulative_reward = 0

    def update_buffer(self, observation_history, action_history):
        pass

    def learn_from_buffer(self):
        pass

    def act(self, observation_history, action_history):
        pass

    def get_episode_reward(self, observation_history, action_history):
        tau = len(action_history)
        reward_history = np.zeros(tau)
        for t in range(tau):
            reward_history[t] = self.reward_function(
                observation_history[:t+2], action_history[:t+1])
        return reward_history

    def _random_argmax(self, action_values):
        argmax_list = np.where(action_values == np.max(action_values))[0]
        return self.action_set[argmax_list[np.random.randint(argmax_list.size)]]

    def _epsilon_greedy_action(self, action_values, epsilon):
        if np.random.random() < 1 - epsilon:
            return self._random_argmax(action_values)
        else:
            return np.random.choice(self.action_set, 1)[0]

    def _boltzmann_action(self, action_values, beta):
        action_values = action_values - max(action_values)
        action_probabilities = np.exp(action_values / beta)
        action_probabilities /= np.sum(action_probabilities)
        return np.random.choice(self.action_set, 1, p=action_probabilities)[0]

    def _epsilon_boltzmann_action(self, action_values, epsilon):
        action_values = action_values - max(action_values)
        action_probabilities = np.exp(action_values / (np.exp(1)*epsilon))
        action_probabilities /= np.sum(action_probabilities)
        return np.random.choice(self.action_set, 1, p=action_probabilities)[0]


class RandomAgent(Agent):
    """
    selects actions uniformly at random from the action set
    """
    def __str__(self):
        return "random agent"

    def act(self, observation_history, action_history):
        return np.random.choice(self.action_set, 1)[0]

    def update_buffer(self, observation_history, action_history):
        reward_history = self.get_episode_reward(observation_history, action_history)
        self.cummulative_reward += np.sum(reward_history)


class MLP(nn.Module):
    def __init__(self, dimensions):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dimensions)-1):
            self.layers.append(nn.Linear(dimensions[i], dimensions[i+1]))

    def forward(self, x):
        for l in self.layers[:-1]:
            x = nn.functional.relu(l(x))
        x = self.layers[-1](x)
        return x


class DQNAgent(Agent):
    def __init__(self, action_set, reward_function, feature_extractor, 
        hidden_dims=[50, 50], learning_rate=5e-4, buffer_size=50000, 
        batch_size=64, num_batches=100, starts_learning=5000, final_epsilon=0.02, 
        discount=0.99, target_freq=10, verbose=False, print_every=1, 
        test_model_path=None):

        Agent.__init__(self, action_set, reward_function)
        self.feature_extractor = feature_extractor
        self.feature_dim = self.feature_extractor.dimension

        # build Q network
        # we use a multilayer perceptron
        dims = [self.feature_dim] + hidden_dims + [len(self.action_set)]
        self.model = MLP(dims)

        if test_model_path is None:
            self.test_mode = False
            self.learning_rate = learning_rate
            self.buffer_size = buffer_size
            self.batch_size = batch_size
            self.num_batches = num_batches
            self.starts_learning = starts_learning
            self.epsilon = 1.0  # anneals starts_learning/(starts_learning + t)
            self.final_epsilon = 0.02
            self.timestep = 0
            self.discount = discount
            
            self.buffer = Buffer(self.buffer_size)

            self.target_net = MLP(dims)
            self.target_net.load_state_dict(self.model.state_dict())
            self.target_net.eval()

            self.target_freq = target_freq # target nn updated every target_freq episodes
            self.num_episodes = 0

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # for debugging purposes
            self.verbose = verbose
            self.running_loss = 1.
            self.print_every = print_every

        else:
            self.test_mode = True
            self.model.load_state_dict(torch.load(test_model_path))
            self.model.eval()
        

    def __str__(self):
        return "dqn"


    def update_buffer(self, observation_history, action_history):
        """
        update buffer with data collected from current episode
        """
        reward_history = self.get_episode_reward(observation_history, action_history)
        self.cummulative_reward += np.sum(reward_history)

        tau = len(action_history)
        feature_history = np.zeros((tau+1, self.feature_extractor.dimension))
        for t in range(tau+1):
            feature_history[t] = self.feature_extractor.get_feature(observation_history[:t+1])

        for t in range(tau-1):
            self.buffer.add(feature_history[t], action_history[t], 
                reward_history[t], feature_history[t+1])
        done = observation_history[tau][1]
        if done:
            feat_next = None
        else:
            feat_next = feature_history[tau]
        self.buffer.add(feature_history[tau-1], action_history[tau-1], 
            reward_history[tau-1], feat_next)


    def learn_from_buffer(self):
        """
        update Q network by applying TD steps
        """
        if self.timestep < self.starts_learning:
            pass

        for _ in range(self.num_batches):
            minibatch = self.buffer.sample(batch_size=self.batch_size)
            
            feature_batch = torch.zeros(self.batch_size, self.feature_dim)
            action_batch = torch.zeros(self.batch_size, 1, dtype=torch.long)
            reward_batch = torch.zeros(self.batch_size, 1)
            non_terminal_idxs = []
            next_feature_batch = []
            for i, d in enumerate(minibatch):
                x, a, r, x_next = d
                feature_batch[i] = torch.from_numpy(x)
                action_batch[i] = torch.tensor(a, dtype=torch.long)
                reward_batch[i] = r
                if x_next is not None:
                    non_terminal_idxs.append(i)
                    next_feature_batch.append(x_next)

            model_estimates = self.model(feature_batch).gather(1, action_batch)
            future_values = torch.zeros(self.batch_size)
            if next_feature_batch != []:
                next_feature_batch = torch.tensor(next_feature_batch, dtype=torch.float)
                future_values[non_terminal_idxs] = self.target_net(next_feature_batch).max(1)[0].detach()
            future_values = future_values.unsqueeze(1)
            target_values = reward_batch + self.discount * future_values

            loss = nn.functional.mse_loss(model_estimates, target_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.running_loss = 0.99 * self.running_loss + 0.01 * loss.item()

        self.epsilon = self.starts_learning / (self.starts_learning + self.timestep)
        self.epsilon = max(self.final_epsilon, self.epsilon)

        self.num_episodes += 1

        if self.verbose and (self.num_episodes % self.print_every == 0):
            print("dqn ep %d, running loss %.2f" % (self.num_episodes, self.running_loss))

        if self.num_episodes % self.target_freq == 0:
            self.target_net.load_state_dict(self.model.state_dict())
            if self.verbose:
                print("dqn ep %d update target network" % self.num_episodes)

    def act(self, observation_history, action_history):
        """ select action according to an epsilon greedy policy with respect to 
        the Q network """
        feature = self.feature_extractor.get_feature(observation_history)
        with torch.no_grad():
            action_values = self.model(torch.from_numpy(feature).float()).numpy()
        if not self.test_mode:
            action = self._epsilon_greedy_action(action_values, self.epsilon)
            self.timestep += 1
        else:
            action = self._random_argmax(action_values)
        return action

    def save(self, path=None):
        if path is None:
            path = './dqn.pt'
        torch.save(self.model.state_dict(), path)


def mountain_car_reward_function(observation_history, action_history):
    """
    Always returns -1 because mountain car has a reward of -1 at every timestep
    """
    return -1
