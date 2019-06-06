'''
RLSVI agent for cartpole via ensemble sampling 
'''

import numpy as np
from agents import Agent

class TabularRlsviAgent(Agent):
    def __init__(self,
                 action_set,
                 reward_function,
                 prior_variance,
                 noise_variance,
                 num_iterations,
                 feature_extractor):
        Agent.__init__(self,action_set,reward_function)
        self.prior_variance = prior_variance
        self.noise_variance = noise_variance
        self.num_iterations = num_iterations
        self.feature_extractor = feature_extractor

        # buffer is a dictionary of lists
        # the key is a feature-action pair
        self.buffer = {(f, a): [] for f in self.feature_extractor.feature_space for a in self.action_set}
        self.Q = {key: np.sqrt(self.prior_variance) * np.random.randn() for key in self.buffer.keys()}

    def __str__(self):
        return 'tabular_rlsvi_agent'

    def update_buffer(self, observation_history, action_history):
        reward_history = self.get_episode_reward(observation_history, action_history)
        self.cummulative_reward += np.sum(reward_history)
        tau = len(action_history)
        feature_history = [self.feature_extractor.get_feature(observation_history[:t + 1])
                           for t in range(tau + 1)]
        for t in range(tau - 1):
            new_key = (feature_history[t], action_history[t])
            new_item = (reward_history[t], feature_history[t + 1])
            self.buffer[new_key].append(new_item)
        done = observation_history[tau][1]
        if done:
            feat_next = None
        else:
            feat_next = feature_history[tau]

        new_key = (feature_history[tau - 1], action_history[tau - 1])
        new_item = (reward_history[tau - 1], feat_next)
        self.buffer[new_key].append(new_item)

    def learn_from_buffer(self):
        perturbed_buffer = {key: [(transition[0] + np.sqrt(self.noise_variance) * np.random.randn(),
                                   transition[1]) for transition in self.buffer[key]]
                            for key in self.buffer.keys()}
        random_Q = {key: np.sqrt(self.prior_variance) * np.random.randn() for key in self.buffer.keys()}
        Q = {key: 0.0 for key in self.buffer.keys()}
        Q_temp = {key: 0.0 for key in self.buffer.keys()}
        for n in range(self.num_iterations):
            for key in self.buffer.keys():
                q = 0.0
                for transition in perturbed_buffer[key]:
                    if transition[1] == None:
                        q += transition[0]
                    else:
                        v = max(Q[(transition[1], a)] for a in self.action_set)
                        q += transition[0] + v
                Q_temp[key] = (1.0 / ((len(self.buffer[key]) / self.noise_variance) + (1.0 / self.prior_variance))) \
                              * ((q / self.noise_variance) + (random_Q[key] / self.prior_variance))
            Q = Q_temp
            Q_temp = {key: 0.0 for key in self.buffer.keys()}
        self.Q = Q

    def act(self, observation_history, action_history):
        feature = self.feature_extractor.get_feature(observation_history)
        return self._random_argmax([self.Q[(feature, a)] for a in self.action_set])


# -----------------------------------------------------------
# Incremental TD using neural networks
# ----------------------------------------------------------

import torch
import torch.multiprocessing
import torch.nn as nn
import typing

has_gpu = torch.cuda.is_available()
device = torch.device("cuda" if has_gpu else "cpu")
#device = torch.device("cpu") # comment this if you want to use a GPU
print('using '+str(device))

from utils import Buffer, MLP, DQNWithPrior



# -----------------------------------------------------------------
# RLSVI Agent (with ensemble)
# ----------------------------------------------------------------

class RLSVIIncrementalTDAgent(Agent):
    def __init__(self, action_set, reward_function,
                 prior_variance,noise_variance,
                 feature_extractor, prior_network, num_ensemble,
                 hidden_dims=[10, 10], learning_rate=5e-4, buffer_size=50000,
                 batch_size=64, num_batches=100, starts_learning=5000,
                 discount=0.99, target_freq=10, verbose=False, print_every=1,
                 test_model_path=None, GRLSVI=True, BRLSVI=True):
        Agent.__init__(self, action_set, reward_function)

        self.GRLSVI = GRLSVI
        self.BRLSVI = BRLSVI
        self.prior_variance = prior_variance

        ## No gaussian perturbations equivalent to zero variance
        if GRLSVI:
            self.noise_variance = noise_variance
        else:
            self.noise_variance = 0.0

        ## On average, with the double or none bootstrap, each batch will
        ## effectively be half as large. Account for this by doubling batch size

        if BRLSVI:
            batch_size *= 2    
        

        self.feature_extractor = feature_extractor
        self.feature_dim = self.feature_extractor.dimension

        dims = [self.feature_dim] + hidden_dims + [len(self.action_set)]

        self.prior_network = prior_network
        self.num_ensemble = num_ensemble # number of models in ensemble

        self.index = np.random.randint(self.num_ensemble)


        # build Q network
        # we use a multilayer perceptron

        if test_model_path is None:
            self.test_mode = False
            self.learning_rate = learning_rate
            self.buffer_size = buffer_size
            self.batch_size = batch_size
            self.num_batches = num_batches
            self.starts_learning = starts_learning
            self.discount = discount
            self.timestep = 0

            self.buffer = Buffer(self.buffer_size)
            self.models = []
            for i in range(self.num_ensemble):
                if self.prior_network:
                    '''
                    Second network is a prior network whose weights are fixed
                    and first network is difference network learned i.e, weights are mutable
                    '''
                    self.models.append(DQNWithPrior(dims,
                                        scale = np.sqrt(self.prior_variance)).to(device))
                else:
                    self.models.append(MLP(dims).to(device))
                self.models[i].initialize()

            '''
            prior networks weights are immutable so enough to keep difference network
            '''
            self.target_nets = []
            for i in range(self.num_ensemble):
                if self.prior_network:
                    self.target_nets.append(DQNWithPrior(dims,
                                        scale = np.sqrt(self.prior_variance)).to(device))
                else:
                    self.target_nets.append(MLP(dims).to(device))
                    self.target_nets[i].load_state_dict(self.models[i].state_dict())
                    self.target_nets[i].eval()

            self.target_freq = target_freq #   target nn updated every target_freq episodes
            self.num_episodes = 0

            self.optimizer = []
            for i in range(self.num_ensemble):
                self.optimizer.append(torch.optim.Adam(self.models[i].parameters(),
                                                     lr=self.learning_rate))

            # for debugging purposes
            self.verbose = verbose
            self.running_loss = 1.
            self.print_every = print_every

        else:
            self.models =[]
            self.test_mode = True
            if self.prior_network:
                self.models.append(DQNWithPrior(dims,scale=self.prior_variance))
            else:
                self.models.append(MLP(dims))
            self.models[0].load_state_dict(torch.load(test_model_path))
            self.models[0].eval()
            self.index = 0



    def __str__(self):
        return 'rlsvi_incremental_TD_'+str(self.num_ensemble)+'models'


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
            if self.BRLSVI:
                mask = np.random.randint(0, 2, size=self.num_ensemble)
            else:
                mask = np.random.randint(1, 2, size=self.num_ensemble)
            
            perturbations = np.random.randn(self.num_ensemble)*np.sqrt(self.noise_variance)
            self.buffer.add((feature_history[t], action_history[t],
                reward_history[t], feature_history[t+1], perturbations, mask))
        
        done = observation_history[tau][1]
        if done:
            feat_next = None
        else:
            feat_next = feature_history[tau]

        if self.BRLSVI:
            mask = np.random.randint(0, 2, size=self.num_ensemble)
        else:
            mask = np.random.randint(1, 2, size=self.num_ensemble)  
        perturbations = np.random.randn(self.num_ensemble)*np.sqrt(self.noise_variance)
        self.buffer.add((feature_history[tau-1], action_history[tau-1],
            reward_history[tau-1], feat_next, perturbations, mask))


    def learn_from_buffer(self):
        """
        update Q network by applying TD steps
        """
        if self.timestep < self.starts_learning:
            pass

        loss_ensemble = 0

        for _ in range(self.num_batches):
            for sample_num in range(self.num_ensemble):
                minibatch = self.buffer.sample(batch_size=self.batch_size)
                
                bootstrap_keep = []
                for transition_ind in range(self.batch_size):
                    if minibatch[transition_ind][5][sample_num] == 1:
                        bootstrap_keep.append(transition_ind)

                minibatch = [minibatch[i] for i in bootstrap_keep]
                effective_batch_size = len(minibatch)

                feature_batch = torch.zeros(effective_batch_size, self.feature_dim, device=device)
                action_batch = torch.zeros(effective_batch_size, 1, dtype=torch.long, device=device)
                reward_batch = torch.zeros(effective_batch_size, 1, device=device)
                perturb_batch = torch.zeros(effective_batch_size, self.num_ensemble, device=device)
                mask_batch = torch.zeros(effective_batch_size, self.num_ensemble, device=device)
                non_terminal_idxs = []
                next_feature_batch = []

                for i, d in enumerate(minibatch):
                    s, a, r, s_next, perturb, mask = d
                    feature_batch[i] = torch.from_numpy(s)
                    action_batch[i] = torch.tensor(a, dtype=torch.long)
                    reward_batch[i] = r
                    perturb_batch[i] = torch.from_numpy(perturb)
                    mask_batch[i] = torch.from_numpy(mask)
                    if s_next is not None:
                        non_terminal_idxs.append(i)
                        next_feature_batch.append(s_next)

                model_estimates = ( self.models[sample_num](feature_batch)
                                ).gather(1, action_batch).float()

                future_values = torch.zeros(effective_batch_size, device=device)
                if non_terminal_idxs != []:
                    next_feature_batch = torch.tensor(next_feature_batch,
                                                      dtype=torch.float, device=device)
                    future_values[non_terminal_idxs] = (
                        self.target_nets[sample_num](next_feature_batch)
                        ).max(1)[0].detach()
                future_values = future_values.unsqueeze(1)
                target_values = reward_batch + self.discount * future_values \
                                + perturb_batch[:,sample_num].unsqueeze(1)

                assert(model_estimates.shape==target_values.shape)

                loss = nn.functional.mse_loss(model_estimates, target_values)

                self.optimizer[sample_num].zero_grad()
                loss.backward()
                self.optimizer[sample_num].step()
                loss_ensemble += loss.item()
        self.running_loss = 0.99 * self.running_loss + 0.01 * loss_ensemble

        self.num_episodes += 1

        self.index = np.random.randint(self.num_ensemble)

        if self.verbose and (self.num_episodes % self.print_every == 0):
            print("rlsvi ep %d, running loss %.2f, reward %.3f, index %d" % (self.num_episodes,
                                self.running_loss, self.cummulative_reward, self.index))

        if self.num_episodes % self.target_freq == 0:
            for sample_num in range(self.num_ensemble):
                self.target_nets[sample_num].load_state_dict(self.models[sample_num].state_dict())
            # if self.verbose:
            #     print("rlsvi via ensemble sampling ep %d update target network" % self.num_episodes)


    def act(self, observation_history, action_history):
        """ select action according to an epsilon greedy policy with respect to
        the Q network """
        feature = self.feature_extractor.get_feature(observation_history)
        with torch.no_grad():
            if str(device)=="cpu":
                action_values = (
                    self.models[self.index](torch.tensor(feature).float())
                    ).numpy()
            else:
                out = (self.models[self.index](torch.tensor(feature).float().to(device)))
                action_values = (out.to("cpu")).numpy()

            action = self._random_argmax(action_values)
        return action

    def save(self, path=None):
        if path is None:
            path = './'+self.__str__()+'.pt'
        torch.save(self.models[self.index].state_dict(), path)
