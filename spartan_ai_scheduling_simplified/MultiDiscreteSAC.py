import numpy as np
import os

import torch.cuda
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import INIT_PARAMS_SIMPLE as IPS
from typing import List, Tuple
import helpers
import storage_classes as sc
import PhysicalStorageClasses as psc


def get_conv_input_shape() -> Tuple[float]:
    x = 1 + 1 + max(IPS.FURNACE_WIDTH)
    y = max(IPS.WAIT_BAY_LEN, IPS.n_furnaces * max(IPS.FURNACE_HEIGHT) * IPS.N_MAPS_FUR)
    return x, y


class CriticNetwork(nn.Module):
    def __init__(self, lr, batch_size, n_actions, fc1_dims=256,
                 fc2_dims=256, name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()

        self.input_shape = get_conv_input_shape()
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.net_name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name + '_sac')

        # Convolution layer for Waiting Bay:
        # Dim 1: Len of Waiting Bay
        # Dim 2: 1
        # Dim 3: Num of maps - availability and slab thickness
        # Dim 4: Batch size
        self.conv_wb = nn.Sequential(
            nn.Conv1d(IPS.N_MAPS_WB, 8, kernel_size=2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 16, kernel_size=2),
            nn.MaxPool1d(kernel_size=2)
        )

        # Convolution layer for Furnaces:
        # Dim 1: Height of Furnace
        # Dim 2: Width of Furnace
        # Dim 3: Num of maps - Num of furnaces and total maps considered for each
        # Dim 4: Batch size
        self.conv_fur = nn.Sequential(
            nn.Conv2d(IPS.n_furnaces * IPS.N_MAPS_FUR, 16, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(3, 3))
        )

        # Action:
        # Dim 1: All different actions - [a_1, a_2, a_3]
        # Dim 2: Batch size. Ex: [[0, 1, 2], [1, 1, 2], [3, 1, 2], [0, 1, 1] .... [0, 2, 1]]

        # Fully connected layer:
        self.fc = nn.Sequential(
            nn.Linear(self.get_conv_out() + self.n_actions, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0 ' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: np.ndarray, action: List[float]):
        # Apply Convolution:
        # Return a flattened results with two dims: a batch size and all the params from conv as one long vector
        conv_out_wb = torch.flatten(self.conv_wb(state[0]).view(state[0].size()[0], -1))
        conv_out_fur = torch.flatten(self.conv_fur(state[1]).view(state[1].size()[0], -1))
        roll_mill_time = torch.tensor(state[2])

        # Pass the long vector batches to the fully connected layers for Q values for every batch input
        action = torch.tensor(action)

        conv_out = torch.cat([conv_out_wb, conv_out_fur, roll_mill_time, action], 1)

        # Apply FFN:
        q_value = self.fc(conv_out)  # q_values for the batch [1, 1, batch_size]
        return q_value

    def get_conv_out(self, batch_size):
        input_shape_1 = self.conv_wb(torch.zeros([IPS.WAIT_BAY_LEN, 1, IPS.N_MAPS_WB, batch_size]))
        input_shape_2 = self.conv_fur(torch.zeros([max(IPS.FURNACE_HEIGHT), max(IPS.FURNACE_WIDTH),
                                                   IPS.n_furnaces * IPS.N_MAPS_FUR, batch_size]))

        conv_out_wb = torch.flatten(self.conv_wb(torch.zeros(input_shape_1)).view(input_shape_1.size()[0], -1))
        conv_out_fur = torch.flatten(self.conv_fur(torch.zeros(input_shape_2)).view(input_shape_2.size()[0], -1))
        roll_time = torch.zeros([1, self.batch_size])

        # Pass the long vector batches to the fully connected layers for Q values for every batch input
        input_shape_3 = torch.zeros([self.n_actions, batch_size])
        action = torch.flatten(input_shape_3.view(input_shape_3.size()[0], -1))  # A = {a_1, a_2, a_3}

        conv_out_1d = torch.cat([conv_out_wb, conv_out_fur, roll_time, action], 1)
        return conv_out_1d.size(dim=1)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, lr, action_space_size, num_actions, batch_size,
                 fc_dims=256, n_actions=3, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.lr = lr
        self.input_dims = get_conv_input_shape()
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.fc_dims = fc_dims
        self.n_actions = n_actions
        self.action_space_size = action_space_size
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + 'sac')

        # Convolution layer for Waiting Bay:
        # Dim 1: Len of Waiting Bay
        # Dim 2: 1
        # Dim 3: Num of maps - availability and slab thickness
        # Dim 4: Batch size
        self.conv_wb = nn.Sequential(
            nn.Conv1d(IPS.N_MAPS_WB, 8, kernel_size=2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 16, kernel_size=2),
            nn.MaxPool1d(kernel_size=2)
        )

        # Convolution layer for Furnaces:
        # Dim 1: Height of Furnace
        # Dim 2: Width of Furnace
        # Dim 3: Num of maps - Num of furnaces and total maps considered for each
        # Dim 4: Batch size
        self.conv_fur = nn.Sequential(
            nn.Conv2d(IPS.n_furnaces * IPS.N_MAPS_FUR, 16, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(3, 3))
        )

        # LSTM layer for sequential action prediction:
        # Input sequence length: Output of convolution + n_actions
        self.lstm_layer = nn.LSTM([self.get_conv_out() + self.n_actions], self.fc_dims)

        # Output through Linear and Softmax:
        self.lin_actions = nn.Sequential(nn.Linear(self.fc_dims, self.action_space_size),
                                         nn.ReLU(), nn.Softmax(dim=1))

    def forward(self, state) -> torch.tensor(List[float]):
        # Apply Convolution:
        # Return a flattened results with two dims: a batch size and all the params from conv as one long vector
        conv_out_wb = torch.flatten(self.conv_wb(state[0]).view(state[0].size()[0], -1))
        conv_out_fur = torch.flatten(self.conv_fur(state[1]).view(state[1].size()[0], -1))
        roll_time = torch.tensor(state[2])

        # Initialise LSTM:
        h_0 = torch.zeros(1, self.batch_size, self.fc_dims)
        c_0 = torch.zeros(1, self.batch_size, self.fc_dims)
        input_st = torch.cat([conv_out_wb, conv_out_fur, roll_time], 1)

        # 1st forward pass
        input = torch.cat([input_st, torch.zeros([self.n_actions, self.batch_size])])
        _, (h_1, c_1) = self.lstm(input, (h_0, c_0))
        a1_probabilities = self.lin_actions(h_1)
        action_1 = torch.argmax(a1_probabilities)

        # 2nd forward pass
        input_a1 = torch.cat([input_st, action_1, torch.zeros([self.n_actions - 1, self.batch_size])])
        _, (h_2, c_2) = self.lstm(input_a1, (h_1, c_1))
        a2_probabilities = self.lin_actions(h_2)
        action_2 = torch.argmax(a2_probabilities)

        # 3rd forward pass
        input_a2 = torch.cat([input_st, action_1, action_2, torch.zeros([self.n_actions - 2, self.batch_size])])
        output_3, (h_3, _) = self.lstm(input_a2, (h_2, c_2))
        a3_probabilities = self.lin_actions(h_3)
        action_3 = torch.argmax(a3_probabilities)

        action_probabilities = torch.matmul(torch.matmul(a1_probabilities, a2_probabilities), a3_probabilities)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities, [action_1.numpy(), action_2.numpy(), action_3.numpy()]

    def get_conv_out(self, batch_size):
        input_shape_1 = self.conv_wb(torch.zeros([IPS.WAIT_BAY_LEN, 1, IPS.N_MAPS_WB, batch_size]))
        input_shape_2 = self.conv_fur(torch.zeros([max(IPS.FURNACE_HEIGHT), max(IPS.FURNACE_WIDTH),
                                                   IPS.n_furnaces * IPS.N_MAPS_FUR, batch_size]))

        conv_out_wb = torch.flatten(self.conv_wb(torch.zeros(input_shape_1)).view(input_shape_1.size()[0], -1))
        conv_out_fur = torch.flatten(self.conv_fur(torch.zeros(input_shape_2)).view(input_shape_2.size()[0], -1))
        roll_time = torch.zeros([1, self.batch_size])

        conv_out_1d = torch.cat([conv_out_wb, conv_out_fur, roll_time], 1)
        return conv_out_1d.size(dim=1)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent():
    def __init__(self, env=None, gamma=IPS.DISCOUNT_FACTOR, n_actions=3,
                 max_size=IPS.L_REPLAY_BUFFER, tau=IPS.SMOOTHING_RATE,
                 reward_scale=2):
        self.gamma = gamma
        self.n_actions = n_actions
        self.max_size = max_size
        self.replay_memory = sc.ExperienceReplay()
        self.tau = tau
        self.reward_scale = reward_scale

        # Networks:
        self.actor = ActorNetwork(lr=IPS.LR_ACTOR, batch_size=IPS.N_BATCHES, name='actor',
                                  action_space_size=IPS.WAIT_BAY_LEN, n_actions=n_actions)
        self.critic1 = CriticNetwork(lr=IPS.LR_CRITIC, batch_size=IPS.N_BATCHES, n_actions=3,
                                     name='critic_actual1')
        self.critic2 = CriticNetwork(lr=IPS.LR_CRITIC, batch_size=IPS.N_BATCHES, n_actions=3,
                                     name='critic_actual2')
        self.critic_target1 = CriticNetwork(lr=IPS.LR_CRITIC, batch_size=IPS.N_BATCHES, n_actions=3,
                                            name='critic_target1')
        self.critic_target2 = CriticNetwork(lr=IPS.LR_CRITIC, batch_size=IPS.N_BATCHES, n_actions=3,
                                            name='critic_target2')

        # Temperature parameter computations
        self.target_entropy = 0.98 * -np.log(1 / (IPS.WAIT_BAY_LEN+2))  # Based on num of actions
        self.log_alpha = torch.tensor(np.log(1.0), requires_grad=True)
        self.alpha = self.log_alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=IPS.LR_ENTROPY)

    def choose_action(self, state):
        state = torch.Tensor([state]).to(self.actor.device)
        _, _, actions = self.actor.forward(state)
        actions_array = []
        for action in actions:
            actions_array.append(action.cpu().detach().numpy())
        return np.array(actions_array)

    def remember(self, state, action, reward, new_state, done):
        state_obj = sc.State()
        state_obj.state = state
        action_obj = sc.Action(list(action))
        reward_obj = sc.Reward(reward)
        new_state_obj = sc.State()
        new_state_obj.state = new_state
        exp_sample = tuple([sc.Experience(state_obj, action_obj, reward_obj, new_state_obj, done)])

        self.replay_memory.add_elements(exp_sample)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        else:
            critic_target1_params = self.critic_target1.named_parameters()
            critic_target2_params = self.critic_target2.named_parameters()
            critic1_params = self.critic1.named_parameters()
            critic2_params = self.critic2.named_parameters()

            critic_target1_state_dict = dict(critic_target1_params)
            critic_target2_state_dict = dict(critic_target2_params)

            critic1_state_dict = dict(critic1_params)
            critic2_state_dict = dict(critic2_params)

            for name in critic1_state_dict:
                critic1_state_dict[name] = tau*critic1_state_dict[name].clone() + \
                                           (1-tau)*critic_target1_state_dict[name].clone()
            self.critic_target1.load_state_dict(critic1_state_dict)

            for name in critic2_state_dict:
                critic2_state_dict[name] = tau*critic2_state_dict[name].clone() + \
                                           (1-tau)*critic_target2_state_dict[name].clone()
            self.critic_target2.load_state_dict(critic2_state_dict)

    def save_models(self):
        print('.....saving models.....')
        self.actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic_target1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.critic_target2.save_checkpoint()

    def load_models(self):
        print('.....loading models.....')
        self.actor.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic_target1.load_checkpoint()
        self.critic2.load_checkpoint()
        self.critic_target2.load_checkpoint()

    def train_networks(self, experience: sc.Experience):
        # Set all the gradients stored in the optimisers to zero.
        self.critic1.optimiser.zero_grad()
        self.critic2.optimiser2.zero_grad()
        self.actor.optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()

        # Calculate the loss for this transition.
        self.replay_memory.add_elements(experience)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        if len(self.replay_memory) >= IPS.N_BATCHES:
            # get minibatch of 100 transitions from replay buffer
            states, actions, rewards, new_states, dones = self.replay_memory.sample_elements_arrays(IPS.N_BATCHES)

            # unravel transitions to get states, actions, rewards and next states
            states_tensor = torch.tensor(states)
            actions_tensor = torch.tensor(actions)
            rewards_tensor = torch.tensor(rewards).float()
            next_states_tensor = torch.tensor(new_states)
            done_tensor = torch.tensor(dones)

            critic1_loss, critic2_loss = \
                self.critic_loss(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor)

            critic1_loss.backward()
            critic2_loss.backward()
            self.critic1.optimiser.step()
            self.critic2.optimiser.step()

            actor_loss, log_action_probabilities = self.actor_loss(states_tensor)

            actor_loss.backward()
            self.actor.optimiser.step()

            alpha_loss = self.temperature_loss(log_action_probabilities)

            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

            self.update_network_parameters()

    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor):
        with torch.no_grad():
            action_probabilities, log_action_probabilities, _ = self.actor.forward(next_states_tensor)
            next_q_values_target1 = self.critic_target1.forward(next_states_tensor)
            next_q_values_target2 = self.critic_target2.forward(next_states_tensor)
            soft_state_values = (action_probabilities * (
                    torch.min(next_q_values_target1, next_q_values_target2) - self.alpha * log_action_probabilities
            )).sum(dim=1)

            next_q_values = rewards_tensor + ~done_tensor * IPS.DISCOUNT_FACTOR*soft_state_values

        soft_q_values1 = self.critic1(states_tensor, actions_tensor)
        soft_q_values2 = self.critic2(states_tensor, actions_tensor)

        critic1_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values1, next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values)

        critic1_loss = critic1_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic1_loss, critic2_loss

    def actor_loss(self, states_tensor):
        action_probabilities, log_action_probabilities, actions_list = self.actor.forward(states_tensor)
        q_values1 = self.critic1(states_tensor, torch.tensor(actions_list))
        q_values2 = self.critic2(states_tensor)
        inside_term = self.alpha * log_action_probabilities - torch.min(q_values1, q_values2)
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss