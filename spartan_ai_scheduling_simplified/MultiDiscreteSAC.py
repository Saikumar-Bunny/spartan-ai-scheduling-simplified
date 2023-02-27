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
    x = 1 + 1 + max(IPS.FURNACE_WIDTH[1:])
    y = max(IPS.WAIT_BAY_LEN, IPS.n_furnaces * max(IPS.FURNACE_HEIGHT[1:]) * IPS.N_MAPS_FUR)
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
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        # Convolution layer for Waiting Bay:
        # Dim 3: Len of Waiting Bay
        # Dim 2: Num of maps - availability and slab thickness
        # Dim 1: Batch size
        self.conv_wb = nn.Sequential(
            nn.Conv1d(IPS.N_MAPS_WB, 8, kernel_size=2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 16, kernel_size=2),
            nn.MaxPool1d(kernel_size=2)
        )

        # Convolution layer for Furnaces:
        # Dim 4: Height of Furnace
        # Dim 3: Width of Furnace
        # Dim 2: Num of maps - Num of furnaces and total maps considered for each
        # Dim 1: Batch size
        self.conv_fur = nn.Sequential(
            nn.Conv2d(IPS.n_furnaces * IPS.N_MAPS_FUR, 16, kernel_size=(2, 2), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(2, 2), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # Action:
        # Dim 1: All different actions - [a_1, a_2, a_3]
        # Dim 2: Batch size. Ex: [[0, 1, 2], [1, 1, 2], [3, 1, 2], [0, 1, 1] .... [0, 2, 1]]

        # Fully connected layer:
        self.fc = nn.Sequential(
            nn.Linear(self.get_conv_out(), self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, 1),
        )
        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0 ' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.double()

    def forward(self, state_arr: List[np.ndarray], action: np.ndarray) -> np.ndarray:

        if len(state_arr[0].shape) != 3:   # Batch info unavailable
            state_wb = torch.tensor(state_arr[0], requires_grad=True,
                                    dtype=torch.float32).unsqueeze(-1).transpose(1, 2).transpose(0, 1)
            state_fur = torch.tensor(state_arr[1], requires_grad=True,
                                     dtype=torch.float32).unsqueeze(-1).transpose(2, 3).transpose(1, 2).transpose(0, 1)
            state_rolltime = torch.tensor([state_arr[2]], requires_grad=True,
                            dtype=torch.float32).unsqueeze(-1)
        else:
            state_wb = torch.tensor(state_arr[0], requires_grad=True, dtype=torch.float32)
            state_fur = torch.tensor(state_arr[1], requires_grad=True, dtype=torch.float32)
            state_rolltime = torch.tensor([state_arr[2]], requires_grad=True, dtype=torch.float32)
        # Apply Convolution:
        # Return a flattened results with two dims: a batch size and all the params from conv as one long vector
        conv_out_wb = self.conv_wb(state_wb).view(state_wb.shape[0], -1)
        conv_out_fur = self.conv_fur(state_fur).view(state_fur.shape[0], -1)
        roll_mill_time = torch.tensor(state_rolltime)

        # Pass the long vector batches to the fully connected layers for Q values for every batch input
        action = torch.tensor(action)

        conv_out = torch.cat([conv_out_wb, conv_out_fur, roll_mill_time, action], 1)

        # Apply FFN:
        q_value = self.fc(conv_out)  # q_values for the batch [1, 1, batch_size]
        return q_value

    def get_conv_out(self):
        with torch.no_grad():
            input_shape_1 = self.conv_wb(torch.zeros([self.batch_size, IPS.N_MAPS_WB, IPS.WAIT_BAY_LEN]))
            input_shape_2 = self.conv_fur(torch.zeros([self.batch_size, IPS.n_furnaces * IPS.N_MAPS_FUR,
                                                       max(IPS.FURNACE_HEIGHT[1:]), max(IPS.FURNACE_WIDTH[1:])]))

            conv_out_wb = input_shape_1.view(input_shape_1.shape[0], -1)
            conv_out_fur = input_shape_2.view(input_shape_2.shape[0], -1)
            roll_time = torch.zeros([self.batch_size, 1])

            # Pass the long vector batches to the fully connected layers for Q values for every batch input
            action_shape = torch.zeros([self.batch_size, self.n_actions])

            conv_out_1d = torch.cat([conv_out_wb, conv_out_fur, roll_time, action_shape], 1)
        return conv_out_1d.size(dim=1)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, lr, action_space_size, batch_size,
                 fc_dims=128, n_actions=3, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.lr = lr
        self.input_dims = get_conv_input_shape()
        self.num_actions = n_actions
        self.batch_size = batch_size
        self.fc_dims = fc_dims
        self.n_actions = n_actions
        self.action_space_size = action_space_size
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + 'sac')

        # Convolution layer for Waiting Bay:
        # Dim 3: Len of Waiting Bay
        # Dim 2: Num of maps - availability and slab thickness
        # Dim 1: Batch size
        self.conv_wb = nn.Sequential(
            nn.Conv1d(IPS.N_MAPS_WB, 8, kernel_size=2),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 16, kernel_size=2),
            nn.MaxPool1d(kernel_size=2)
        )

        # Convolution layer for Furnaces:
        # Dim 4: Height of Furnace
        # Dim 3: Width of Furnace
        # Dim 2: Num of maps - Num of furnaces and total maps considered for each
        # Dim 1: Batch size
        self.conv_fur = nn.Sequential(
            nn.Conv2d(IPS.n_furnaces * IPS.N_MAPS_FUR, 16, kernel_size=(2, 2), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(2, 2), padding=1),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # LSTM layer for sequential action prediction:
        # Input sequence length: Output of convolution + n_actions
        self.lstm_layer = nn.LSTM(1, hidden_size=self.fc_dims, batch_first=True)

        # Output through Linear and Softmax:
        self.lin_actions = nn.Sequential(nn.Linear(self.fc_dims, self.action_space_size),
                                         nn.ReLU(), nn.Softmax(dim=2)) #  Softmax applied on the probs in dim2

        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0 ' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.double()

    def forward(self, state_arr: List[np.ndarray]) -> torch.tensor:
        if len(state_arr[0].shape) != 3:   # Batch info unavailable
            state_wb = torch.tensor(state_arr[0], requires_grad=True,
                                    dtype=torch.float32).unsqueeze(-1).transpose(1, 2).transpose(0, 1)
            state_fur = torch.tensor(state_arr[1], requires_grad=True,
                                     dtype=torch.float32).unsqueeze(-1).transpose(2, 3).transpose(1, 2).transpose(0, 1)
            state_rolltime = torch.tensor([state_arr[2]], requires_grad=True,
                            dtype=torch.float32).unsqueeze(-1)
        else:
            state_wb = torch.tensor(state_arr[0], requires_grad=True, dtype=torch.float32)
            state_fur = torch.tensor(state_arr[1], requires_grad=True, dtype=torch.float32)
            state_rolltime = torch.tensor([state_arr[2]], requires_grad=True, dtype=torch.float32)

        # Apply Convolution:
        # Return a flattened results with two dims: a batch size and all the params from conv as one long vector
        conv_out_wb = self.conv_wb(state_wb).view(state_wb.shape[0], -1)
        conv_out_fur = self.conv_fur(state_fur).view(state_fur.shape[0], -1)

        # Initialise LSTM:
        h_0 = torch.zeros([1, self.batch_size, self.fc_dims])
        c_0 = torch.zeros([1, self.batch_size, self.fc_dims])
        input_st = torch.cat([conv_out_wb, conv_out_fur, state_rolltime], 1)
        action_space_init = torch.zeros([self.batch_size, self.n_actions])

        # 1st forward pass
        input = torch.cat([input_st, action_space_init], dim=1).unsqueeze(-1)
        _, (h_1, c_1) = self.lstm_layer(input, (h_0, c_0))
        # h_1 dims: # dim 0: output layer - 1 in our case;
        # dim 1: batch_size
        # dim 2: hidden layer output size - fc_dims - 128
        a1_probabilities = self.lin_actions(h_1)
        action_1 = torch.argmax(a1_probabilities, dim=2)    # A single number across each batch - argmax on dim2

        # 2nd forward pass
        input_a1 = torch.cat([input_st, action_1, torch.zeros([self.batch_size, self.n_actions - 1])], dim=1).unsqueeze(-1)
        _, (h_2, c_2) = self.lstm_layer(input_a1, (h_1, c_1))
        a2_probabilities = self.lin_actions(h_2)
        action_2 = torch.argmax(a2_probabilities, dim=2)

        # 3rd forward pass
        input_a2 = torch.cat([input_st, action_1, action_2, torch.zeros([self.batch_size, self.n_actions - 2])], dim=1).unsqueeze(-1)
        output_3, (h_3, _) = self.lstm_layer(input_a2, (h_2, c_2))
        a3_probabilities = self.lin_actions(h_3)
        action_3 = torch.argmax(a3_probabilities, dim=2)

        action_probabilities = torch.mul(torch.mul(a1_probabilities, a2_probabilities), a3_probabilities)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)  # Has the shape [1, batch_size, actionspace_size]
        return action_probabilities[0], log_action_probabilities[0], [action_1.numpy(), action_2.numpy(), action_3.numpy()]

    def get_conv_out(self):
        with torch.no_grad():
            input_shape_1 = self.conv_wb(torch.zeros([self.batch_size, IPS.N_MAPS_WB, IPS.WAIT_BAY_LEN]))
            input_shape_2 = self.conv_fur(torch.zeros([self.batch_size, IPS.n_furnaces * IPS.N_MAPS_FUR,
                                                       max(IPS.FURNACE_HEIGHT[1:]), max(IPS.FURNACE_WIDTH[1:])]))
            conv_out_wb = input_shape_1.view(input_shape_1.shape[0], -1)
            conv_out_fur = input_shape_2.view(input_shape_2.shape[0], -1)
            roll_time = torch.zeros([self.batch_size, 1])

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
                                  action_space_size=IPS.WAIT_BAY_LEN, n_actions=self.n_actions)
        self.critic1 = CriticNetwork(lr=IPS.LR_CRITIC, batch_size=IPS.N_BATCHES, n_actions=self.n_actions,
                                     name='critic_actual1')
        self.critic2 = CriticNetwork(lr=IPS.LR_CRITIC, batch_size=IPS.N_BATCHES, n_actions=self.n_actions,
                                     name='critic_actual2')
        self.critic_target1 = CriticNetwork(lr=IPS.LR_CRITIC, batch_size=IPS.N_BATCHES, n_actions=self.n_actions,
                                            name='critic_target1')
        self.critic_target2 = CriticNetwork(lr=IPS.LR_CRITIC, batch_size=IPS.N_BATCHES, n_actions=self.n_actions,
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
        exp_sample = sc.Experience(state_obj, action_obj, reward_obj, new_state_obj, done)

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
        self.critic2.optimiser.zero_grad()
        self.actor.optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()

        # Calculate the loss for this transition.
        self.replay_memory.add_elements(experience)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        if len(self.replay_memory) >= IPS.N_BATCHES:
            # get minibatch of 100 transitions from replay buffer
            states_list, actions_arr, rewards_arr, new_states_list, dones_arr = \
                self.replay_memory.sample_elements(IPS.N_BATCHES)

            critic1_loss, critic2_loss = \
                self.critic_loss(states_list, actions_arr, rewards_arr, new_states_list, dones_arr)

            critic1_loss.backward()
            critic2_loss.backward()
            self.critic1.optimiser.step()
            self.critic2.optimiser.step()

            actor_loss, log_action_probabilities = self.actor_loss(states_list)

            actor_loss.backward()
            self.actor.optimiser.step()

            alpha_loss = self.temperature_loss(log_action_probabilities)

            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

            self.update_network_parameters()

    def critic_loss(self, states_list, actions_arr, rewards_arr, new_states_list, dones_arr):
        with torch.no_grad():
            rewards_tensor = torch.tensor(rewards_arr)
            dones_tensor = torch.tensor(dones_arr)
            action_probabilities, log_action_probabilities, _ = self.actor.forward(new_states_list)
            next_q_values_target1 = torch.tensor(self.critic_target1.forward(new_states_list, actions_arr), dtype=torch.double)
            next_q_values_target2 = torch.tensor(self.critic_target2.forward(new_states_list, actions_arr), dtype=torch.double)
            soft_state_values = (action_probabilities * (
                    torch.min(next_q_values_target1, next_q_values_target2) - self.alpha * log_action_probabilities
            )).sum(dim=1)

            next_q_values = rewards_tensor + ~dones_tensor * IPS.DISCOUNT_FACTOR*soft_state_values

        soft_q_values1 = self.critic1(states_list, actions_arr)
        soft_q_values2 = self.critic2(states_list, actions_arr)

        critic1_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values1, next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values)

        critic1_loss = critic1_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic1_loss, critic2_loss

    def actor_loss(self, states_list):
        with torch.no_grad:
            action_probabilities, log_action_probabilities, actions_list = self.actor.forward(states_list)
            q_values1 = self.critic1(states_list, torch.tensor(actions_list))
            q_values2 = self.critic2(states_list, torch.tensor(actions_list))
            inside_term = self.alpha * log_action_probabilities - torch.min(q_values1, q_values2)
            policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss