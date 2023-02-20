import numpy as np
import os

import torch.cuda
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import INIT_PARAMS_SIMPLE as IPS
from typing import List, Tuple
from spartan_ai_scheduling_simplified import helpers
from spartan_ai_scheduling_simplified import storage_classes as sc


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

    def forward(self, state, action):
        # Apply Convolution:
        # Return a flattened results with two dims: a batch size and all the params from conv as one long vector
        conv_out_wb = torch.flatten(self.conv_wb(state).view(state.size()[0], -1))
        conv_out_fur = torch.flatten(self.conv_fur(state).view(state.size()[0], -1))

        # Pass the long vector batches to the fully connected layers for Q values for every batch input
        action = torch.tensor(action)

        conv_out = torch.cat([conv_out_wb, conv_out_fur, action], 1)

        # Apply FFN:
        q_value = self.fc(conv_out)  # q_values for the batch [1, 1, batch_size]
        return q_value

    def get_conv_out(self, batch_size):
        input_shape_1 = self.conv_wb(torch.zeros([IPS.WAIT_BAY_LEN, 1, IPS.N_MAPS_WB, batch_size]))
        input_shape_2 = self.conv_fur(torch.zeros([max(IPS.FURNACE_HEIGHT), max(IPS.FURNACE_WIDTH),
                                                   IPS.n_furnaces * IPS.N_MAPS_FUR, batch_size]))

        conv_out_wb = torch.flatten(self.conv_wb(torch.zeros(input_shape_1)).view(input_shape_1.size()[0], -1))
        conv_out_fur = torch.flatten(self.conv_fur(torch.zeros(input_shape_2)).view(input_shape_2.size()[0], -1))

        # Pass the long vector batches to the fully connected layers for Q values for every batch input
        input_shape_3 = torch.zeros([self.n_actions, batch_size])
        action = torch.flatten(input_shape_3.view(input_shape_3.size()[0], -1))  # A = {a_1, a_2, a_3}

        conv_out_1d = torch.cat([conv_out_wb, conv_out_fur, action], 1)
        return conv_out_1d.size(dim=1)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims=256, fc2_dims=256,
                 name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.net_name = name
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
        # Fully connected layer:
        self.fc = nn.Sequential(
            nn.Linear(self.get_conv_out(), self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # Apply Convolution:
        # Return a flattened results with two dims: a batch size and all the params from conv as one long vector
        conv_out_wb = torch.flatten(self.conv_wb(state).view(state.size()[0], -1))
        conv_out_fur = torch.flatten(self.conv_fur(state).view(state.size()[0], -1))

        conv_out = torch.cat([conv_out_wb, conv_out_fur], 1)

        # Apply FFN:
        value = self.fc(conv_out)  # values for the batch [1, 1, batch_size]
        return value

    def get_conv_out(self, batch_size):
        input_shape_1 = self.conv_wb(torch.zeros([IPS.WAIT_BAY_LEN, 1, IPS.N_MAPS_WB, batch_size]))
        input_shape_2 = self.conv_fur(torch.zeros([max(IPS.FURNACE_HEIGHT), max(IPS.FURNACE_WIDTH),
                                                   IPS.n_furnaces * IPS.N_MAPS_FUR, batch_size]))

        conv_out_wb = torch.flatten(self.conv_wb(torch.zeros(input_shape_1)).view(input_shape_1.size()[0], -1))
        conv_out_fur = torch.flatten(self.conv_fur(torch.zeros(input_shape_2)).view(input_shape_2.size()[0], -1))

        conv_out_1d = torch.cat([conv_out_wb, conv_out_fur], 1)
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
        conv_out_wb = torch.flatten(self.conv_wb(state).view(state.size()[0], -1))
        conv_out_fur = torch.flatten(self.conv_fur(state).view(state.size()[0], -1))

        # Initialise LSTM:
        h_0 = torch.zeros(1, self.batch_size, self.fc_dims)
        c_0 = torch.zeros(1, self.batch_size, self.fc_dims)
        input_st = torch.cat([conv_out_wb, conv_out_fur], 1)

        # 1st forward pass
        input = torch.cat([input_st, torch.zeros([self.n_actions, self.batch_size])])
        _, (h_1, c_1) = self.lstm(input, (h_0, c_0))
        action_1 = torch.argmax(self.lin_actions(h_1))

        # 2nd forward pass
        input_a1 = torch.cat([input_st, action_1, torch.zeros([self.n_actions - 1, self.batch_size])])
        _, (h_2, c_2) = self.lstm(input_a1, (h_1, c_1))
        action_2 = torch.argmax(self.lin_actions(h_2))

        # 3rd forward pass
        input_a2 = torch.cat([input_st, action_1, action_2, torch.zeros([self.n_actions - 2, self.batch_size])])
        output_3, (h_3, _) = self.lstm(input_a2, (h_2, c_2))
        action_3 = torch.argmax(self.lin_actions(h_3))
        return torch.tensor([action_1, action_2, action_3])

    def get_conv_out(self, batch_size):
        input_shape_1 = self.conv_wb(torch.zeros([IPS.WAIT_BAY_LEN, 1, IPS.N_MAPS_WB, batch_size]))
        input_shape_2 = self.conv_fur(torch.zeros([max(IPS.FURNACE_HEIGHT), max(IPS.FURNACE_WIDTH),
                                                   IPS.n_furnaces * IPS.N_MAPS_FUR, batch_size]))

        conv_out_wb = torch.flatten(self.conv_wb(torch.zeros(input_shape_1)).view(input_shape_1.size()[0], -1))
        conv_out_fur = torch.flatten(self.conv_fur(torch.zeros(input_shape_2)).view(input_shape_2.size()[0], -1))

        conv_out_1d = torch.cat([conv_out_wb, conv_out_fur], 1)
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

    def choose_action(self, state):
        state = torch.Tensor([state]).to(self.actor.device)
        action = self.actor.forward(state)
        return action.cpu().detach().numpy()

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

    def learn(self):
        if len(self.replay_memory) < IPS.N_BATCHES:
            return
        experience_tuple = self.replay_memory.sample_elements(sample_size=IPS.N_BATCHES)
        state, action, reward, new_state, done = helpers.process_experience(experience_tuple)

        state = torch.Tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.Tensor(action, dtype=torch.float).to(self.actor.device)
        reward = torch.Tensor(reward, dtype=torch.float).to(self.actor.device)
        new_state = torch.Tensor(new_state, dtype=torch.float).to(self.actor.device)
        done = torch.Tensor(done).to(self.actor.device)

        qval_critic1 = self.critic1.forward(state, action)
        qval_critic2 = self.critic2.forward(state, action)

        critic_value = torch.min(qval_critic1, qval_critic2)
        critic_value = critic_value.view(-1)

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss = 0.5 * F.mse_loss(critic_value, reward)
        self.critic1.backward(retain_graph=True)
        self.critic2.backward(retain_graph=True)
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        actions = self.actor.forward(state)

