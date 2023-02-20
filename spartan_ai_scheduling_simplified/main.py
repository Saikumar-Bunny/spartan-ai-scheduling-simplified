import numpy as np
import os
import warnings
import torch
import torch.optim as optim
import torch.nn as nn
from collections import deque
import matplotlib.pyplot as plt

from spartan_ai_scheduling_simplified import environment_simple
from spartan_ai_scheduling_simplified import helpers
from spartan_ai_scheduling_simplified import INIT_PARAMS_SIMPLE as IPS
from spartan_ai_scheduling_simplified import MultiDiscreteSAC as md_sac
from spartan_ai_scheduling_simplified import PhysicalStorageClasses as psc
from spartan_ai_scheduling_simplified import storage_classes as sc

import time

start = time.time()

# Unset any warnings
warnings.filterwarnings('ignore')

# Set the Random Number Generator seed - For reproducibility
helpers.seed_torch(IPS.SEED)

# TRAINING
# Initiate the waiting bay and replay buffer
replay_buffer = sc.ExperienceReplay()

# Instantiate all the networks
critic_actual1 = md_sac.CriticNetwork(lr=IPS.LR_CRITIC, batch_size=IPS.N_BATCHES, n_actions=3, name='critic_actual1')
critic_actual2 = md_sac.CriticNetwork(lr=IPS.LR_CRITIC, batch_size=IPS.N_BATCHES, n_actions=3, name='critic_actual2')
actor_net = md_sac.ActorNetwork(lr=IPS.LR_ACTOR, batch_size=IPS.N_BATCHES,
                                action_space_size=IPS.WAIT_BAY_LEN, n_actions=3)

critic_target1 = md_sac.CriticNetwork(lr=IPS.LR_CRITIC, batch_size=IPS.N_BATCHES, n_actions=3, name='critic_target1')
critic_target2 = md_sac.CriticNetwork(lr=IPS.LR_CRITIC, batch_size=IPS.N_BATCHES, n_actions=3, name='critic_target2')

# Waiting bay:
waiting_bay = psc.WaitingBay(IPS.WAIT_BAY_LEN)

# Furnaces:
furnaces = []
for i in range(1, IPS.n_furnaces+1):
    furnaces.append(psc.Furnace(i))

# Instantiate the environment of the mill:
env = environment_simple.ReheatFurnaceEnvironment(wait_bay=waiting_bay, furnaces=furnaces)

# Load previous model, if available:
# TODO Ticket 5: Yet to make use of loading saved models feature.
#       Because we have multiple networks - Soft Q Network or (Critic) and its Target,
#       multiple Q nets to reduce over prediction of value which are all to be loaded.
#       Policy Network (Actor) to predict the action given state.
#       Entropy network to optimise the temperature parameter.
# if IPS.USE_PREV_MODEL:
#     network.load_state_dict(torch.load(IPS.MODEL))

# Iterate over all the episodes
if not IPS.SKIP_TRAINING:
    print('Episode, Average Reward, Average NN Loss\n')
    global_timer = 0
    for episode in range(IPS.NUM_EPISODES):
        labels = torch.zeros(1)
        targets = torch.zeros(1)

        # Reset the environment and initiate the scoring-variables
        state = env.reset()

        reward = sc.Reward(0.0)
        done = False
        action, new_state = [], []
        reward_vals = []

        # Iterate over the time-steps
        while not done:
            experience_list = []

            # Select the action value - explore or exploit
            if not done:
                # TODO: Get the state as a image type array with both waiting buffer and furnaces live info:
                state = env.state

            if len(replay_buffer) >= IPS.L_REPLAY_BUFFER:
                pass

            # Calculate the output with the step action
            state, reward, new_state, done, info = env.step(action)

            # Gather as observation and append to Replay Buffer
            experience = sc.Experience(state, sc.Action(action), reward, new_state, done)
            replay_buffer.add_elements(tuple(experience))

            # Value function - targets are the reward, labels are current values
            targets = reward

            # End the episode
            if done:
                break

            # Step to the new state
            state = new_state
            reward_vals.append(reward)

        # Check for early stopping - Stop if the last 100 scores have crossed the SOLVED_SCORE value

        # Calculate Gt (cumulative discounted rewards)
        reward = helpers.process_rewards(reward_vals)
        loss = loss_fn(labels, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy())

    env.close()
    end = time.time()
    print(f'Completed training in {round(end - start, 2)} seconds \n')

else:
    print('Skipped the training, using the pre-saved model \n')