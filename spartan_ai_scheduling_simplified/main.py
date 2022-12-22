import numpy as np
import os
import warnings
import torch
import torch.optim as optim
import torch.nn as nn
from collections import deque
import matplotlib.pyplot as plt

from spartan_ai_scheduling_simplified import INIT_PARAMS_SIMPLE as IPS
from spartan_ai_scheduling_simplified import helpers
from spartan_ai_scheduling_simplified import environment_simple

import time

start = time.time()

# Unset any warnings
warnings.filterwarnings('ignore')

# Set the Random Number Generator seed - For reproducibility
helpers.seed_torch(IPS.SEED)

# TRAINING
# Instantiate the environment for the particle
env = environment_simple.FurnaceEnvironment()

# Instantiate the network in which we train the policy
network = helpers.PolicyNetwork(env.observation_space.shape[0]).to(helpers.DEVICE)
# network = torch.jit.load(INIT_PARAMS.model_save_name)

# Instantiate the Optimizer - set it to adaptive-momentum optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=IPS.LR)

# Track the scores and losses as a list
scores, losses = [], []
plot_score, plot_loss = [], []

# Store the last the 100 scores only and average for mean reward
recent_scores = deque(maxlen=100)

# Delete the old log files if any
if os.path.exists(IPS.LOG_FILE):
    os.remove(IPS.LOG_FILE)

# Iterate over all the episodes
if not IPS.SKIP_TRAINING:
    print('1. Training the policy network\n')
    print('Episode, Reward score, Current state\n')
    for episode in range(IPS.NUM_EPISODES):
        # Reset the environment and initiate the scoring-variables
        if episode % IPS.TEST_SAMPLES == 0:
            env.data_counter = 0
        state = env.reset()
        rewards = []
        targets, labels = [], []
        score = 0
        done = False

        # Iterate over the time-steps
        for step in range(IPS.TOTAL_STEPS):
            # env.render()
            state = env.new_product()
            # Select the action value - Squeeze it to [-1, 1]
            exploration = np.random.random() < 1 - (1-IPS.EPSILON) * episode/IPS.NUM_EPISODES

            if exploration:
                action = env.action_space.sample()
            else:
                val = helpers.select_action(network, state)
                values = val.detach().numpy()
                if values[0] <= values[1]:
                    action = 0
                else:
                    action = 1

            # Calculate the output with the step action
            new_state, reward, done, _ = env.step(action)
            reward = reward/IPS.REWARD_NRMLZ_CONSTANT

            if action == 0:
                targets.append([reward, values[1]])
                labels.append([values[0], values[1]])
            else:
                targets.append([values[0], reward])
                labels.append([values[0], values[1]])

            # Track the episode score
            score += reward

            # Store reward and log probability
            rewards.append(reward)

            # End the episode
            if done:
                break

            # Step to the new state
            state = new_state

        # Attach the score
        scores.append(score)
        recent_scores.append(score)

        # Check for early stopping - Stop if the last 100 scores have crossed the SOLVED_SCORE value
        # if np.array(recent_scores).mean() >= IPS.SOLVED_SCORE and len(recent_scores) >= 100:
        #     break

        # Calculate Gt (cumulative discounted rewards)
        rewards = helpers.process_rewards(rewards)
        loss = loss_fn(torch.tensor(labels[0], dtype=torch.double, requires_grad=True),
                       torch.tensor(targets[0], dtype=torch.double, requires_grad=True))
        loss = loss.double()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # for param in network.parameters():
        #     print(f'{episode} \n')
        #     print(param.data)
        losses.append(loss.detach().numpy())

        if episode % 50 == 0:
            print(f'Episode: {episode}, Cumu-reward: {round(score,2)}')
            plot_score.append(np.mean(scores))
            plot_loss.append(np.mean(losses))
            print(f'{np.mean(losses)} loss \n')
            losses, scores = [], []

    env.close()
    end = time.time()
    print(f'Completed training in {round(end - start, 2)} seconds \n')

    # Save the model parameters
    model_scripted = torch.save(network.state_dict(), IPS.MODEL)       # Export to TorchScript
    # model_scripted.save(IPS.MODEL) # Save

    # Plot the episodes vs score and save the image
    graphed_array = plot_score
    fig = plt.figure
    plt.plot(graphed_array)
    plt.ylabel('Scores')
    plt.xlabel('Episodes')
    plt.title('Training scores')
    plt.savefig('Training_RewardsVsEpisodes.jpg')
    plt.close()
else:
    print('Skipped the training, using the pre-saved model \n')

# if __name__ == '__main__':
