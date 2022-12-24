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
network = helpers.PolicyNetwork().to(helpers.DEVICE)

if IPS.USE_PREV_MODEL:
    network.load_state_dict(torch.load(IPS.MODEL))

# network = torch.jit.load(INIT_PARAMS.model_save_name)

# Instantiate the Optimizer - set it to adaptive-momentum optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(network.parameters(), lr=IPS.LR)

# Track the scores and losses as a list
scores, losses = [], []
plot_score, plot_loss = [], []
prev_lowest = 10000

# Store the last the 100 scores only and average for mean reward
recent_scores = deque(maxlen=100)

# Delete the old log files if any
if os.path.exists(IPS.LOG_FILE):
    os.remove(IPS.LOG_FILE)

# Iterate over all the episodes
if not IPS.SKIP_TRAINING:
    print('Episode, Average Reward, Average NN Loss\n')
    for episode in range(IPS.NUM_EPISODES):
        labels = torch.zeros(1)
        targets = torch.zeros(1)
        # Reset the environment and initiate the scoring-variables
        if episode % IPS.TEST_SAMPLES == 0:
            env.data_counter = 0
        state = env.reset()
        rewards = []
        score = 0
        done = False

        # Iterate over the time-steps
        for step in range(IPS.TOTAL_STEPS):
            state = env.new_product()

            # Find the expected rewards for all different actions
            val = torch.zeros(IPS.n_furnaces)
            for i in range(IPS.n_furnaces):
                val[i] = helpers.select_action(network, state,
                                               action=0 + i * (1 - 0) / (IPS.n_furnaces - 1))  # E[R]_a=i | s

            # Select the action value - explore or exploit
            exploration = np.random.random() < (IPS.EPSILON if IPS.USE_PREV_MODEL else 1 - (1 - IPS.EPSILON) \
                                                                                      * episode / IPS.NUM_EPISODES)

            if exploration:
                action = env.action_space.sample()
            else:
                action = torch.argmin(val)

            # Calculate the output with the step action
            new_state, reward, done, _ = env.step(action)
            reward = reward / IPS.REWARD_NRMLZ_CONSTANT

            # Value function - targets the reward, labels are current values
            targets[0] = reward
            labels[0] = torch.min(val)

            # Track the episode score and rewards
            score += reward
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
        loss = loss_fn(labels, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy())

        if episode % IPS.EPISODE_SAVE_TRIGGER == 0 and episode != 0:
            print(f'recent loss: {loss.detach().numpy()}')
            print(f'{episode}, \t '
                  f'{IPS.REWARD_NRMLZ_CONSTANT * np.mean(scores)}, \t '
                  f'{np.mean(losses)}')
            plot_score.append(np.mean(scores))
            plot_loss.append(np.mean(losses))
            if (np.mean(scores) < prev_lowest) and episode > 50000:
                # Override the model parameters
                print('Better model --> saved')
                prev_lowest = np.mean(scores)
                torch.save(network.state_dict(), IPS.MODEL)  # Export to TorchScript
            losses, scores = [], []

    env.close()
    end = time.time()
    print(f'Completed training in {round(end - start, 2)} seconds \n')

    # Plot the episodes vs score and save the image
    graphed_array = plot_score
    fig = plt.figure
    plt.plot(graphed_array)
    plt.ylabel('Scores')
    plt.xlabel('Episodes')
    plt.title('Training scores')
    plt.savefig(IPS.EPOCH_VS_REWARD_FIG)
    plt.close()
else:
    print('Skipped the training, using the pre-saved model \n')

# if __name__ == '__main__':
