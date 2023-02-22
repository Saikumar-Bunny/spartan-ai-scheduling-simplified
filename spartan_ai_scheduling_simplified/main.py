import matplotlib.pyplot as plt
import numpy as np

import INIT_PARAMS_SIMPLE as IPS
import MultiDiscreteSAC as mdsac
import PhysicalStorageClasses as psc
import environment_simple
import storage_classes as sc

TRAINING_EVALUATION_RATIO = 4
RUNS = 5
EPISODES_PER_RUN = 400
STEPS_PER_EPISODE = 200
TRAINING_SKIP = True

waiting_bay = psc.WaitingBay()
furnaces = []
for i in range(1, IPS.n_furnaces + 1):
    furnaces.append(psc.Furnace(i))


if __name__ == "__main__":
    env = environment_simple.ReheatFurnaceEnvironment(waiting_bay, furnaces)
    agent_results = []
    for run in range(RUNS):
        agent = mdsac.Agent(env)
        run_results = []
        for episode_number in range(EPISODES_PER_RUN):
            print('\r', f'Run: {run + 1}/{RUNS} | Episode: {episode_number + 1}/{EPISODES_PER_RUN}', end=' ')
            evaluation_episode = episode_number % TRAINING_EVALUATION_RATIO == 0
            episode_reward = 0
            state = env.reset()
            done = False
            i = 0
            while not done and i < STEPS_PER_EPISODE:
                i += 1
                if ~TRAINING_SKIP:
                    _, _, action = agent.actor.forward(state)
                else:
                    action = env.action_space.sample()
                state_obj, reward_obj, new_state_obj, done, info_obj = env.step(action)
                action_obj = sc.Action(action)
                if not evaluation_episode:
                    experience = sc.Experience(state_obj, action_obj, reward_obj, new_state_obj, done)
                    if ~TRAINING_SKIP:
                        agent.train_networks(experience)
                else:
                    episode_reward += reward_obj.reward
                state = new_state_obj.state
            if evaluation_episode:
                run_results.append(episode_reward)
        agent_results.append(run_results)

    env.close()

    n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
    results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
    mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

    x_vals = list(range(len(results_mean)))
    x_vals = [x_val * (TRAINING_EVALUATION_RATIO - 1) for x_val in x_vals]

    ax = plt.gca()
    ax.set_ylim([0, 200])
    ax.set_ylabel('Episode Score')
    ax.set_xlabel('Training Episode')
    ax.plot(x_vals, results_mean, label='Average Result', color='blue')
    ax.plot(x_vals, mean_plus_std, color='blue', alpha=0.1)
    ax.plot(x_vals, mean_minus_std, color='blue', alpha=0.1)
    ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='blue')
    plt.legend(loc='best')
    plt.show()