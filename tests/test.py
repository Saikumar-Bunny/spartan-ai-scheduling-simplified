import numpy as np
import os
import pandas as pd
import torch
import time
from spartan_ai_scheduling_simplified import INIT_PARAMS_SIMPLE as IPS
from spartan_ai_scheduling_simplified import environment_simple
from spartan_ai_scheduling_simplified import helpers


# TESTING THE MODEL:
def testing_model(test_env, test_network):
    '''
    """We load the pre-trained model and run it for a different type of forcing.
    The final scores are printed to the console. """
    :param test_env: The test environment for evaluating model performance
    :param test_network: The network with the final trained parameters
    '''

    test_scores = []
    save_dict = {'thickness': [], 'action': [], 'reward': []}

    df_test_states = helpers.load_test_data()

    for i in range(len(df_test_states.index)):
        test_score = 0
        test_state = torch.tensor(df_test_states['Slab thickness'].iloc[i])
        print(test_state)
        test_action = helpers.select_action(test_network, test_state)
        print(test_action, type(test_action))
        test_new_state, test_reward, test_done, info = test_env.step(test_action)
        test_score += test_reward

        if test_done:
            break
        test_scores.append(test_score)
        save_dict['thickness'].append(df_test_states['Slab thickness'].iloc[i])
        save_dict['action'].append(test_action)
        save_dict['reward'].append(test_reward)

    df_test_records = pd.DataFrame(save_dict)
    df_test_records.to_csv(IPS.TEST_RECORDS)
    print(f'The cumulative average scores of the test: {np.array(test_scores).mean()}\n')


# TESTING
print('1. Testing the saved model: \n')
# try:
test_env = environment_simple.FurnaceEnvironment()
test_network = helpers.PolicyNetwork(test_env.observation_space.shape[0])
test_network.load_state_dict(torch.load('../spartan_ai_scheduling_simplified/' + IPS.MODEL))
    # test_network = torch.jit.load(IPS.MODEL)
# except:
#     print(f'Cannot find the saved model {IPS.MODEL}\n. \
#     Make sure the file is in the directory \n \
#     Or set SkipTraining = False in INIT_PARAMS file to retrain. \n.')

start_0 = time.time()
testing_model(test_env, test_network)
end_0 = time.time()
print(f'Completed testing in {round(end_0 - start_0, 2)} seconds \n')