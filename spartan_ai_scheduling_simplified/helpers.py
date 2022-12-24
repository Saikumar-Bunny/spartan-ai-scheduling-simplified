from spartan_ai_scheduling_simplified import INIT_PARAMS_SIMPLE as IPS
from spartan_ai_scheduling_simplified import helpers
import numpy as np
import pandas as pd
import datetime
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Device to run model on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_test_data(th_slab_min=IPS.th_slab_min,
                   th_slab_max=IPS.th_slab_max,
                   alpha=IPS.BETA_DIST_ALPHA,
                   beta=IPS.BETA_DIST_BETA,
                   total_samples=IPS.TEST_SAMPLES,
                   save_name=IPS.DATA_SAVE_NAME) -> pd.DataFrame:
    """
    Loads/samples slab thickness samples using a beta distribution
    :param th_slab_min: Minimum slab thickness value
    :param th_slab_max: Maximum slab thickness value
    :param alpha: Assumed Beta distribution parameter
    :param beta: Assumed Beta distribution parameter
    :param total_samples: Total slabs in the test data
    :param save_name: Test data file name
    :return: Dataframe with normalised 'Slab Thickness' column & values.
    """

    if os.path.exists(save_name):
        df = pd.read_csv(save_name)
    else:
        test_states = []
        for i in range(total_samples):
            test_states.append((th_slab_min / th_slab_max) + \
                               np.random.beta(alpha, beta, 1) * \
                               (1 - th_slab_min / th_slab_max))
        df = pd.DataFrame(test_states, columns=['Slab thickness'])
        df.to_csv(save_name)
    return df


def get_slab_thickness(data_counter) -> float:
    """
    Sample one value of slab property from the data set
    :param data_counter: Corresponding slab number in sequence
    :return: Normalised slab thickness value as float (0 to 1)
    """
    # Thickness distribution of the slabs - Normalised thickness
    df = helpers.load_test_data()
    th_slab_nrmlzd = df['Slab thickness'].iloc[data_counter]
    return th_slab_nrmlzd


def ideal_heating_time(th_slab_nrmlzd, upward=True, c_1=8 * 60) -> datetime.timedelta:
    # TODO: At the moment the ideal heating time is only a function of
    #   slab thickness. It can further include the impact of l_slab,
    #   vol_slab, mat_comp as separate parametric gaussian processes or
    #   a supervised learning model.

    # Sinusoidal function
    if th_slab_nrmlzd <= 1:
        if upward:
            t_heating_ideal = c_1 * np.sin((np.pi / 2) * th_slab_nrmlzd)
        else:
            t_heating_ideal = c_1 * (1 - np.sin((np.pi / 2) * th_slab_nrmlzd))
    else:
        raise ValueError(f"Length of the slab at exceeds the max limit: "
                         f"{th_slab_nrmlzd} >  1")
    return t_heating_ideal


def seed_torch(seed=IPS.SEED):
    """ Removes all the sources of randomness to ensure reproducibility of the
    results. Makes both CPU and GPU clock based random generators to take
    given seed value = 42.

    :param: - seed = seed value
    """

    # Set python based generators to be seeded from seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # For numpy - src:reproducibility docs

    # Set all CPU and GPU clocks based generators to be based on seed_val
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # Stops benchmarking across algorithms
    torch.backends.cudnn.deterministic = True  # Avoids non-deterministic algorithms


# Neural network for selecting appropriate action
def select_action(network, state, action=0):
    """  Selects an action given state
    :param: network (Pytorch Model): neural network used in forward pass
    :param: state (Array): environment state

    :return: action.item() (float): discrete action
    """

    # Create the state tensor
    state_action_new = [state, action]

    state_action_tensor = torch.tensor(state_action_new).float().unsqueeze(0).to(DEVICE)
    state_action_tensor.required_grad = True

    # Forward pass
    values = network(state_action_tensor)

    # Compute mean and standard-deviation to obtain the normal distribution
    # action = max(action_values)
    return values


# Using a neural network to learn our policy parameters for one continuous action
class PolicyNetwork(nn.Module):
    """ A neural network that takes in the observation, learns and outputs the
    appropriate parameters for the action-policy is designed here.
    """

    # Takes in observations and outputs actions mu and sigma
    def __init__(self):
        """ Read the observation_space, instantiate the layers in the network.
        Hidden-layer has 128 nodes while the output has 2 nodes.
        """

        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(2, 8)
        self.hidden_layer_1 = nn.Linear(8, 64)
        self.hidden_layer_2 = nn.Linear(64, 128)
        self.hidden_layer_3 = nn.Linear(128, 64)
        self.hidden_layer_4 = nn.Linear(64, 8)
        self.output_layer = nn.Linear(8, 1)

    # Forward pass
    def forward(self, x):
        """ Forward propagation through the network.
        Relu activation is applied.
        """

        # Input states
        x = self.input_layer(x)
        x = F.relu(x)  # Change on 20_07_22
        x = F.relu(self.hidden_layer_4(
            F.relu(self.hidden_layer_3(
                F.relu(self.hidden_layer_2(
                    F.relu(self.hidden_layer_1(x))))))))

        # Actions
        value = self.output_layer(x)
        return value


def process_rewards(rewards):
    """ Converts our rewards history into cumulative discounted rewards
    :param: rewards (Array): array of rewards
    :return: G (Array): array of cumulative discounted rewards
    """

    # Calculate G (cumulative discounted rewards)
    G = []

    # Track cumulative reward
    total_r = 0
    if len(rewards) == 1:
        G.append(rewards[0] / IPS.REWARD_NRMLZ_CONSTANT)
        # Whitening rewards
        G = torch.tensor(G, requires_grad=True).to(DEVICE)
    else:
        # Iterate rewards from Gt to G0
        for r in reversed(rewards):
            # Base case: G(T) = r(T)
            # Recursive: G(t) = r(t) + G(t+1)*DISCOUNT
            total_r = r + total_r * IPS.DISCOUNT_FACTOR
            G.insert(0, total_r)
            # Whitening rewards
            G = torch.tensor(G, requires_grad=True).to(DEVICE)
            G = (G - G.mean()) / G.std()
    return G