import INIT_PARAMS_SIMPLE as IPS
from PhysicalStorageClasses import Slab, Furnace
from PhysicalStorageClasses import WaitingBay
import helpers
import numpy as np
import pandas as pd
import datetime
import random
import os
from typing import List, Tuple

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


def get_slab_samples(data_counter: int) -> Tuple[float, str]:
    """
    Sample one value of slab property from the data set
    :param data_counter: Corresponding slab number in sequence
    :return: Normalised slab thickness value as float (0 to 1)
    """
    # Thickness distribution of the slabs - Normalised thickness
    df = helpers.load_test_data()
    return df['Slab thickness'].iloc[data_counter], df['Slab ID'].iloc[data_counter]


def heating_time(slab: Slab, action: int, c_1=IPS.TIME_CONSTANT) -> Tuple[float, float]:
    # TODO: At the moment the ideal heating time is only a function of
    #   slab thickness. It can further include the impact of l_slab,
    #   vol_slab, mat_comp as separate parametric gaussian processes or
    #   a supervised learning model.

    th_slab_nrmlzd = slab.th_slab
    if th_slab_nrmlzd <= 1:
        # Various types of heating functions can be defined here:
        # TODO: Add other types of heating curves or replace with supervised model
        # Sinusoidal
        t_heating_sin = c_1 * np.sin((np.pi / 2) * th_slab_nrmlzd)

        # Inverse Sinusoidal
        t_heating_inv_sin = c_1 * (1 - np.sin((np.pi / 2) * th_slab_nrmlzd))

        if IPS.HEAT_CURVE_TYPE[action] == 1:  # If we prescribed heat curve 1
            t_heating = t_heating_sin * IPS.FURNACE_EFFICIENCY[action]
            t1, t2 = t_heating_sin, t_heating_inv_sin
        else:
            t_heating = t_heating_inv_sin * IPS.FURNACE_EFFICIENCY[action]
            t2, t1 = t_heating_sin, t_heating_inv_sin

        t_heating_ideal = min(t1 * IPS.FURNACE_EFFICIENCY[0], t2 * IPS.FURNACE_EFFICIENCY[1])
        return t_heating, t_heating_ideal
    else:
        raise ValueError(f"Length of the slab at exceeds the max limit: "
                         f"{th_slab_nrmlzd} has slab >  1")


def get_rolling_time(th_slab: float):
    """
    Given the relevant slab properties, estimate the rolling time.
    Let's assume for now rolling time is constant and 2 units
    :param th_slab:
    :return:
    """
    return 2

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
    def __init__(self, n_batches, n_furnaces):
        """ Read the observation_space, instantiate the layers in the network.
        Hidden-layer has 128 nodes while the output has 2 nodes.
        """

        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=2)  # No padding; 1 pixel stride
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2)  # No padding; 1 pixel stride
        self.pool = nn.MaxPool2d(kernel_size=2)  # No padding; 2 pixel stride

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=30 * 30 * 32, out_features=128)
        self.drop = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.hidden_layer_2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, n_batches * n_furnaces)

    # Forward pass
    def forward(self, x):
        """ Forward propagation through the network.
        Relu activation is applied.
        """

        # Input states
        x = self.input_layer(x)
        x = F.relu(x)  # Change on 20_07_22
        x = F.relu(self.hidden_layer_2(
            F.relu(self.hidden_layer_1(x))))

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
        G.append(rewards[0])
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
            # G = (G - G.mean()) / G.std()
    return G


# Make state space:
def state_space_gen(slab_thickness_list: List[float], furnace_list: List[Furnace]) -> np.ndarray:
    fur_state_ht = 0
    for i in range(len(furnace_list)):
        fur_state_ht += furnace_list[i].furnace_height

    fur_state_wd = max([furnace_list[i].furnace_width for i in range(len(furnace_list))])

    assert len(slab_thickness_list) <= fur_state_ht

    fur_state_joined = np.zeros([fur_state_ht, fur_state_wd])

    count = 0
    for i in range(len(furnace_list)):
        for j in furnace_list[i].furnace_height:
            for k in furnace_list[i].furnace_width:
                fur_state_joined[count][k] = furnace_list[i].furnace_slots_slabs[j][k].ideal_heat_time \
                                             - furnace_list[i].furnace_slots_slabs[j][k].actual_fur_time
            count += 1

    for j in range(fur_state_ht - len(slab_thickness_list)):
        slab_thickness_list.append(0)

    assert len(slab_thickness_list) == fur_state_ht

    state_input = np.array(np.transpose(np.array(slab_thickness_list)), fur_state_joined)
    return state_input