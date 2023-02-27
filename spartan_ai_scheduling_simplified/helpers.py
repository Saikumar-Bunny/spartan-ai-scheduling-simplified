import INIT_PARAMS_SIMPLE as IPS
import numpy as np
import pandas as pd
import os
from typing import Tuple
from string import ascii_lowercase
import itertools


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
        test_states.append((th_slab_min / th_slab_max) + np.random.beta(alpha, beta, total_samples) * (1 - th_slab_min / th_slab_max))
        test_states.append(np.array([str(i) for i in range(total_samples)]))
        df = pd.DataFrame(np.transpose(test_states), columns=['Slab thickness', 'Slab IDs'])
        df.to_csv(save_name)
    return df


def get_slab_samples(data_counter: int) -> Tuple[float, str]:
    """
    Sample one value of slab property from the data set
    :param data_counter: Corresponding slab number in sequence
    :return: Normalised slab thickness value as float (0 to 1)
    """
    # Thickness distribution of the slabs - Normalised thickness
    df = load_test_data()
    return df['Slab thickness'].iloc[data_counter], df['Slab IDs'].iloc[data_counter]


def heating_time(th_slab_nrmlzd: float, action: int, c_1=IPS.TIME_CONSTANT) -> float:
    # TODO: At the moment the ideal heating time is only a function of
    #   slab thickness. It can further include the impact of l_slab,
    #   vol_slab, mat_comp as separate parametric gaussian processes or
    #   a supervised learning model.

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
        return t_heating
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
    return 20*th_slab
