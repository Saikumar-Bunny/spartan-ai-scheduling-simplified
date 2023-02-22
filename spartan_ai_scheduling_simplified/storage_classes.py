import collections
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Dict

import INIT_PARAMS_SIMPLE as IPS
from PhysicalStorageClasses import Slab, Furnace, WaitingBay


# Common datatypes for RL:
@dataclass
class State:
    """
    State/Observation is a combination of:
    a. information maps of Waiting Bay - boolean of slabs, thickness of slabs,
    Ex: 4 slots in WB with 2 slabs in locations 2 and 3, with thickness 36.3 and 45.5 inches rpy
    [4 * 2] - 4 slabs and 2 property maps since we are using only 1 slab property, 1 boolean.
    --->  [[0, 0]
          [1, 36.3]
          [1, 45.5]
          [0, 0]]
    b. information maps of furnaces,
    Ex: We have say 2 furnaces with a 2x2 capacity. We need 2 furnace maps - Boolean, Remaining heat time
    --->  Fur 1: Boolean  [[0, 1]  t_remaining  [[0,  33]
                          [1, 1]]                [31, 35]]
          Fur 2: Boolean  [[0, 0]  t_remaining  [[0,  0]
                          [1, 1]]                [38, 43]]
    c. waiting time of current slab on the Rolling Mill.
    Ex: float: [33.01]
    """
    def __init__(self):
        self.state = []

    def generate_state(self, wait_bay: WaitingBay,
                        furnaces: List[Furnace],
                        rolled_slab: Slab):
        state_wb, state_furnace, state_roll_time = [], [], 0.0
        state_wb = np.transpose(np.array([wait_bay.locs_status, wait_bay.th_slabs_list]))
        for furnace in furnaces:
            state_furnace.append(furnace.furnace_slots_bool)
        for furnace in furnaces:
            state_furnace.append(furnace.balance_furnace_times)

        state_roll_time = rolled_slab.balance_rolling_time
        self.state = [state_wb, state_furnace, state_roll_time]


@dataclass
class Action:
    def __init__(self, action_values: List[int]):
        self.action_1 = action_values[0]
        self.action_2 = action_values[1]
        self.action_3 = action_values[2]
        self.action = action_values

    def get_val(self):
        return self.action


@dataclass
class Reward:
    def __init__(self, reward_val: float):
        self.reward = reward_val

    def get_val(self):
        return self.reward


@dataclass
class Info:
    def __init__(self, info: Dict):
        self.info = info

    def get_val(self):
        return self.info


@dataclass
class Experience:
    def __init__(
            self,
            state: State,
            action: Action,
            reward: Reward,
            new_state: State,
            done: bool
    ):
        self.state = state
        self.action = action
        self.new_state = new_state
        self.reward = reward
        self.done = done


class ExperienceReplay:

    def __init__(self):
        self.data = collections.deque([], maxlen=IPS.L_REPLAY_BUFFER)

    def __len__(self):
        return len(self.data)

    def add_elements(self, experience: Experience):
        self.data.appendleft(experience)
        if len(self.data) >= IPS.L_REPLAY_BUFFER:
            self.data.pop()

    def sample_elements_arrays(self, sample_size: int) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:

        states_obj, actions_obj, rewards_obj, new_states_obj, dones = self.sample_elements(sample_size)
        states_arr, actions_arr, rewards_arr, new_states_arr = [], [], [], []
        for i in range(len(states_obj)):
            states_arr.append(states_obj[i].state)
            actions_arr.append(actions_obj[i].action)
            rewards_arr.append(rewards_obj[i].reward)
            new_states_arr.append(new_states_obj[i].state)

        return np.array(states_arr), np.array(actions_arr), np.array(rewards_arr), np.array(new_states_arr), \
               np.array(dones, dtype=bool)

    def sample_elements(self, sample_size: int) -> Tuple[List[State], List[Action], List[Reward], List[State], List[bool]]:
        sample_list = [list(self.data)[i] for i in np.random.choice(IPS.L_REPLAY_BUFFER, sample_size, replace=False)]
        states_obj, actions_obj, rewards_obj, new_states_obj, dones = [], [], [], [], []
        i = 0
        for exp in sample_list:
            states_obj[i], actions_obj[i], rewards_obj[i], new_states_obj[i], dones[i] = \
                exp.state, exp.action, exp.reward, exp.new_state, exp.done
            i += 1
        return states_obj, actions_obj, rewards_obj, new_states_obj, dones