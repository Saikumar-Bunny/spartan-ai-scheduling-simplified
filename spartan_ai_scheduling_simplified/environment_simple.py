# -*- coding: utf-8 -*-
from gym import Env
from gym.spaces import Box, Discrete, MultiDiscrete
from typing import List, Tuple, Dict

import numpy as np

import storage_classes as sc
import PhysicalStorageClasses as psc
import INIT_PARAMS_SIMPLE as IPS


class ReheatFurnaceEnvironment(Env):
    """
    State/Observation of the Furnace space availability.
    """
    waiting_bay: psc.WaitingBay
    furnaces: List[psc.Furnace]
    rolled_slab: psc.Slab
    state_obj: sc.State
    slabs_placed: int
    slabs_removed: int
    GLOBAL_TIME_CTR: float
    reward: sc.Reward = 0

    def __init__(self):
        # action space: {a_1, a_2, a_3}
        # a_1: Slab location from which agent picks the slab from the Waiting Bay -
        # 0 - Don't pick; 1 to wb_length - Pick from 1 to wb_length rpy; (n_locations+1) - Dummy.
        # a_2: Add Slab to furnace:
        # 0 - Don't place any slab; 1 to n_fur - Put in 1 to n_fur rpy; (n_fur+1) Dummy action.
        # a_3: Remove slab from furnace -
        # 0 - Don't remove any slab; 1 to n_fur - Remove from 1 to n_fur; (n_fur+1) Inconsequential action.
        # Make each action size the same - max(waiting bay length, num of furnaces)
        action_space_size = max(IPS.WAIT_BAY_LEN + 1, IPS.n_furnaces + 1)
        self.action_space = MultiDiscrete([action_space_size, action_space_size, action_space_size])

    def step(self, action: List) -> Tuple[sc.State, sc.Reward, sc.State, bool, sc.Info]:
        """ Take the "action" and step to a new_state when a new slab arrives
        """

        reward = 0

        # Convert array like actions to integer List:
        for i in range(len(action)):
            action[i] = action[i][0][0]

        skip_action_1, skip_action_2, skip_action_3 = False, False, False
        if action[0] > IPS.WAIT_BAY_LEN:
            skip_action_1 = True
            action[0] = 0
        if action[1] > IPS.n_furnaces:
            skip_action_2 = True
            action[1] = 0
        if action[2] > IPS.n_furnaces:
            skip_action_3 = True
            action[2] = 0

        # Execute Action and Get Reward::
        # Action - a1: Pick the slab from Waiting Buffer
        # Flags: 1 - properly picked a slab; -1 - picked from empty location; 0 - action to not pick a slab.
        if action[0] != 0:    # If action was 0 - Don't pick any slab - Else pick
            picked_slab, action1_status = self.waiting_bay.remove_from_buffer(action[0])
            if action1_status == -1:
                # Tried to pick slab from an empty location.
                reward += IPS.reward_improper_pick_wb
            else:
                # Properly picked the slab
                reward += IPS.reward_proper_pick_wb
        else:
            action1_status = 0    # Not a bad action but just that no slab is to be picked
            action3_status = 0    # Can't pick also implies can't add

        # Action - a2: Remove a slab from furnace. Furnace num = action[1]
        # Flag - 1 Removed the slab; -1 Tried to remove from empty location; 0 - Action to not remove any slab
        if action[1] != 0:          # Not empty action
            removed_slab, action2_status = self.furnaces[action[1]].remove_slab()
            if action2_status == -1:
                # Tried to remove slab from an empty furnace
                reward += IPS.reward_improper_remove_fur
            else:
                # Properly removed the slab but the rolling mill is blocked - soft constraint
                if self.rolled_slab.balance_rolling_time >= IPS.threshold_rolling_time:
                    reward += IPS.reward_rolling_block
                    # Place the slab back in the furnace
                    place_back_status = self.furnaces[action[1]].add_slab(removed_slab)
                    assert place_back_status == 1

                # Properly removed the slab and the rolling mill is free
                else:
                    self.rolled_slab = removed_slab
                    # Removed slab is properly cooked
                    if abs(removed_slab.overcook[1]) <= IPS.threshold_overcook:
                        reward += IPS.reward_propercook
                    # Removed slab is under or over cooked
                    else:
                        reward += removed_slab.overcook[1] * IPS.reward_overcook if removed_slab.overcook[0] \
                            else removed_slab.overcook[1] * IPS.reward_undercook
        else:
            action2_status = 0    # Assign don't remove any slab status

        # Action - a3: Place the picked slab in the furnace
        if action1_status == 1:  # Only if a slab was picked
            if action[2] == 0:  # Don't add the picked slab
                action3_status = 0
                # Place the slab back in the waiting bay
                place_back_status = self.waiting_bay.add_to_buffer(location=action[0], slab=picked_slab)
                assert place_back_status == 1
            else:     # action3 was to add a slab
                action3_status = self.furnaces[action[2]].add_slab(picked_slab)
                if action3_status == -1:
                    # Cannot place the slab in furnace as it is full
                    reward += IPS.reward_improper_add_fur
                    # Place the slab back in the waiting bay
                    place_back_status = self.waiting_bay.add_to_buffer(location=action[0], slab=picked_slab)
                    assert place_back_status == 1
                else:
                    # Properly added to the furnace
                    reward += IPS.reward_proper_add_fur
        elif action1_status == -1:
            action3_status = 0  # Dont pick implies dont add
        else:
            action3_status = 0  # Dont pick implies dont add

        # Average accumulated overcooks in the furnaces - negative reward before realised:
        avg_overcook_time = 0
        for i in range(1, IPS.n_furnaces + 1):
            avg_overcook_time += (np.where(np.array(self.furnaces[i].balance_furnace_times) < 0.0,
                              np.array(self.furnaces[i].balance_furnace_times),
                              0).sum())/((self.furnaces[i].furnace_height+1) * (self.furnaces[i].furnace_width+1))
        reward += avg_overcook_time * IPS.reward_overcooks_in_furnace

        # Reward to penalise longer furnace on times and reduce energy
        avg_furnace_time = 0
        for i in range(1, IPS.n_furnaces + 1):
            avg_furnace_time += (np.where(np.array(self.furnaces[i].balance_furnace_times) > 0.0,
                                       np.array(self.furnaces[i].balance_furnace_times),
                                       0).sum()) / ((self.furnaces[i].furnace_height + 1) * (self.furnaces[i].furnace_width + 1))
        reward += avg_furnace_time * IPS.reward_longer_heating_in_furnace

        # Time-step: Reduce number of time-steps
        reward += IPS.reward_per_step

        if action2_status == 1:
            self.slabs_removed += 1
        if action3_status == 1:
            self.slabs_placed += 1

        done = False
        if (self.slabs_removed >= IPS.TOTAL_REMOVED) and (self.slabs_placed >= IPS.TOTAL_PLACED):
            done = True
        elif self.GLOBAL_TIME_CTR >= IPS.MAX_TIME_STEPS * IPS.UNIT_TIME_STEP:
            done = True
        reward_obj = sc.Reward(reward)
        info = {}  # any high level info that needs to be passed
        info_obj = sc.Info(info)

        # Time-step dynamics:
        self.waiting_bay.time_step()
        for i in range(1, IPS.n_furnaces + 1):
            self.furnaces[i].time_step()
        self.rolled_slab.balance_rolling_time -= 1
        self.GLOBAL_TIME_CTR += IPS.UNIT_TIME_STEP

        # Inconsequential actions:
        if skip_action_1 or skip_action_2 or skip_action_3:
            reward += IPS.reward_inconsequential_action  # Penalise inconsequential actions

        # New state:
        new_state_obj = sc.State()
        new_state_obj.generate_state(self.waiting_bay, self.furnaces, self.rolled_slab)
        new_state = new_state_obj.state

        return self.state_obj, reward_obj, new_state_obj, done, info_obj

    def render(self):
        pass

    def reset(self) -> List[np.ndarray]:
        """ Reset the slab thickness to 0.
        """
        self.waiting_bay = psc.WaitingBay()
        self.waiting_bay.replenish_wb()
        self.furnaces = [None, ]
        for i in range(1, IPS.n_furnaces + 1):
            self.furnaces.append(psc.Furnace(i))

        self.GLOBAL_TIME_CTR = 0.0
        self.rolled_slab = psc.Slab(0)
        self.slabs_placed = 0
        self.slabs_removed = 0

        self.state_obj = sc.State()
        self.state_obj.generate_state(self.waiting_bay, self.furnaces, self.rolled_slab)
        self.state = self.state_obj.state
        self.reward = 0
        return self.state

    def close(self):
        pass