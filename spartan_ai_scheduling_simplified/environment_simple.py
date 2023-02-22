# -*- coding: utf-8 -*-
from collections import deque, namedtuple
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

    def __init__(self, wait_bay: psc.WaitingBay, furnaces: List[psc.Furnace]):
        self.waiting_bay = wait_bay
        self.furnaces = furnaces
        self.global_time_ctr = 0.0
        self.rolled_slab = psc.Slab(0)
        self.slabs_placed = 0
        self.slabs_removed = 0

        # Observation space: length of the slab observed at time t
        self.state_obj = sc.State()
        self.state = self.state_obj.state
        self.reward = 0

        # action space: {a_1, a_2, a_3}
        # a_1: Slab location from which agent picks the slab from the Waiting Bay -
        # 0 - Don't pick; 1 to wb_length - Pick from 1 to wb_length rpy; (n_locations+1) - Dummy.
        # a_2: Add Slab to furnace:
        # 0 - Don't place any slab; 1 to n_fur - Put in 1 to n_fur rpy; (n_fur+1) Dummy action.
        # a_3: Remove slab from furnace -
        # 0 - Don't remove any slab; 1 to n_fur - Remove from 1 to n_fur; (n_fur+1) Inconsequential action.
        self.action_space = MultiDiscrete([IPS.WAIT_BAY_LEN+1, IPS.n_furnaces+1, IPS.n_furnaces+1])

    def step(self, action: List[int]) -> Tuple[sc.State, sc.Reward, sc.State, bool, sc.Info]:
        """ Take the "action" and step to a new_state when a new slab arrives
        """

        reward = 0

        # For every step, generate state using wb, fur, rolled_slab
        self.state_obj.generate_state(self.waiting_bay, self.furnaces, self.rolled_slab)
        self.state = self.state_obj.state

        # Execute Action and Get Reward::
        # Action - a1: Pick the slab from Waiting Buffer
        picked_slab, action1_status = self.waiting_bay.remove_from_buffer(action[0])
        if action1_status == -1:
            # Tried to pick slab from an empty location. Stop the episode - hard constraint
            reward += IPS.reward_improper_pick_wb
            # Place the slab back in the waiting bay
            place_back_status = self.waiting_bay.add_to_buffer(location=action[0], slab=picked_slab)
            assert place_back_status == 1
        else:
            # Properly picked the slab
            reward += IPS.reward_proper_pick_wb

        # Action - a2: Remove a slab from furnace. Furnace num = action[1]
            removed_slab, action2_status = self.furnaces[action[1]].remove_slab()
            if action2_status == -1:
                # Tried to remove slab from an empty furnace - soft constraint
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

            # Action - a3: Place the picked slab in the furnace
            action3_status = self.furnaces[action[2]].add_slab(picked_slab)
            if action3_status == -1:
                # Cannot place the slab in furnace as it is full - soft constraint
                reward += IPS.reward_improper_add_fur
                # Place the slab back in the waiting bay
                place_back_status = self.waiting_bay.add_to_buffer(location=action[0], slab=picked_slab)
                assert place_back_status == 1
            else:
                # Properly added to the furnace
                reward += IPS.reward_proper_add_fur

            # Average accumulated negative reward for the overcooks in the furnaces:
            total_avg_sum = 0
            for furnace in self.furnaces:
                total_avg_sum += (np.where(np.array(furnace.balance_furnace_times) < 0.0,
                                  np.array(furnace.balance_furnace_times),
                                  0).sum())/((furnace.furnace_height+1) * (furnace.furnace_width+1))
            reward += total_avg_sum * IPS.reward_overcooks_in_furnace

            # Reward to penalise longer furnace on times and reduce energy
            total_avg_sum = 0
            for furnace in self.furnaces:
                total_avg_sum += (np.where(np.array(furnace.balance_furnace_times) > 0.0,
                                           np.array(furnace.balance_furnace_times),
                                           0).sum()) / ((furnace.furnace_height + 1) * (furnace.furnace_width + 1))
            reward += total_avg_sum * IPS.reward_longer_heating_in_furnace

            # Time-step: Reduce number of timesteps
            reward += IPS.reward_per_step

            if action2_status != -1:
                self.slabs_placed += 1
            if action3_status != -1:
                self.slabs_removed += 1

        done = False
        if (self.slabs_removed >= IPS.TOTAL_REMOVED) and (self.slabs_placed >= IPS.TOTAL_PLACED):
            done = True
        elif self.global_time_ctr >= IPS.MAX_TIME_STEPS * IPS.UNIT_TIME_STEP:
            done = True
        reward_obj = sc.Reward(reward)
        info = {}  # any high level info that needs to be passed
        info_obj = sc.Info(info)

        # Time-step dynamics:
        self.waiting_bay.time_step()
        for furnace in self.furnaces:
            furnace.time_step()
        self.rolled_slab.balance_rolling_time -= 1

        # New state:
        new_state_obj = sc.State()
        new_state_obj.generate_state(self.waiting_bay, self.furnaces, self.rolled_slab)
        new_state = new_state_obj.state

        return self.state_obj, reward_obj, new_state_obj, done, info_obj

    def render(self):
        pass

    def reset(self):
        """ Reset the slab thickness to 0.
        """
        self.waiting_bay = psc.WaitingBay()
        self.furnaces = []
        for i in range(1, IPS.n_furnaces + 1):
            self.furnaces.append(psc.Furnace(i))

        self.global_time_ctr = 0.0
        self.rolled_slab = psc.Slab(0)
        self.slabs_placed = 0
        self.slabs_removed = 0

        state_obj = sc.State()
        self.state = state_obj.state
        self.reward = 0
        return self.state

    def close(self):
        pass