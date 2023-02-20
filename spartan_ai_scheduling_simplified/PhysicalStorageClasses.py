# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Dict, Tuple

from spartan_ai_scheduling_simplified import INIT_PARAMS_SIMPLE as IPS
from spartan_ai_scheduling_simplified import helpers

# Physical Storage Classes for Slab, Furnace, Waiting Bay:


@dataclass
class Slab:
    th_slab: float  # Slab thickness
    id: str  # Slab ID
    ideal_heat_time: float  # Assigned when placed in furnace
    actual_fur_time: float  # Assigned when removed out of furnace
    wait_bay_time: float  # Assigned when placed in Wait buffer
    overcook: List[bool, float]
    balance_rolling_time: float # Assigned when placed on the rolling mill

    def __init__(self, data_counter):
        self.th_slab, self.slab_id = helpers.get_slab_samples(data_counter)
        self.balance_rolling_time = helpers.get_rolling_time(self.th_slab)


class Furnace:
    furnace_slots_slabs: List[List[Slab]]

    def __init__(self, furnace_number: int):
        """
         Initialise a furnace with a furnace number, its dimensions, efficiency, empty slab slots.

        :param furnace_number: furnace we are referring.
        :param heat_curve_type: the relation b/n slab thickness and its heating time.
        """
        self.furnace_number: int = furnace_number
        self.heat_curve_type: int = IPS.HEAT_CURVE_TYPE[self.furnace_number]
        self.furnace_eff: float = IPS.FURNACE_EFFICIENCY[self.furnace_number]
        self.furnace_height: int = IPS.FURNACE_HEIGHT[self.furnace_number]
        self.furnace_width: int = IPS.FURNACE_WIDTH[self.furnace_number]
        self.furnace_slots_bool: List[List[int]] \
            = [[0 for _ in range(self.furnace_width)] for _ in range(self.furnace_height)]

        # Initiate all the furnace spaces with 'empty' type slabs:
        self.furnace_slots_slabs: List[List[Slab]] \
            = [[Slab(0) for _ in range(self.furnace_width)] for _ in range(self.furnace_height)]

        # Stores of location of highest non-empty furnace location:
        self.latest_slab_loc: List[int] = [self.furnace_height, self.furnace_width]

        # Map of remaining heat time for the slabs
        self.balance_furnace_times: List[List[float]] \
            = [[0.0 for _ in range(self.furnace_width)] for _ in range(self.furnace_height)]

    def is_full(self):
        """
        Checks if the furnace is full.
        Ex:
        [[1, 1]     [[Last slab placed here ----------> (1), 1]
        [1, 1]                                          [1, 1]
        [1, 1]                                          [1, 1]
        [1, 1]]                                         [1, 1]]
        :return: True if the furnace is full, False otherwise.
        """
        # Row of the latest slab is 0th row (Top) and Column of the latest slab is 0th col (Left col):
        if (self.latest_slab_loc[0] == 0) & (self.latest_slab_loc[1] == 0):
            return True
        return False

    def is_empty(self):
        """
        Checks if the furnace is empty.
        Ex:
        [[0, 0]                 [[0, 0]
        [0, 0]                  [0, 0]
        [0, 0]                  [0, 0]
        [0, 0]]                 [0, (0) <---------- First slab is always placed here. ]]
        :return: True if the furnace is empty, False otherwise.
        """
        # Row of the latest slab is last row (Bottom) and Column of the latest slab is last col (Left col):
        if (self.latest_slab_loc[0] == self.furnace_height - 1) & (self.latest_slab_loc[1] == self.furnace_width - 1):
            return True
        return False

    def add_slab(self, slab: Slab) -> int:
        """
        From the first physical row for each physical column, check if there is a slab. Place the current slab in the
        first empty slab as we move from bottom to top.

        Ex:
        [[0, 0]                 [[0, 0]
        [0, 0]                  [0, 0]
        [0, 1]   - - - - - - >  [1, 1]
        [1, 1]]                 [1, 1]]

        :param slab: the new slab to be placed
        :return: status of the action: 1 success, -1 failed as furnace is full.
        """
        if self.is_full():
            return -1
        if self.latest_slab_loc[1] > 0:  # Latest slab not to "Left end" of furnace
            self.latest_slab_loc[1] -= 1
        else:
            self.latest_slab_loc[0] -= 1
            self.latest_slab_loc[1] = (self.furnace_width - 1)

        self.furnace_slots_bool[self.latest_slab_loc[0]][self.latest_slab_loc[1]] = True
        self.furnace_slots_slabs[self.latest_slab_loc[0]][self.latest_slab_loc[1]] = slab
        slab.ideal_heat_time, _ = helpers.heating_time(slab, self.furnace_number)
        self.balance_furnace_times[self.latest_slab_loc[0]][self.latest_slab_loc[1]] = slab.ideal_heat_time
        return 1

    def remove_slab(self):
        """
        If the agent says to remove a slab at a particular, all the slabs above and upto are also removed INCLUDING
        the specified slab.

        Ex:

        [[0, 0]                 [[0, 0]
        [0, 0]                  [0, 0]
        [0, *1*]   - - - - - >  [0, 0]
        [1, 1]]                 [1, 1]]

        :param slab_loc: a tuple of row and column id for the slab starting with 0 from top to bottom and 0 from
        left to right.
        :return: 1 if successful, -1 if failed
        """
        if self.is_empty():
            return Slab(0), -1
        if self.latest_slab_loc[1] < (self.furnace_width - 1):  # Latest slab not to "Right end" of furnace
            self.latest_slab_loc[1] += 1
        else:
            self.latest_slab_loc[0] += 1
            self.latest_slab_loc[1] = 0

        slab_rmvd = self.furnace_slots_slabs[self.latest_slab_loc[0]][self.latest_slab_loc[1]]  # Of Dtype Slab
        overcook_time = slab_rmvd.actual_fur_time - slab_rmvd.ideal_heat_time
        if overcook_time > 0:
            slab_rmvd.overcook = [True, overcook_time]
        else:
            slab_rmvd.overcook = [False, overcook_time]
        return slab_rmvd, 1

    def time_step(self):
        """
        After every time step in real time, we have to decrease each slabs remaining heating time by 1 unit.
        This helps to keep note if the slab was overcooked/undercooked.
        :return:
        """
        for i in reversed(range(self.furnace_height)):
            for j in reversed(range(self.furnace_width)):
                if self.furnace_slots_bool[i][j]:  # Slab exists
                    slab = self.furnace_slots_slabs[i][j]
                    slab.actual_fur_time += IPS.UNIT_TIME_STEP
                    self.balance_furnace_times[i][j] -= IPS.UNIT_TIME_STEP
                else:
                    break  # Reached the latest slab loc


class WaitingBay:
    """
    size - size of the buffer.
    total_slabs - total number of slabs at this time step.
    container - array/list of slab datatypes.
    th_slabs_dict - dict of all slab thickness values with slab ids as keys.
    """
    def __init__(self, size: int = IPS.WAIT_BAY_LEN):
        """
        A Waiting Bay temporarily maintains the slabs before the agent assigns them to the furnace.
        Only when the Waiting Bay length is non-empty the agent is triggered and makes a decision.

        :param size: Maximum how many slabs can be in the waiting bay.
        """
        self.size: int = size
        self.slabs_in_bay: int = 0
        self.container: List[Slab] = [Slab(0) for _ in range(size)]     # Stores the actual slab
        self.locs_status: List[int] = [0 for _ in range(size)]   # Boolean: 0 for empty, 1 for full
        self.th_slabs_list: List[int] = [0 for _ in range(size)]
        self.th_slabs_dict: Dict = {}
        self.data_counter: int = 1

    def replenish_wb(self):
        """
        Add slab to all the empty locations in the Waiting Bay, when num of slabs < minimum capacity.
        """
        if self.slabs_in_bay > IPS.MIN_CAPACITY_WB:
            return -1

        for i, status in enumerate(self.locs_status):
            if status == 1:
                continue
            self.locs_status[i] = 1
            slab = Slab(self.data_counter)
            self.container[i] = slab
            self.data_counter += 1
            self.slabs_in_bay += 1
            self.th_slabs_list[i] = slab.th_slab
            self.th_slabs_dict[slab.slab_id] = slab.th_slab
        return 1

    def add_to_buffer(self, location: int, slab: Slab):
        """
        Add the slab (back) to the Waiting Bay. Mostly this action takes when furnace is full.
        :param location: Location where the slab is to be placed
        :param slab: The slab to be placed back.
        :return: status, -1 failed, 1 pass
        """
        self.locs_status[location] = 1
        self.container[location] = slab
        self.slabs_in_bay += 1
        self.th_slabs_list[location] = slab.th_slab
        self.th_slabs_dict[slab.slab_id] = slab.th_slab
        return 1

    def remove_from_buffer(self, location: int) -> Tuple[Slab, int]:
        """Remove a particular slab from the buffer and empty its spot.
        :param location: The location from where slab is to be picked. Specified by the agent.
        """
        if self.locs_status[location] == 0:
            return Slab(0), -1

        tmp_slab = self.container[location]
        self.container[location] = Slab(0)  # Reset the location.
        self.locs_status[location] = 0
        self.th_slabs_list[location] = 0
        self.th_slabs_dict[tmp_slab.slab_id] = 0
        self.locs_status[location] = 0

        # Check if the Waiting Bay needs to be replenished:
        self.replenish_wb()
        return tmp_slab, 1

    def time_step(self):
        """
        Step time for the slabs in the Waiting Bay.
        """
        for i, status in enumerate(self.locs_status):
            if status == 0:
                continue
            tmp_slab = self.container[i]
            tmp_slab.wait_bay_time += IPS.UNIT_TIME_STEP
