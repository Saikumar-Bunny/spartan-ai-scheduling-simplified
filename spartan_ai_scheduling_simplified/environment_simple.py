# -*- coding: utf-8 -*-
from gym import Env
from gym.spaces import Box, Discrete
from spartan_ai_scheduling_simplified import INIT_PARAMS_SIMPLE as IPS
import time
from spartan_ai_scheduling_simplified.data_generation import Slab


class FurnaceEnvironment(Env):
    """
    State/Observation of the Furnace space availability."""

    def __init__(self):
        self._furnace1_eff = IPS.FUR1_EFF
        self._furnace2_eff = IPS.FUR2_EFF
        self.current_time = 0
        self.t_heating = 0
        self.t_heating_ideal_a1 = 0
        self.t_heating_ideal_a2 = 0
        self.th_slab = 0
        self.action = 0
        self.products_heated = 0
        self.data_counter = 0

        # Observation space: length of the slab observed at time t
        self.observation_space = Box(low=0.9 * IPS.th_slab_min,
                                     high=1.1 * IPS.th_slab_max,
                                     shape=(1,))
        # action space: 0 Implies furnace1, 1 Implies furnace2
        self.action_space = Discrete(2)

    def new_product(self):
        slab_sample = Slab(self.data_counter)
        self.t_heating_ideal_a1 = slab_sample.ideal_heating_time()
        self.t_heating_ideal_a2 = slab_sample.ideal_heating_time()
        self.data_counter += 1
        self.th_slab = slab_sample.th_slab
        return slab_sample.th_slab

    def step(self, action):
        """ Take the "action" and step to a new_state when a new slab arrives
        """

        # Generate 1 sample-slab
        self.action = action

        # Observes the slab property - only thickness here
        observation = (self.th_slab,)

        # Action - 0 Furnace1; 1 Furnace2
        if action == 0:
            # Find the heating time in furnace1 for slab with given properties
            self.t_heating = self.t_heating_ideal_a1 / IPS.FUR1_EFF
        elif action == 1:
            # Find the heating time in furnace2 for slab with given properties
            self.t_heating = self.t_heating_ideal_a2 / IPS.FUR2_EFF
        else:
            print("Cannot interpret \n")

        # Reward is inversely proportional to the total time at the end of TOTAL_TIME
        reward = self.t_heating
        self.current_time += self.t_heating
        self.products_heated += 1

        done = False
        if self.products_heated >= IPS.TOTAL_STEPS: # IPS.TOTAL_TIME:
            done = True

        return observation, reward, done, {}

    def render(self):
        print(f"Slab was {self.th_slab} thick and assigned to Furnace {self.action + 1}")

    def reset(self):
        """ Reset the slab thickness to 0.
        """
        self.current_time = 0
        self.t_heating = 0
        self.th_slab = 0
        self.products_heated = 0
        return self.th_slab,

    def close(self):
        pass
