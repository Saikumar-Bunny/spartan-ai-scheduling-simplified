from dataclasses import dataclass
import datetime
from spartan_ai_scheduling_simplified import helpers


@dataclass
class Slab:
    t_heating_ideal: datetime.timedelta

    def __init__(self, data_counter: int):
        self.th_slab = helpers.get_slab_thickness(data_counter)

    def ideal_heating_time(self, upward=True) -> datetime.timedelta:
        self.t_heating_ideal = helpers.ideal_heating_time(
            self.th_slab,
            upward)
        return self.t_heating_ideal
