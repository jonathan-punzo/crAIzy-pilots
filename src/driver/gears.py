from __future__ import annotations

from .config import DriverConfig
from .sensors import sensor


class Gearbox:
    def __init__(self, config: DriverConfig) -> None:
        self.config = config
        self.gear = 1

    def update(self, sensors: dict[str, object]) -> int:
        speed = sensor(sensors, "speedX")
        rpm = sensor(sensors, "rpm")
        current = int(sensor(sensors, "gear", self.gear))

        if current in [1, 2, 3, 4, 5, 6]:
            self.gear = current

        if speed < 2.0:
            self.gear = 1
            return self.gear

        speed_based = 1
        for index, threshold in enumerate(self.config.gear_speeds, start=1):
            if speed >= threshold:
                speed_based = index
        speed_based = min(max(speed_based, 1), 6)

        if self.gear > speed_based + 1:
            self.gear = speed_based + 1
        elif self.gear < speed_based - 1:
            self.gear = speed_based - 1

        up_threshold = self.config.gear_speeds[min(self.gear, 5)]
        down_threshold = max(0.0, self.config.gear_speeds[max(self.gear - 1, 1)] - self.config.gear_hysteresis)

        if self.gear < 6 and (speed >= up_threshold or rpm >= self.config.upshift_rpm):
            self.gear += 1
        elif self.gear > 1 and (speed < down_threshold or rpm <= self.config.downshift_rpm):
            self.gear -= 1

        return min(max(self.gear, 1), 6)
