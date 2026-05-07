from __future__ import annotations

import math

from .config import DriverConfig
from .math_utils import clamp, mean
from .sensors import sensor, track


class SteeringController:
    def __init__(self, config: DriverConfig) -> None:
        self.config = config
        self.last_info: dict[str, float | bool] = {}

    def update(self, sensors: dict[str, object]) -> float:
        values = track(sensors)
        angle = sensor(sensors, "angle")
        track_pos = sensor(sensors, "trackPos")

        min_side = min(min(values[:9]), min(values[10:]))
        in_corner = min_side < self.config.corner_distance or values[9] < sensor(sensors, "speedX") * 0.65
        steer = (angle * self.config.steer_gain / math.pi) - (track_pos * self.config.centering_gain)

        if in_corner:
            left_avg = mean(values[:9])
            right_avg = mean(values[10:])
            bias = right_avg - left_avg
            if bias < 0:
                steer += self.config.corner_bias
            elif bias > 0:
                steer -= self.config.corner_bias

        self.last_info = {
            "radar_front": round(values[9], 3),
            "radar_bias": round(mean(values[10:]) - mean(values[:9]), 3),
            "wall_bias": round(min_side, 3),
            "straight_enough": values[9] > self.config.straight_distance,
        }
        return clamp(steer, -1.0, 1.0)
