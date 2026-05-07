from __future__ import annotations

import math

from .config import DriverConfig
from .math_utils import clamp
from .sensors import sensor, track


class SteeringController:
    def __init__(self, config: DriverConfig) -> None:
        self.config = config
        self.last_info: dict[str, float] = {}

    def update(self, sensors: dict[str, object]) -> float:
        angle = sensor(sensors, "angle")
        track_pos = sensor(sensors, "trackPos")
        
        # The previous radar logic crashed because max() returns the first index (0 = rightmost) 
        # when all rays read 200.0m on a straight, causing a hard right turn.
        # The standard, highly stable TORCS heuristic is simply:
        target_angle = angle - track_pos * self.config.centering_gain
        steer = target_angle / self.config.steer_lock
        
        self.last_info = {
            "angle": round(angle, 3),
            "track_pos": round(track_pos, 3),
            "target_angle": round(target_angle, 3)
        }
        
        return clamp(steer, -1.0, 1.0)
