from __future__ import annotations

import math

from .config import DriverConfig
from .math_utils import clamp
from .sensors import sensor


class RecoveryController:
    def __init__(self, config: DriverConfig) -> None:
        self.config = config
        self.stuck_counter = 0
        self.reverse_counter = 0

    def apply(self, sensors: dict[str, object], action: dict[str, float | int]) -> tuple[dict[str, float | int], str]:
        speed = sensor(sensors, "speedX")
        angle = sensor(sensors, "angle")
        track_pos = sensor(sensors, "trackPos")
        
        offtrack = abs(track_pos) > 1.0
        wrong_way = abs(angle) > math.pi / 2.0
        
        # Increment stuck counter only if very slow and either off-track/wrong-way or trying to accelerate
        if speed < self.config.stuck_speed and (offtrack or wrong_way or float(action.get("accel", 0.0)) > 0.5):
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            
        if self.stuck_counter > self.config.stuck_time:
            self.reverse_counter = self.config.reverse_time
            self.stuck_counter = 0
            
        if self.reverse_counter > 0:
            self.reverse_counter -= 1
            steer = clamp(-angle - track_pos, -1.0, 1.0)
            return {
                "steer": steer,
                "accel": 0.5,
                "brake": 0.0,
                "gear": -1,
                "clutch": 0.0,
                "meta": 0,
            }, "reverse"
            
        if offtrack:
            action["accel"] = min(float(action.get("accel", 0.0)), 0.3)
            action["gear"] = 1 if speed < 40 else 2
            return action, "offtrack"
            
        return action, "normal"
