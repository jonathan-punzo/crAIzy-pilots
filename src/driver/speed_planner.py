from __future__ import annotations

from .config import DriverConfig
from .math_utils import clamp
from .sensors import sensor, track


class SpeedPlanner:
    def __init__(self, config: DriverConfig) -> None:
        self.config = config

    def target_speed(self, sensors: dict[str, object], steer: float) -> tuple[float, float]:
        values = track(sensors)
        front = values[9]
        angle = sensor(sensors, "angle")
        
        max_dist = max(values)
        
        # Base target speed on the longest ray visible
        target = max_dist * self.config.speed_distance_factor
        
        # Penalize speed if we are steering heavily or not facing forward
        steer_penalty = 1.0 + abs(steer) * self.config.steer_speed_penalty
        angle_penalty = 1.0 + abs(angle) * self.config.angle_speed_penalty
        
        target = target / (steer_penalty * angle_penalty)
        
        # Anticipate sharp turns: if longest ray is far but front is short
        if front < 50 and max_dist > 80:
            target = min(target, self.config.cornering_speed)
            
        target = clamp(target, self.config.min_speed, self.config.max_speed)
        
        return target, 0.0
