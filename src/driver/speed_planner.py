from __future__ import annotations

from .config import DriverConfig
from .math_utils import clamp
from .sensors import sensor, track


class SpeedPlanner:
    def __init__(self, config: DriverConfig) -> None:
        self.config = config

    def target_speed(self, sensors: dict[str, object], steer: float) -> tuple[float, float]:
        values = track(sensors)
        speed = sensor(sensors, "speedX")
        min_side = min(min(values[:9]), min(values[10:]))
        front = values[9]
        max_front = max(values[7:12])

        in_corner = min_side < self.config.corner_distance or front < speed * 0.65
        safe_speed = self.config.gentle_corner_speed
        if max_front < self.config.slow_down_distance:
            safe_speed = self.config.sharp_corner_speed

        target = self.config.target_speed
        if speed >= self.config.target_speed - 5.0 and front > self.config.straight_distance:
            target = self.config.straight_speed
        if in_corner:
            target = min(target, safe_speed)

        if abs(sensor(sensors, "trackPos")) > 1.0:
            target = min(target, self.config.offtrack_speed)

        corner_pressure = 1.0 if in_corner else 0.0
        if max_front < speed * 0.60:
            corner_pressure = max(corner_pressure, clamp((speed * 0.60 - max_front) / 80.0, 0.0, 1.0))

        return target, corner_pressure
