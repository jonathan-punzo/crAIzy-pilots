from __future__ import annotations

from .config import DriverConfig
from .math_utils import clamp
from .sensors import sensor, track


class RecoveryController:
    def __init__(self, config: DriverConfig) -> None:
        self.config = config
        self.stuck_counter = 0
        self.reverse_counter = 0

    def apply(self, sensors: dict[str, object], action: dict[str, float | int]) -> tuple[dict[str, float | int], str]:
        speed = sensor(sensors, "speedX")
        angle = sensor(sensors, "angle")
        track_pos = sensor(sensors, "trackPos")
        values = track(sensors)
        offtrack = abs(track_pos) > 1.0
        blind = min(values) <= 0.0
        wrong_way = abs(angle) > self.config.stuck_angle_threshold

        if speed < self.config.stuck_speed_threshold and (offtrack or blind or wrong_way):
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 2)

        if self.stuck_counter >= self.config.stuck_steps:
            self.reverse_counter = self.config.reverse_steps
            self.stuck_counter = 0

        if self.reverse_counter > 0:
            self.reverse_counter -= 1
            steer = clamp(angle * 0.45 + track_pos * 0.55, -1.0, 1.0)
            return {
                "steer": steer,
                "accel": 0.45,
                "brake": 0.0,
                "gear": -1,
                "clutch": 0.0,
                "meta": 0,
            }, "reverse_recovery"

        if offtrack or blind:
            steer = clamp(angle * 0.35 - track_pos * self.config.recovery_steer_gain, -1.0, 1.0)
            action["steer"] = steer
            action["accel"] = min(float(action.get("accel", 0.0)), 0.28)
            action["brake"] = max(float(action.get("brake", 0.0)), 0.18 if speed > self.config.offtrack_speed else 0.0)
            action["gear"] = 1 if speed < 45.0 else 2
            return action, "offtrack_recovery"

        return action, "normal"
