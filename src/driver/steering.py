from __future__ import annotations

import math

from .config import DriverConfig
from .math_utils import clamp, mean
from .sensors import sensor, track


RADAR_COMMANDS = [-1.0, -0.83, -0.67, -0.50, -0.33, -0.22, -0.17, -0.11, -0.06, 0.0, 0.06, 0.11, 0.17, 0.22, 0.33, 0.50, 0.67, 0.83, 1.0]


class SteeringController:
    def __init__(self, config: DriverConfig) -> None:
        self.config = config
        self.previous = 0.0
        self.last_info: dict[str, float | bool] = {}

    def update(self, sensors: dict[str, object]) -> float:
        values = track(sensors)
        angle = sensor(sensors, "angle")
        track_pos = sensor(sensors, "trackPos")
        speed_y = sensor(sensors, "speedY")

        negative_side = values[:8]
        positive_side = values[11:]
        front_clearance = mean(values[7:12])
        negative_clearance = mean(negative_side)
        positive_clearance = mean(positive_side)
        clearance_total = max(negative_clearance + positive_clearance, 1.0)
        curve_bias = (negative_clearance - positive_clearance) / clearance_total

        weights = [min(value, 200.0) ** 2 for value in values]
        radar_bias = sum(command * weight for command, weight in zip(RADAR_COMMANDS, weights)) / max(sum(weights), 1.0)

        min_negative = min(negative_side) if negative_side else 200.0
        min_positive = min(positive_side) if positive_side else 200.0
        wall_bias = 0.0
        if min_negative < self.config.wall_distance:
            wall_bias += (self.config.wall_distance - min_negative) / self.config.wall_distance
        if min_positive < self.config.wall_distance:
            wall_bias -= (self.config.wall_distance - min_positive) / self.config.wall_distance

        raw = angle * self.config.steer_gain / math.pi
        raw -= track_pos * self.config.centering_gain
        raw -= curve_bias * self.config.curve_gain
        raw -= speed_y * self.config.lateral_steer_damping
        raw += radar_bias * self.config.radar_steer_gain
        raw += wall_bias * self.config.wall_avoid_gain

        straight_enough = (
            front_clearance > self.config.straight_front_distance
            and abs(track_pos) < self.config.straight_trackpos_deadband
            and abs(angle) < 0.055
            and abs(speed_y) < 3.0
            and abs(radar_bias) < 0.16
            and min_negative > self.config.wall_distance
            and min_positive > self.config.wall_distance
        )
        self.last_info = {
            "radar_front": round(front_clearance, 3),
            "radar_bias": round(radar_bias, 3),
            "wall_bias": round(wall_bias, 3),
            "straight_enough": straight_enough,
        }
        if straight_enough:
            raw = 0.0
        elif abs(raw) < self.config.steering_deadband:
            raw = 0.0

        speed = max(sensor(sensors, "speedX"), 0.0)
        speed_limit = clamp(0.80 - speed / 265.0, self.config.high_speed_steer_floor, 1.0)
        # TORCS applies the steering sign opposite to our track-position correction.
        target = clamp(-raw, -speed_limit, speed_limit)

        speed_ratio = clamp(speed / 160.0, 0.0, 1.0)
        max_delta = (
            self.config.steer_rate_low_speed * (1.0 - speed_ratio)
            + self.config.steer_rate_high_speed * speed_ratio
        )
        delta = clamp(target - self.previous, -max_delta, max_delta)
        self.previous = clamp(self.previous + delta, -speed_limit, speed_limit)
        return self.previous
