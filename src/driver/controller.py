from __future__ import annotations

from .config import DriverConfig
from .gears import Gearbox
from .math_utils import clamp, lerp
from .opponents import OpponentGuard
from .recovery import RecoveryController
from .sensors import sensor
from .speed_planner import SpeedPlanner
from .steering import SteeringController
from .traction import TractionController


class TorcsAIDriver:
    def __init__(self, config: DriverConfig) -> None:
        self.config = config
        self.steering = SteeringController(config)
        self.speed_planner = SpeedPlanner(config)
        self.traction = TractionController(config)
        self.gearbox = Gearbox(config)
        self.recovery = RecoveryController(config)
        self.opponents = OpponentGuard(config)
        self.previous_accel = 0.2
        self.previous_brake = 0.0
        self.last_info: dict[str, object] = {}

    def update(self, sensors: dict[str, object]) -> dict[str, float | int]:
        steer = self.steering.update(sensors)
        target_speed, corner_pressure = self.speed_planner.target_speed(sensors, steer)
        speed = sensor(sensors, "speedX")
        track_pos = sensor(sensors, "trackPos")
        edge_pressure = clamp(
            (abs(track_pos) - self.config.edge_slowdown_start)
            / max(self.config.edge_slowdown_full - self.config.edge_slowdown_start, 0.05),
            0.0,
            1.0,
        )

        if edge_pressure > 0.0:
            target_speed = min(target_speed, 95.0 - edge_pressure * 36.0)

        if speed < target_speed:
            speed_gap = clamp((target_speed - speed) / max(target_speed, 1.0), 0.0, 1.0)
            accel_target = clamp(0.20 + speed_gap * 0.85, 0.0, 1.0)
            brake_target = 0.0
        else:
            over_speed = clamp((speed - target_speed) / 55.0, 0.0, 1.0)
            accel_target = 0.0
            brake_target = over_speed * self.config.brake_strength

        if corner_pressure > self.config.brake_threshold and speed > target_speed * 0.98:
            brake_target = max(
                brake_target,
                (corner_pressure - self.config.brake_threshold) * self.config.brake_strength,
            )
            accel_target = min(accel_target, 0.18)

        if edge_pressure > 0.0 and speed > target_speed * 0.92:
            brake_target = max(brake_target, 0.10 + edge_pressure * 0.30)
            accel_target = min(accel_target, 0.12)

        accel = lerp(self.previous_accel, accel_target, self.config.pedal_smoothing)
        brake = lerp(self.previous_brake, brake_target, self.config.pedal_smoothing)
        accel, slip = self.traction.apply(sensors, accel)

        if brake > 0.05:
            accel = min(accel, 0.10)

        action: dict[str, float | int] = {
            "steer": clamp(steer, -1.0, 1.0),
            "accel": clamp(accel, 0.0, 1.0),
            "brake": clamp(brake, 0.0, 1.0),
            "gear": self.gearbox.update(sensors),
            "clutch": 0.0,
            "meta": 0,
        }

        action, guarded = self.opponents.apply(action, sensors) if self.config.opponent_enabled else (action, False)
        action, mode = self.recovery.apply(sensors, action)

        self.previous_accel = float(action.get("accel", 0.0))
        self.previous_brake = float(action.get("brake", 0.0))
        self.last_info = {
            "target_speed": round(target_speed, 3),
            "corner_pressure": round(corner_pressure, 3),
            "edge_pressure": round(edge_pressure, 3),
            "radar_front": self.steering.last_info.get("radar_front", ""),
            "radar_bias": self.steering.last_info.get("radar_bias", ""),
            "wall_bias": self.steering.last_info.get("wall_bias", ""),
            "straight_enough": self.steering.last_info.get("straight_enough", ""),
            "slip": round(slip, 3),
            "mode": mode,
            "opponent_guard": guarded,
        }
        return action
