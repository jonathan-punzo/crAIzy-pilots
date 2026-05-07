from __future__ import annotations

from .config import DriverConfig
from .gears import Gearbox
from .math_utils import clamp
from .opponents import OpponentGuard
from .recovery import RecoveryController
from .sensors import sensor
from .speed_planner import SpeedPlanner
from .steering import SteeringController


class TorcsAIDriver:
    def __init__(self, config: DriverConfig) -> None:
        self.config = config
        self.steering = SteeringController(config)
        self.speed_planner = SpeedPlanner(config)
        self.gearbox = Gearbox(config)
        self.recovery = RecoveryController(config)
        self.opponents = OpponentGuard(config)
        self.last_info: dict[str, object] = {}

    def update(self, sensors: dict[str, object]) -> dict[str, float | int]:
        steer = self.steering.update(sensors)
        target_speed, _ = self.speed_planner.target_speed(sensors, steer)
        speed = sensor(sensors, "speedX")
        
        accel = 0.0
        brake = 0.0
        
        speed_error = target_speed - speed
        
        if speed_error > 0:
            accel = clamp(speed_error / self.config.accel_smoothness, 0.1, 1.0)
        else:
            brake = clamp(-speed_error / self.config.brake_smoothness, 0.1, 1.0)
            
        if speed_error < -25.0:
            brake = 1.0
            
        if speed < 30 and accel > 0.5:
            accel = 0.5

        action: dict[str, float | int] = {
            "steer": steer,
            "accel": accel,
            "brake": brake,
            "gear": self.gearbox.update(sensors),
            "clutch": 0.0,
            "meta": 0,
        }

        if self.config.opponent_enabled:
            action, guarded = self.opponents.apply(action, sensors)
        else:
            guarded = False
            
        action, mode = self.recovery.apply(sensors, action)

        self.last_info = {
            "speed": round(speed, 1),
            "target": round(target_speed, 1),
            "steer": round(steer, 3),
            "accel": round(accel, 3),
            "brake": round(brake, 3),
            "mode": mode,
            "opp_guard": guarded
        }
        self.last_info.update(self.steering.last_info)
        
        return action
