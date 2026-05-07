from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriverConfig:
    name: str = "best_lap"
    
    # Steering
    steer_lock: float = 0.785398
    centering_gain: float = 0.5
    
    # Speed Planner
    speed_distance_factor: float = 2.0
    steer_speed_penalty: float = 1.5
    angle_speed_penalty: float = 2.0
    cornering_speed: float = 75.0
    min_speed: float = 45.0
    max_speed: float = 280.0
    
    # Controller
    accel_smoothness: float = 20.0
    brake_smoothness: float = 30.0
    
    # Gears
    gear_speeds: tuple[float, float, float, float, float, float] = (0, 36, 72, 112, 158, 208)
    upshift_rpm: float = 7600.0
    downshift_rpm: float = 2900.0
    gear_hysteresis: float = 10.0
    
    # Recovery
    stuck_speed: float = 2.0
    stuck_time: int = 150
    reverse_time: int = 50
    
    # Opponents
    opponent_enabled: bool = False
    
    log_every: int = 1
