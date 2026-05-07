from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriverConfig:
    name: str = "best_lap"

    target_speed: float = 160.0
    straight_speed: float = 194.0
    gentle_corner_speed: float = 140.0
    sharp_corner_speed: float = 65.0
    corner_distance: float = 2.0
    slow_down_distance: float = 60.0
    straight_distance: float = 120.0

    steer_gain: float = 30.0
    centering_gain: float = 0.20
    corner_bias: float = 0.46
    steering_effect: float = 1.6

    brake_threshold: float = 0.40
    braking_intensity: float = 0.30
    extra_brake: float = 0.10

    throttle_step_up: float = 0.40
    throttle_step_down: float = 0.20
    launch_full_accel_speed: float = 10.0

    gear_speeds: tuple[float, float, float, float, float, float] = (0, 50, 80, 120, 150, 200)
    upshift_rpm: float = 7600.0
    downshift_rpm: float = 2900.0
    gear_hysteresis: float = 18.0

    traction_enabled: bool = True
    slip_threshold: float = 2.0
    traction_cut: float = 0.10

    stuck_speed_threshold: float = 3.0
    stuck_angle_threshold: float = 0.95
    stuck_steps: int = 120
    reverse_steps: int = 45
    recovery_steer_gain: float = 0.80
    offtrack_speed: float = 45.0

    opponent_enabled: bool = False
    opponent_distance: float = 14.0
    opponent_brake: float = 0.12

    log_every: int = 1
