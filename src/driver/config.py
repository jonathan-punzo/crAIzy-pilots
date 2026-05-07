from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriverConfig:
    name: str = "best_lap"
    target_speed: float = 178.0
    max_speed: float = 255.0
    min_corner_speed: float = 82.0
    steer_gain: float = 10.8
    centering_gain: float = 0.24
    curve_gain: float = 0.08
    lateral_steer_damping: float = 0.018
    radar_steer_gain: float = 0.08
    wall_avoid_gain: float = 0.28
    wall_distance: float = 22.0
    straight_front_distance: float = 120.0
    straight_trackpos_deadband: float = 0.18
    steering_deadband: float = 0.045
    steer_smoothing: float = 0.50
    steer_rate_low_speed: float = 0.080
    steer_rate_high_speed: float = 0.022
    high_speed_steer_floor: float = 0.16
    pedal_smoothing: float = 0.58
    brake_threshold: float = 0.66
    brake_strength: float = 0.34
    lateral_speed_gain: float = 0.95
    traction_enabled: bool = True
    slip_threshold: float = 3.2
    traction_cut: float = 0.34
    gear_speeds: tuple[float, float, float, float, float, float] = (0, 36, 72, 112, 158, 208)
    upshift_rpm: float = 7600.0
    downshift_rpm: float = 2900.0
    gear_hysteresis: float = 20.0
    launch_speed: float = 44.0
    launch_accel: float = 0.52
    low_gear_accel: float = 0.70
    rpm_accel_cut: float = 0.22
    offtrack_speed: float = 52.0
    stuck_speed_threshold: float = 4.5
    stuck_angle_threshold: float = 0.84
    stuck_steps: int = 80
    reverse_steps: int = 38
    recovery_steer_gain: float = 0.75
    opponent_enabled: bool = False
    opponent_distance: float = 14.0
    opponent_brake: float = 0.12
    log_every: int = 1
