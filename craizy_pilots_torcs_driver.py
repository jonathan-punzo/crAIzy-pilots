"""
crAIzy pilots TORCS Hybrid Corkscrew Driver
==========================================
Drop-in autonomous driver for IBM AI Racing League / TORCS.

Recommended use:
    Put this file next to torcs_jm_par.py and in torcs_jm_par.py use:
        from craizy_pilots_torcs_driver import drive_modular

Then start TORCS, select scr_server 1, and run the official Python client.

Architecture:
- modular rule-based controller for stability;
- Corkscrew sector speed profile via distFromStart;
- vision-like curve anticipation from track[19];
- guardrails: off-track, large angle, lateral slip, wheelspin;
- traction control and automatic gear;
- CSV logging for first-lap tuning.

This script modifies only Python AI commands. It does not alter TORCS physics,
car files, liveries or SCR/UDP protocol.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_list(value: Any, n: int, default: float = 0.0) -> List[float]:
    if value is None:
        return [default] * n
    if isinstance(value, str):
        cleaned = value.replace("[", " ").replace("]", " ").replace(",", " ")
        out = [safe_float(v, default) for v in cleaned.split()]
    else:
        try:
            out = [safe_float(v, default) for v in list(value)]
        except Exception:
            out = [default] * n
    if len(out) < n:
        out += [default] * (n - len(out))
    return out[:n]


def avg(values: Sequence[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else 0.0


@dataclass
class DriverConfig:
    DRIVER_NAME: str = "crAIzy_pilots_hybrid_corkscrew"

    # Logging
    ENABLE_LOGGING: bool = True
    LOG_DIR: str = "logs"
    LOG_FILE: str = "craizy_driver_runs.csv"
    PROFILE_PATH: str = os.path.join("configs", "speed_profile_corkscrew.json")

    # Steering: angle + centering + free-space correction
    STEER_GAIN: float = 1.38
    CENTERING_GAIN: float = 0.62
    FREE_SPACE_GAIN: float = 0.36
    STEER_SPEED_REFERENCE: float = 165.0
    MAX_STEER_LOW_SPEED: float = 1.00
    MAX_STEER_HIGH_SPEED: float = 0.43

    # Target speed and braking. TORCS SCR speedX is usually treated as km/h.
    DEFAULT_TARGET_SPEED: float = 130.0
    START_TARGET_SPEED: float = 155.0
    MAX_TARGET_SPEED: float = 190.0
    MIN_TARGET_SPEED: float = 38.0
    FRONT_FAST: float = 145.0
    FRONT_MEDIUM: float = 92.0
    FRONT_SLOW: float = 55.0
    EMERGENCY_FRONT: float = 28.0
    BRAKE_GAIN: float = 75.0
    MAX_BRAKE: float = 0.92

    # Guardrails
    TRACKPOS_WARN: float = 0.72
    TRACKPOS_DANGER: float = 0.92
    ANGLE_WARN: float = 0.32
    ANGLE_DANGER: float = 0.68
    LATERAL_SPEED_WARN: float = 11.0
    LATERAL_SPEED_DANGER: float = 20.0

    # Traction
    ENABLE_TRACTION_CONTROL: bool = True
    WHEEL_RADIUS_ESTIMATE: float = 0.33
    SLIP_KMH_WARN: float = 22.0
    SLIP_KMH_DANGER: float = 42.0
    START_TRACTION_TIME: float = 3.0

    # Gears: thresholds 1->2, 2->3, 3->4, 4->5, 5->6
    GEAR_SPEEDS: Tuple[float, float, float, float, float] = (42.0, 78.0, 113.0, 150.0, 183.0)
    RPM_UPSHIFT: float = 8200.0
    RPM_DOWNSHIFT: float = 3300.0

    # Recovery
    ENABLE_RECOVERY: bool = True
    STUCK_SPEED: float = 4.0
    STUCK_TIME: float = 2.4
    OFFTRACK_RECOVERY_SPEED: float = 34.0

    # Starting Corkscrew profile. Improve these values with logs.
    DEFAULT_SPEED_PROFILE: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 145.0), (120.0, 168.0), (280.0, 155.0), (430.0, 118.0),
        (620.0, 138.0), (850.0, 162.0), (1080.0, 128.0), (1300.0, 108.0),
        (1500.0, 142.0), (1780.0, 170.0), (2050.0, 122.0), (2300.0, 92.0),
        (2520.0, 116.0), (2780.0, 145.0), (3050.0, 168.0), (3350.0, 132.0),
        (3600.0, 150.0), (3900.0, 172.0),
    ])


class CsvLogger:
    def __init__(self, cfg: DriverConfig):
        self.cfg = cfg
        self.tick = 0
        self.enabled = cfg.ENABLE_LOGGING
        self.path = os.path.join(cfg.LOG_DIR, cfg.LOG_FILE)
        self.header_written = False
        if self.enabled:
            os.makedirs(cfg.LOG_DIR, exist_ok=True)
            self.header_written = os.path.exists(self.path) and os.path.getsize(self.path) > 0

    def log(self, row: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.tick += 1
        row = {"tick": self.tick, "wall_time": time.time(), **row}
        fields = [
            "tick", "wall_time", "driver", "curLapTime", "lastLapTime", "distFromStart", "distRaced",
            "speedX", "speedY", "speedZ", "angle", "trackPos", "rpm", "gear_in", "gear_out",
            "target_speed", "profile_speed", "vision_speed", "front", "front_left", "front_right",
            "best_track_idx", "curve_score", "space_steer", "steer", "accel", "brake", "clutch",
            "damage", "offtrack", "stuck", "recovery", "slip_kmh",
        ] + [f"track_{i}" for i in range(19)]
        for f in fields:
            row.setdefault(f, "")
        with open(self.path, "a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow({f: row.get(f, "") for f in fields})


class HybridCorkscrewDriver:
    def __init__(self, cfg: Optional[DriverConfig] = None):
        self.cfg = cfg or DriverConfig()
        self.logger = CsvLogger(self.cfg)
        self.speed_profile = self._load_speed_profile()
        self.prev_damage = 0.0
        self.low_speed_since: Optional[float] = None
        self.last_action = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": 1}

    def drive(self, c: Any) -> None:
        S = c.S.d
        R = c.R.d
        s = self._read(S)
        if self.cfg.ENABLE_RECOVERY and self._needs_recovery(s):
            action, debug = self._recovery(s)
        else:
            action, debug = self._race(s)
        self._write(R, action)
        self._log(s, action, debug)
        self.prev_damage = s["damage"]
        self.last_action = action

    def _read(self, S: Dict[str, Any]) -> Dict[str, Any]:
        track = safe_list(S.get("track", []), 19, 0.0)
        track = [v if v > 0.0 else 0.0 for v in track]
        return {
            "angle": safe_float(S.get("angle", 0.0)),
            "curLapTime": safe_float(S.get("curLapTime", 0.0)),
            "lastLapTime": safe_float(S.get("lastLapTime", 0.0)),
            "damage": safe_float(S.get("damage", 0.0)),
            "distFromStart": safe_float(S.get("distFromStart", 0.0)),
            "distRaced": safe_float(S.get("distRaced", 0.0)),
            "gear": int(safe_float(S.get("gear", 1))),
            "rpm": safe_float(S.get("rpm", 0.0)),
            "speedX": safe_float(S.get("speedX", 0.0)),
            "speedY": safe_float(S.get("speedY", 0.0)),
            "speedZ": safe_float(S.get("speedZ", 0.0)),
            "track": track,
            "trackPos": safe_float(S.get("trackPos", S.get("tracPos", 0.0))),
            "wheelSpinVel": safe_list(S.get("wheelSpinVel", []), 4, 0.0),
        }

    def _race(self, s: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cfg = self.cfg
        track = s["track"]
        speed = s["speedX"]
        angle = s["angle"]
        track_pos = s["trackPos"]
        speed_y = s["speedY"]

        front = track[9]
        front_left = avg([track[6], track[7], track[8]])
        front_right = avg([track[10], track[11], track[12]])
        far_left = avg([track[0], track[1], track[2], track[3]])
        far_right = avg([track[15], track[16], track[17], track[18]])

        # Look for the widest forward opening.
        search_indices = list(range(4, 15))
        best_idx = max(search_indices, key=lambda i: track[i])
        space_steer = (best_idx - 9) / 9.0

        forward_clearance = (
            0.48 * front +
            0.22 * avg([track[8], track[10]]) +
            0.18 * avg([track[7], track[11]]) +
            0.12 * avg([track[6], track[12]])
        )
        curve_score = 1.0 - clip(forward_clearance / 175.0, 0.0, 1.0)

        profile_speed = self._profile_speed(s["distFromStart"])
        vision_speed = self._vision_speed(forward_clearance, front, curve_score)
        target_speed = min(profile_speed, vision_speed, cfg.MAX_TARGET_SPEED)

        # Standing start push: only if aligned and safe.
        if (s["curLapTime"] < cfg.START_TRACTION_TIME and s["distFromStart"] < 160.0 and
                abs(angle) < 0.22 and abs(track_pos) < 0.55 and front > 65.0):
            target_speed = max(target_speed, cfg.START_TARGET_SPEED)

        target_speed = self._guarded_speed(target_speed, s, front, curve_score)
        target_speed = clip(target_speed, cfg.MIN_TARGET_SPEED, cfg.MAX_TARGET_SPEED)

        side_imbalance = clip((front_left - front_right) / 120.0, -1.0, 1.0)
        far_imbalance = clip((far_left - far_right) / 160.0, -1.0, 1.0)

        steer = (
            (angle * cfg.STEER_GAIN / math.pi)
            - (track_pos * cfg.CENTERING_GAIN)
            + (space_steer * cfg.FREE_SPACE_GAIN)
            + (0.12 * side_imbalance)
            + (0.05 * far_imbalance)
        )

        # Limit steering at high speed.
        speed_ratio = clip(abs(speed) / cfg.STEER_SPEED_REFERENCE, 0.0, 1.0)
        max_steer = cfg.MAX_STEER_LOW_SPEED - 0.58 * speed_ratio
        max_steer = clip(max_steer, cfg.MAX_STEER_HIGH_SPEED, cfg.MAX_STEER_LOW_SPEED)
        steer = clip(steer, -max_steer, max_steer)

        # Smooth steering to avoid zig-zag.
        prev_steer = float(self.last_action.get("steer", 0.0))
        max_delta = 0.22 if speed < 80.0 else 0.12
        steer = clip(steer, prev_steer - max_delta, prev_steer + max_delta)

        accel, brake = self._speed_control(speed, target_speed, front, curve_score)

        # Stability reductions.
        if abs(steer) > 0.45 and speed > 105:
            accel *= 0.74
        if abs(steer) > 0.65 and speed > 80:
            accel *= 0.58
            brake = max(brake, 0.06)
        if abs(speed_y) > cfg.LATERAL_SPEED_WARN:
            factor = clip(1.0 - (abs(speed_y) - cfg.LATERAL_SPEED_WARN) / 28.0, 0.38, 1.0)
            accel *= factor
            if abs(speed_y) > cfg.LATERAL_SPEED_DANGER:
                brake = max(brake, 0.15)

        slip_kmh = self._slip_kmh(s)
        if cfg.ENABLE_TRACTION_CONTROL:
            if slip_kmh > cfg.SLIP_KMH_DANGER:
                accel *= 0.45
                brake = max(brake, 0.05)
            elif slip_kmh > cfg.SLIP_KMH_WARN:
                accel *= 0.72

        if 0 < front < cfg.EMERGENCY_FRONT and speed > 55:
            accel = 0.0
            brake = max(brake, 0.65)

        gear = self._gear(speed=speed, rpm=s["rpm"], current=s["gear"])

        action = {
            "steer": clip(steer, -1.0, 1.0),
            "accel": clip(accel, 0.0, 1.0),
            "brake": clip(brake, 0.0, cfg.MAX_BRAKE),
            "gear": int(gear),
            "clutch": 0.0,
            "meta": 0,
        }
        debug = {
            "front": front, "front_left": front_left, "front_right": front_right,
            "best_track_idx": best_idx, "curve_score": curve_score, "space_steer": space_steer,
            "profile_speed": profile_speed, "vision_speed": vision_speed, "target_speed": target_speed,
            "slip_kmh": slip_kmh, "offtrack": abs(track_pos) > 1.0, "stuck": False, "recovery": False,
        }
        return action, debug

    def _profile_speed(self, dist: float) -> float:
        chosen = self.cfg.DEFAULT_TARGET_SPEED
        for lower, speed in self.speed_profile:
            if dist >= lower:
                chosen = speed
            else:
                break
        return chosen

    def _vision_speed(self, clearance: float, front: float, curve_score: float) -> float:
        cfg = self.cfg
        if clearance >= cfg.FRONT_FAST:
            base = 182.0
        elif clearance >= cfg.FRONT_MEDIUM:
            base = 150.0
        elif clearance >= cfg.FRONT_SLOW:
            base = 112.0
        else:
            base = 72.0
        base *= (1.0 - 0.42 * curve_score)
        if front < cfg.FRONT_SLOW:
            base = min(base, 92.0)
        if front < cfg.EMERGENCY_FRONT:
            base = min(base, 58.0)
        return clip(base, cfg.MIN_TARGET_SPEED, cfg.MAX_TARGET_SPEED)

    def _guarded_speed(self, target: float, s: Dict[str, Any], front: float, curve_score: float) -> float:
        cfg = self.cfg
        tp = abs(s["trackPos"])
        an = abs(s["angle"])
        sy = abs(s["speedY"])
        if tp > cfg.TRACKPOS_DANGER:
            target *= 0.48
        elif tp > cfg.TRACKPOS_WARN:
            target *= 0.68
        if an > cfg.ANGLE_DANGER:
            target *= 0.48
        elif an > cfg.ANGLE_WARN:
            target *= 0.74
        if sy > cfg.LATERAL_SPEED_DANGER:
            target *= 0.55
        elif sy > cfg.LATERAL_SPEED_WARN:
            target *= 0.76
        if s["damage"] > self.prev_damage + 1.0:
            target *= 0.88
        if curve_score > 0.72 and front < 70:
            target *= 0.75
        return target

    def _speed_control(self, speed: float, target: float, front: float, curve_score: float) -> Tuple[float, float]:
        cfg = self.cfg
        err = target - speed
        if err >= 18.0:
            accel, brake = 1.0, 0.0
        elif err >= 4.0:
            accel, brake = clip(0.55 + err / 40.0, 0.45, 1.0), 0.0
        elif err >= -8.0:
            accel, brake = clip(0.18 + err / 35.0, 0.0, 0.38), 0.0
        else:
            accel, brake = 0.0, clip((-err) / cfg.BRAKE_GAIN, 0.08, cfg.MAX_BRAKE)
        if curve_score > 0.55 and speed > target:
            brake = max(brake, clip((speed - target) / 65.0, 0.10, 0.72))
            accel *= 0.55
        if front < cfg.FRONT_SLOW and speed > target - 3:
            brake = max(brake, 0.18)
            accel *= 0.5
        return clip(accel, 0.0, 1.0), clip(brake, 0.0, cfg.MAX_BRAKE)

    def _gear(self, speed: float, rpm: float, current: int) -> int:
        cfg = self.cfg
        if speed < -2:
            return -1
        gear = 1
        for threshold in cfg.GEAR_SPEEDS:
            if speed > threshold:
                gear += 1
        gear = int(clip(gear, 1, 6))
        if speed > 12:
            if rpm > cfg.RPM_UPSHIFT and gear < 6:
                gear += 1
            elif rpm < cfg.RPM_DOWNSHIFT and gear > 1:
                gear -= 1
        if current >= 1:
            gear = int(clip(gear, current - 1, current + 1))
        return int(clip(gear, 1, 6))

    def _slip_kmh(self, s: Dict[str, Any]) -> float:
        wheel_speed = avg(s["wheelSpinVel"]) * self.cfg.WHEEL_RADIUS_ESTIMATE * 3.6
        return max(0.0, wheel_speed - abs(s["speedX"]))

    def _needs_recovery(self, s: Dict[str, Any]) -> bool:
        offtrack = abs(s["trackPos"]) > 1.05
        invalid_track = max(s["track"]) <= 1.0
        bad_angle = abs(s["angle"]) > 1.35
        now = s["curLapTime"]
        if now > 3.0 and abs(s["speedX"]) < self.cfg.STUCK_SPEED:
            if self.low_speed_since is None:
                self.low_speed_since = now
        else:
            self.low_speed_since = None
        stuck = self.low_speed_since is not None and (now - self.low_speed_since) > self.cfg.STUCK_TIME
        return offtrack or invalid_track or bad_angle or stuck

    def _recovery(self, s: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cfg = self.cfg
        track_pos, angle, speed = s["trackPos"], s["angle"], s["speedX"]
        steer = clip((angle / math.pi) - 0.85 * track_pos, -1.0, 1.0)
        if abs(speed) < 3.0 and abs(angle) > 1.2:
            gear, accel, brake = -1, 0.45, 0.0
            steer = -clip(track_pos, -1.0, 1.0)
        else:
            gear = 1
            accel, brake = (0.45, 0.0) if speed < cfg.OFFTRACK_RECOVERY_SPEED else (0.0, 0.25)
        action = {"steer": steer, "accel": accel, "brake": brake, "gear": gear, "clutch": 0.0, "meta": 0}
        debug = {
            "front": s["track"][9], "front_left": avg(s["track"][6:9]), "front_right": avg(s["track"][10:13]),
            "best_track_idx": -1, "curve_score": 1.0, "space_steer": 0.0,
            "profile_speed": cfg.OFFTRACK_RECOVERY_SPEED, "vision_speed": cfg.OFFTRACK_RECOVERY_SPEED,
            "target_speed": cfg.OFFTRACK_RECOVERY_SPEED, "slip_kmh": self._slip_kmh(s),
            "offtrack": abs(track_pos) > 1.0, "stuck": True, "recovery": True,
        }
        return action, debug

    def _write(self, R: Dict[str, Any], a: Dict[str, Any]) -> None:
        # Common SnakeOil key is steer. Some variants may call it steering.
        if "steer" in R or "steering" not in R:
            R["steer"] = float(a["steer"])
        else:
            R["steering"] = float(a["steer"])
        R["accel"] = float(a["accel"])
        R["brake"] = float(a["brake"])
        R["gear"] = int(a["gear"])
        if "clutch" in R:
            R["clutch"] = float(a.get("clutch", 0.0))
        if "meta" in R:
            R["meta"] = int(a.get("meta", 0))

    def _load_speed_profile(self) -> List[Tuple[float, float]]:
        profile = list(self.cfg.DEFAULT_SPEED_PROFILE)
        path = self.cfg.PROFILE_PATH
        if not os.path.exists(path):
            return sorted(profile)
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            loaded = []
            for item in data:
                if isinstance(item, dict):
                    d = safe_float(item.get("dist", item.get("distance", 0.0)))
                    sp = safe_float(item.get("speed", item.get("target_speed", self.cfg.DEFAULT_TARGET_SPEED)))
                else:
                    d, sp = safe_float(item[0]), safe_float(item[1])
                loaded.append((d, clip(sp, self.cfg.MIN_TARGET_SPEED, self.cfg.MAX_TARGET_SPEED)))
            return sorted(loaded) if loaded else sorted(profile)
        except Exception:
            return sorted(profile)

    def _log(self, s: Dict[str, Any], a: Dict[str, Any], d: Dict[str, Any]) -> None:
        row = {
            "driver": self.cfg.DRIVER_NAME,
            "curLapTime": s["curLapTime"], "lastLapTime": s["lastLapTime"],
            "distFromStart": s["distFromStart"], "distRaced": s["distRaced"],
            "speedX": s["speedX"], "speedY": s["speedY"], "speedZ": s["speedZ"],
            "angle": s["angle"], "trackPos": s["trackPos"], "rpm": s["rpm"],
            "gear_in": s["gear"], "gear_out": a["gear"],
            "target_speed": d.get("target_speed"), "profile_speed": d.get("profile_speed"),
            "vision_speed": d.get("vision_speed"), "front": d.get("front"),
            "front_left": d.get("front_left"), "front_right": d.get("front_right"),
            "best_track_idx": d.get("best_track_idx"), "curve_score": d.get("curve_score"),
            "space_steer": d.get("space_steer"),
            "steer": a["steer"], "accel": a["accel"], "brake": a["brake"], "clutch": a.get("clutch", 0.0),
            "damage": s["damage"], "offtrack": int(bool(d.get("offtrack", False))),
            "stuck": int(bool(d.get("stuck", False))), "recovery": int(bool(d.get("recovery", False))),
            "slip_kmh": d.get("slip_kmh"),
        }
        row.update({f"track_{i}": s["track"][i] for i in range(19)})
        self.logger.log(row)


_GLOBAL_DRIVER = HybridCorkscrewDriver()


def drive_modular(c: Any) -> None:
    """Drop-in replacement for the original drive_modular(c)."""
    _GLOBAL_DRIVER.drive(c)


def drive(c: Any) -> None:
    """Alias for projects that call drive(c)."""
    drive_modular(c)


def drive_example(c: Any) -> None:
    """Alias for projects that call drive_example(c)."""
    drive_modular(c)
