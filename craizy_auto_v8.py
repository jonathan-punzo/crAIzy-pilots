"""V8: deterministic sensor driver with a bounded residual KNN advisor.

The base policy is always capable of driving without the model.  The KNN
only refines steering and target speed, while the safety governor and the
physical assists retain final authority.
"""

import argparse
import csv
import datetime  # Initializes datetime_CAPI before NumPy in the TORCS env.
import json
import math
import os
import shutil
import statistics
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

import snakeoil3_jm2 as snakeoil3


DRIVER_VERSION = "craizy_auto_v8_sensor_base_residual_knn_v9r5_s03_entry"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "torcs_ps4_dataset.csv")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRACE_PATH = os.path.join(LOG_DIR, "auto_v8_latest.csv")
TRACE_ARCHIVE_DIR = os.path.join(LOG_DIR, "auto_v8_runs")
RUNS_PATH = os.path.join(RESULTS_DIR, "auto_v8_runs.csv")
ANALYSIS_PATH = os.path.join(RESULTS_DIR, "auto_v8_analysis.csv")
VALIDATION_PATH = os.path.join(RESULTS_DIR, "auto_v8_validation.csv")

PORT = 3001
MAX_STEPS = 100000
TRACE_EVERY = 5
KNN_NEIGHBORS = 7
K_ANGLE = 3.0
K_POS = 0.15
K_CURVE = 0.10
K_AIM = 0.60
MAX_STEER_ADVICE = 0.12
MAX_SPEED_ADVICE = 25.0
MAX_OPERATIONAL_SPEED = 272.0
TARGET_SPEED_BIAS = 45.0
TIGHT_CURVE_SPEED_PENALTY = 20.0
TIGHT_CURVE_HOLD_TICKS = 90
FAST_CURVE_HOLD_TICKS = 45
TIGHT_CURVE_RELEASE_PER_TICK = 0.8
FAST_CURVE_RELEASE_PER_TICK = 2.0
PERFORMANCE_SPEED_BONUS = 15.0
PERFORMANCE_PROFILE_FACTOR = 0.96
CURVE_PROFILE_FACTOR = 0.90
SPEED_PROFILE_STEP = 5.0
FIRST_CORNER_START = 330.0
FIRST_CORNER_END = 550.0
CORKSCREW_START = 2330.0
CORKSCREW_END = 2530.0
LAST_CORNER_PREPARE_START = 3080.0
LAST_CORNER_BRAKE_START = 3150.0
LAST_CORNER_LINE_END = 3220.0
LAST_CORNER_END = 3310.0
SLOW_SPEED_CAP = 90.0
TRACK_SENSOR_ANGLES = (
    -90.0, -75.0, -60.0, -45.0, -30.0, -20.0, -15.0, -10.0, -5.0,
    0.0,
    5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0,
)


@dataclass(frozen=True)
class TrackBlock:
    name: str
    start: float
    end: float
    role: str
    protected: bool = False

    def contains(self, distance, include_end=False):
        if include_end:
            return self.start <= distance <= self.end
        return self.start <= distance < self.end


TRACK_BLOCKS = (
    TrackBlock("S01", 0.0, FIRST_CORNER_START, "start_straight"),
    TrackBlock(
        "S02_FIRST_CORNER",
        FIRST_CORNER_START,
        FIRST_CORNER_END,
        "first_corner",
        protected=True,
    ),
    TrackBlock("S03", FIRST_CORNER_END, 1000.0, "technical"),
    TrackBlock("S04", 1000.0, 1500.0, "fast"),
    TrackBlock("S05", 1500.0, 2000.0, "technical"),
    TrackBlock("S06", 2000.0, CORKSCREW_START, "corkscrew_approach"),
    TrackBlock(
        "S07_CORKSCREW",
        CORKSCREW_START,
        CORKSCREW_END,
        "corkscrew",
        protected=True,
    ),
    TrackBlock("S08", CORKSCREW_END, LAST_CORNER_PREPARE_START, "technical"),
    TrackBlock(
        "S09_LAST_CORNER",
        LAST_CORNER_PREPARE_START,
        LAST_CORNER_END,
        "last_corner",
        protected=True,
    ),
    TrackBlock("S10", LAST_CORNER_END, 3610.0, "finish_straight"),
)
TRACK_BLOCKS_BY_NAME = {block.name: block for block in TRACK_BLOCKS}
VALIDATION_SECTORS = tuple(
    (block.name, block.start, block.end) for block in TRACK_BLOCKS
)

OFFTRACK_CONFIRM_TICKS = 3
MIN_CLEAN_LAPS = 3
MIN_LAP_ROWS = 800
MIN_LAP_DISTANCE = 3500.0

WHEEL_RADII = (0.3306, 0.3306, 0.3276, 0.3276)
STEER_SMOOTHING = 0.34
STEER_TARGET_FILTER = 0.28
PEDAL_SMOOTHING = 0.20
MAX_STEER_LOW_SPEED = 0.92
MAX_STEER_HIGH_SPEED = 0.24
SPEED_FOR_MIN_STEER = 240.0
THROTTLE_STEER_START = 0.55
THROTTLE_STEER_FULL = 0.90
THROTTLE_STEER_MIN_ACCEL = 0.72

ABS_MIN_SPEED_KMH = 10.0
ABS_SLIP_START_MPS = 2.0
ABS_SLIP_FULL_MPS = 6.0
ABS_MAX_RELEASE = 0.75
TCS_SLIP_START_MPS = 3.0
TCS_SLIP_FULL_MPS = 10.0
TCS_MAX_CUT = 0.40

SHIFT_COOLDOWN = 0.60
UPSHIFT_RPM = 7600.0
PANIC_DOWNSHIFT_RPM = 2000.0
LOW_RPM_DOWNSHIFT = 3000.0
UPSHIFT_SPEED = {1: 45.0, 2: 78.0, 3: 112.0, 4: 148.0, 5: 184.0}
RPM_UPSHIFT_MIN_SPEED = {
    1: 38.0, 2: 66.0, 3: 98.0, 4: 130.0, 5: 163.0,
}
DOWNSHIFT_SPEED = {2: 30.0, 3: 58.0, 4: 90.0, 5: 122.0, 6: 155.0}

NORMAL = "NORMAL"
STABILIZE = "STABILIZE"
REVERSE = "REVERSE"
REJOIN = "REJOIN"

REQUIRED_COLUMNS = (
    "run_id", "step", "steer_action", "accel_action", "brake_action",
    "speedX", "speedY", "wheelSpinVel", "track", "trackPos", "angle",
    "rpm", "damage", "distFromStart",
)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def track_block_at(distance):
    for block in TRACK_BLOCKS:
        if block.contains(distance):
            return block
    if distance >= TRACK_BLOCKS[-1].end:
        return TRACK_BLOCKS[-1]
    return TRACK_BLOCKS[0]


def distance_in_block(distance, name, include_end=False):
    return TRACK_BLOCKS_BY_NAME[name].contains(
        distance, include_end=include_end
    )


def safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def safe_list(value, length, default=0.0):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            value = []
    if not isinstance(value, (list, tuple)):
        value = []
    result = [safe_float(item, default) for item in value[:length]]
    result.extend([default] * (length - len(result)))
    return result


def percentile(values, fraction):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), fraction * 100.0))


def linear_limit(value, start, full, minimum):
    magnitude = abs(value)
    if magnitude <= start:
        return 1.0
    if magnitude >= full:
        return minimum
    ratio = (magnitude - start) / max(full - start, 1e-6)
    return 1.0 + (minimum - 1.0) * ratio


def last_corner_speed_cap(distance):
    points = (
        (LAST_CORNER_BRAKE_START, 220.0),
        (3180.0, 210.0),
        (3200.0, 180.0),
        (3220.0, 140.0),
        (3240.0, 105.0),
        (3260.0, 78.0),
        (3290.0, 78.0),
        (LAST_CORNER_END, 105.0),
    )
    if distance < points[0][0] or distance > points[-1][0]:
        return MAX_OPERATIONAL_SPEED
    for (left_x, left_y), (right_x, right_y) in zip(
        points, points[1:]
    ):
        if left_x <= distance <= right_x:
            ratio = (distance - left_x) / max(right_x - left_x, 1e-6)
            return left_y + ratio * (right_y - left_y)
    return points[-1][1]


def first_corner_speed_cap(distance):
    points = (
        (300.0, 220.0),
        (330.0, 190.0),
        (360.0, 155.0),
        (390.0, 125.0),
        (420.0, 95.0),
        (450.0, 80.0),
        (500.0, 105.0),
        (FIRST_CORNER_END, 145.0),
    )
    if distance < points[0][0] or distance > points[-1][0]:
        return MAX_OPERATIONAL_SPEED
    for (left_x, left_y), (right_x, right_y) in zip(
        points, points[1:]
    ):
        if left_x <= distance <= right_x:
            ratio = (distance - left_x) / max(
                right_x - left_x, 1e-6
            )
            return left_y + ratio * (right_y - left_y)
    return points[-1][1]


def parse_row(raw):
    row = dict(raw)
    row["run_id"] = str(raw.get("run_id", "")).strip()
    row["track"] = safe_list(raw.get("track"), 19, -1.0)
    row["wheelSpinVel"] = safe_list(raw.get("wheelSpinVel"), 4, 0.0)
    for field in (
        "step", "curLapTime", "steer_action", "accel_action",
        "brake_action", "gear_action", "speedX", "speedY", "rpm",
        "damage", "distFromStart",
    ):
        row[field] = safe_float(raw.get(field))
    row["angle"] = safe_float(raw.get("angle"))
    row["trackPos"] = safe_float(raw.get("trackPos"))
    return row


def estimate_curve_from_track(track):
    """Return a signed curve hint using the two track-sensor fans."""
    left = statistics.mean(track[:9])
    right = statistics.mean(track[10:])
    return clamp((left - right) / 200.0, -1.0, 1.0)


def geometry(sensors):
    track = [max(0.0, value) for value in safe_list(
        sensors.get("track"), 19, 0.0
    )]
    front = statistics.median(track[8:11])
    near = min(track[7:12])
    wide = statistics.mean(track[6:13])
    curve_hint = estimate_curve_from_track(track)
    open_sensor = max(range(19), key=lambda index: track[index])
    aim_hint = -TRACK_SENSOR_ANGLES[open_sensor] / 90.0
    curve_urgency = clamp((45.0 - front) / 20.0, 0.0, 1.0)
    openness = clamp((0.45 * front + 0.35 * near + 0.20 * wide) / 200.0, 0.0, 1.0)
    return {
        "track": track,
        "front": front,
        "near": near,
        "wide": wide,
        "curve_hint": curve_hint,
        "aim_hint": aim_hint,
        "curve_urgency": curve_urgency,
        "openness": openness,
    }


class BaseSensorPolicy:
    def __init__(self, slow=False):
        self.slow = slow

    def steer(self, sensors, road=None):
        road = road or geometry(sensors)
        angle = safe_float(sensors.get("angle"))
        track_pos = safe_float(sensors.get("trackPos"))
        steer_center = K_ANGLE * angle / math.pi - K_POS * track_pos
        return clamp(
            steer_center
            + K_CURVE * road["curve_hint"]
            + K_AIM * road["aim_hint"] * road["curve_urgency"],
            -1.0,
            1.0,
        )

    def target_speed(self, sensors, steer, road=None):
        road = road or geometry(sensors)
        angle = abs(safe_float(sensors.get("angle")))
        track_pos = abs(safe_float(sensors.get("trackPos")))
        speed_y = abs(safe_float(sensors.get("speedY")))

        corner_speed = 72.0 + 120.0 * road["openness"]
        corner_speed -= 42.0 * abs(road["curve_hint"])
        corner_speed -= 35.0 * clamp(abs(steer), 0.0, 1.0)

        usable_distance = max(0.0, road["front"] - 12.0)
        braking_speed = 3.6 * math.sqrt(
            (max(55.0, corner_speed) / 3.6) ** 2
            + 2.0 * 9.0 * usable_distance
        )
        target = min(MAX_OPERATIONAL_SPEED, braking_speed + TARGET_SPEED_BIAS)
        target -= TIGHT_CURVE_SPEED_PENALTY * clamp(
            (70.0 - road["front"]) / 35.0,
            0.0,
            1.0,
        )
        target -= 70.0 * clamp((angle - 0.12) / 0.35, 0.0, 1.0)
        target -= 45.0 * clamp((track_pos - 0.65) / 0.35, 0.0, 1.0)
        target -= 55.0 * clamp((speed_y - 8.0) / 20.0, 0.0, 1.0)
        if self.slow:
            target = min(target, SLOW_SPEED_CAP)
        return clamp(target, 55.0, MAX_OPERATIONAL_SPEED)

    @staticmethod
    def pedal(speed_x, target_speed):
        error = target_speed - max(0.0, speed_x)
        if error > 30.0:
            return 1.0
        if error > 8.0:
            return clamp(0.36 + error / 62.0, 0.36, 0.84)
        if error > 1.5:
            return 0.28
        if error >= -3.0:
            return 0.0
        return -clamp(0.08 + (-error - 3.0) / 65.0, 0.08, 0.85)

    def action_intent(self, sensors):
        road = geometry(sensors)
        steer = self.steer(sensors, road)
        target_speed = self.target_speed(sensors, steer, road)
        pedal = self.pedal(safe_float(sensors.get("speedX")), target_speed)
        return {
            "steer": steer,
            "target_speed": target_speed,
            "pedal": pedal,
            **road,
        }


def sensor_features(sensors, base):
    track = safe_list(sensors.get("track"), 19, -1.0)
    wheels = safe_list(sensors.get("wheelSpinVel"), 4, 0.0)
    speed_x = safe_float(sensors.get("speedX"))
    wheel_speed = [
        abs(wheels[index]) * WHEEL_RADII[index] for index in range(4)
    ]
    slip = statistics.mean(wheel_speed[2:]) - abs(speed_x) / 3.6
    values = [clamp(value, 0.0, 200.0) / 200.0 for value in track]
    values.extend((
        clamp(safe_float(sensors.get("trackPos")), -2.0, 2.0),
        clamp(safe_float(sensors.get("angle")) / math.pi, -1.0, 1.0),
        clamp(speed_x / 300.0, -0.2, 1.2),
        clamp(safe_float(sensors.get("speedY")) / 100.0, -1.0, 1.0),
        clamp(safe_float(sensors.get("rpm")) / 10000.0, 0.0, 1.5),
        clamp(slip / 30.0, -1.0, 1.0),
        clamp(base["steer"], -1.0, 1.0),
        clamp(base["target_speed"] / 300.0, 0.0, 1.2),
    ))
    return np.asarray(values, dtype=np.float64)


class DemonstrationDataset:
    def __init__(self, path=DATASET_PATH):
        self.runs = []
        with open(path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            missing = [
                field for field in REQUIRED_COLUMNS
                if field not in (reader.fieldnames or [])
            ]
            if missing:
                raise RuntimeError("Colonne dataset mancanti: %s" % ", ".join(missing))
            groups = defaultdict(list)
            for raw in reader:
                row = parse_row(raw)
                if row["run_id"]:
                    groups[row["run_id"]].append(row)
        for run_id, rows in groups.items():
            rows.sort(key=lambda row: row["step"])
            if not self._invalid_lap(rows):
                self.runs.append({"run_id": run_id, "rows": rows})
        if len(self.runs) < MIN_CLEAN_LAPS:
            raise RuntimeError("Servono almeno tre giri puliti.")

    @staticmethod
    def _invalid_lap(rows):
        if len(rows) < MIN_LAP_ROWS:
            return True
        if max(row["distFromStart"] for row in rows) < MIN_LAP_DISTANCE:
            return True
        if max(row["damage"] for row in rows) > 0.0:
            return True
        bad_ticks = 0
        for row in rows:
            bad = abs(row["trackPos"]) >= 1.0 or min(row["track"]) < 0.0
            bad_ticks = bad_ticks + 1 if bad else 0
            if bad_ticks >= OFFTRACK_CONFIRM_TICKS:
                return True
        return False

    @staticmethod
    def arrays(runs, base_policy):
        features = []
        targets = []
        for run in runs:
            rows = run["rows"]
            for row in rows:
                base = base_policy.action_intent(row)
                features.append(sensor_features(row, base))
                expert_pedal = clamp(
                    row["accel_action"] - row["brake_action"],
                    -1.0,
                    1.0,
                )
                targets.append((
                    clamp(
                        row["steer_action"] - base["steer"],
                        -MAX_STEER_ADVICE,
                        MAX_STEER_ADVICE,
                    ),
                    clamp(
                        30.0 * (expert_pedal - base["pedal"]),
                        -MAX_SPEED_ADVICE,
                        MAX_SPEED_ADVICE,
                    ),
                ))
        return np.vstack(features), np.asarray(targets, dtype=np.float64)


class ExpertSpeedProfile:
    def __init__(self, runs):
        lengths = [
            max(row["distFromStart"] for row in run["rows"])
            for run in runs
        ]
        self.track_length = statistics.median(lengths)
        self.grid = np.arange(
            0.0,
            self.track_length + SPEED_PROFILE_STEP,
            SPEED_PROFILE_STEP,
        )
        run_profiles = []
        for run in runs:
            ordered = sorted(
                run["rows"], key=lambda row: row["distFromStart"]
            )
            distances = np.asarray(
                [row["distFromStart"] for row in ordered],
                dtype=np.float64,
            )
            speeds = np.asarray(
                [row["speedX"] for row in ordered],
                dtype=np.float64,
            )
            unique_distances, unique_indices = np.unique(
                distances, return_index=True
            )
            run_profiles.append(np.interp(
                self.grid,
                unique_distances,
                speeds[unique_indices],
            ))
        self.speeds = np.median(np.vstack(run_profiles), axis=0)

    def speed_at(self, distance):
        if distance < 0.0:
            return 0.0
        wrapped = distance % self.track_length
        return float(np.interp(wrapped, self.grid, self.speeds))


class ResidualAdvisor:
    def __init__(self, features, targets):
        self.targets = np.asarray(targets, dtype=np.float64)
        self.model = KNeighborsRegressor(
            n_neighbors=KNN_NEIGHBORS,
            weights="distance",
            metric="euclidean",
            n_jobs=1,
        )
        self.model.fit(features, targets)
        self.model.kneighbors(features[:1], return_distance=True)

    def predict(self, features):
        vector = np.asarray(features).reshape(1, -1)
        started = time.perf_counter()
        distances, indices = self.model.kneighbors(
            vector, return_distance=True
        )
        neighbor_distances = distances[0]
        neighbor_targets = self.targets[indices[0]]
        exact = neighbor_distances <= 1e-12
        if exact.any():
            prediction = neighbor_targets[exact].mean(axis=0)
        else:
            weights = 1.0 / neighbor_distances
            prediction = np.average(
                neighbor_targets, axis=0, weights=weights
            )
        latency = (time.perf_counter() - started) * 1000.0
        distance = float(np.mean(neighbor_distances))
        confidence = clamp((0.75 - distance) / 0.50, 0.0, 1.0)
        return {
            "raw_delta_steer": float(prediction[0]),
            "raw_delta_speed": float(prediction[1]),
            "delta_steer": clamp(
                float(prediction[0]) * confidence,
                -MAX_STEER_ADVICE,
                MAX_STEER_ADVICE,
            ),
            "delta_speed": clamp(
                float(prediction[1]) * confidence,
                -MAX_SPEED_ADVICE,
                MAX_SPEED_ADVICE,
            ),
            "neighbor_distance": distance,
            "confidence": confidence,
            "inference_ms": latency,
        }


class AutomaticGearbox:
    def __init__(self):
        self.reset()

    def reset(self):
        self.gear = 1
        self.pending_gear = None
        self.last_shift_time = float("-inf")

    @staticmethod
    def _sensor_gear(sensors):
        gear = int(safe_float(sensors.get("gear"), 0.0))
        return gear if 1 <= gear <= 6 else None

    def update(self, sensors, accel=0.0, brake=0.0, now=None):
        now = time.monotonic() if now is None else float(now)
        speed = abs(safe_float(sensors.get("speedX")))
        rpm = safe_float(sensors.get("rpm"))
        accel = clamp(safe_float(accel), 0.0, 1.0)
        brake = clamp(safe_float(brake), 0.0, 1.0)
        sensor_gear = self._sensor_gear(sensors)

        if self.pending_gear is not None:
            if sensor_gear == self.pending_gear:
                self.gear = sensor_gear
                self.pending_gear = None
            elif now - self.last_shift_time < SHIFT_COOLDOWN:
                return self.pending_gear
            else:
                self.pending_gear = None

        if sensor_gear is not None:
            self.gear = sensor_gear

        current = int(clamp(self.gear, 1, 6))
        if now - self.last_shift_time < SHIFT_COOLDOWN:
            return current

        new_gear = current
        if current > 1:
            downshift_speed = DOWNSHIFT_SPEED[current]
            braking_or_coasting = brake > 0.05 or accel < 0.35
            ordinary_downshift = (
                speed < downshift_speed and braking_or_coasting
            )
            panic_downshift = 0.0 < rpm < PANIC_DOWNSHIFT_RPM
            low_rpm_downshift = (
                0.0 < rpm < LOW_RPM_DOWNSHIFT
                and speed < downshift_speed - 5.0
                and (brake > 0.05 or accel < 0.60)
            )
            if ordinary_downshift or panic_downshift or low_rpm_downshift:
                new_gear = current - 1

        if new_gear == current and current < 6:
            rpm_upshift = (
                rpm >= UPSHIFT_RPM
                and speed >= RPM_UPSHIFT_MIN_SPEED[current]
            )
            forced_speed_upshift = speed >= UPSHIFT_SPEED[current]
            if rpm_upshift or forced_speed_upshift:
                new_gear = current + 1

        if new_gear != current:
            self.gear = new_gear
            self.pending_gear = new_gear
            self.last_shift_time = now

        return int(self.gear)


class SharedADAS:
    def __init__(self):
        self.gearbox = AutomaticGearbox()
        self.steer = 0.0
        self.steer_target = 0.0
        self.accel = 0.0
        self.brake = 0.0
        self.last_diagnostics = {}

    def apply(self, sensors, intent, forced_gear=None):
        speed = abs(safe_float(sensors.get("speedX")))
        target_steer = clamp(intent["steer"], -1.0, 1.0)
        target_accel = clamp(intent["accel"], 0.0, 1.0)
        target_brake = clamp(intent["brake"], 0.0, 1.0)
        steer_limit = MAX_STEER_LOW_SPEED + (
            MAX_STEER_HIGH_SPEED - MAX_STEER_LOW_SPEED
        ) * clamp(speed / SPEED_FOR_MIN_STEER, 0.0, 1.0)
        target_steer = clamp(target_steer, -steer_limit, steer_limit)
        self.steer_target += STEER_TARGET_FILTER * (
            target_steer - self.steer_target
        )
        target_accel *= linear_limit(
            self.steer_target,
            THROTTLE_STEER_START,
            THROTTLE_STEER_FULL,
            THROTTLE_STEER_MIN_ACCEL,
        )
        steer_rate = 0.018 if speed > 170.0 else (0.026 if speed > 100.0 else 0.04)
        self.steer += clamp(
            (self.steer_target - self.steer) * STEER_SMOOTHING,
            -steer_rate,
            steer_rate,
        )
        self.accel += PEDAL_SMOOTHING * (target_accel - self.accel)
        brake_alpha = 0.38 if target_brake < self.brake else PEDAL_SMOOTHING
        self.brake += brake_alpha * (target_brake - self.brake)

        wheels = safe_list(sensors.get("wheelSpinVel"), 4, 0.0)
        wheel_speed = [
            abs(wheels[index]) * WHEEL_RADII[index] for index in range(4)
        ]
        vehicle_speed = speed / 3.6
        abs_slip = max(0.0, vehicle_speed - statistics.mean(wheel_speed))
        traction_slip = max(
            0.0, statistics.mean(wheel_speed[2:]) - vehicle_speed
        )
        self.last_diagnostics = {
            "governed_steer": target_steer,
            "filtered_steer_target": self.steer_target,
            "actuator_steer": self.steer,
            "front_wheel_speed": statistics.mean(wheel_speed[:2]),
            "rear_wheel_speed": statistics.mean(wheel_speed[2:]),
            "abs_slip": abs_slip,
            "traction_slip": traction_slip,
        }
        output_brake = self.brake
        output_accel = self.accel
        if speed >= ABS_MIN_SPEED_KMH and abs_slip > ABS_SLIP_START_MPS:
            release = clamp(
                (abs_slip - ABS_SLIP_START_MPS)
                / (ABS_SLIP_FULL_MPS - ABS_SLIP_START_MPS),
                0.0,
                1.0,
            ) * ABS_MAX_RELEASE
            output_brake *= 1.0 - release
        if traction_slip > TCS_SLIP_START_MPS:
            cut = clamp(
                (traction_slip - TCS_SLIP_START_MPS)
                / (TCS_SLIP_FULL_MPS - TCS_SLIP_START_MPS),
                0.0,
                1.0,
            ) * TCS_MAX_CUT
            output_accel *= 1.0 - cut
        if target_brake > 0.05 or output_brake > 0.05:
            output_accel = 0.0
        return {
            "steer": clamp(self.steer, -1.0, 1.0),
            "accel": clamp(output_accel, 0.0, 1.0),
            "brake": clamp(output_brake, 0.0, 1.0),
            "gear": (
                forced_gear
                if forced_gear is not None
                else self.gearbox.update(
                    sensors,
                    accel=output_accel,
                    brake=output_brake,
                )
            ),
            "clutch": 0.0,
            "meta": 0,
        }


class SafetyGovernor:
    def __init__(self, speed_profile=None):
        self.speed_profile = speed_profile
        self.state = NORMAL
        self.state_ticks = 0
        self.bad_ticks = 0
        self.stuck_ticks = 0
        self.previous_track_pos = None
        self.track_pos_rate = 0.0
        self.tight_curve_ticks = 0
        self.tight_curve_cap = MAX_OPERATIONAL_SPEED
        self.corkscrew_rescue = False

    def recovery(self, sensors):
        track_pos = safe_float(sensors.get("trackPos"))
        angle = safe_float(sensors.get("angle"))
        speed = safe_float(sensors.get("speedX"))
        steer = clamp(K_ANGLE * angle / math.pi - 0.55 * track_pos, -0.65, 0.65)
        self.state_ticks += 1
        if self.state == STABILIZE:
            if abs(track_pos) < 0.88 and abs(angle) < 0.30 and speed > 8.0:
                self.state = REJOIN
                self.state_ticks = 0
            elif self.state_ticks > 100 and abs(speed) < 6.0:
                self.state = REVERSE
                self.state_ticks = 0
            return steer, 0.18 if speed < 25.0 else -0.12, None
        if self.state == REVERSE:
            if self.state_ticks > 55:
                self.state = STABILIZE
                self.state_ticks = 0
            return -steer, 0.28, -1
        if self.state == REJOIN:
            if self.state_ticks > 40:
                self.state = NORMAL
                self.state_ticks = 0
                self.bad_ticks = 0
                self.stuck_ticks = 0
            return steer, 0.25, None
        raise RuntimeError("Stato recovery non valido.")

    def apply(self, sensors, base, advisor):
        speed = max(0.0, safe_float(sensors.get("speedX")))
        speed_y = abs(safe_float(sensors.get("speedY")))
        angle = abs(safe_float(sensors.get("angle")))
        signed_track_pos = safe_float(sensors.get("trackPos"))
        track_pos = abs(signed_track_pos)
        track = safe_list(sensors.get("track"), 19, -1.0)
        raw_track_rate = (
            0.0
            if self.previous_track_pos is None
            else signed_track_pos - self.previous_track_pos
        )
        self.previous_track_pos = signed_track_pos
        self.track_pos_rate = (
            0.75 * self.track_pos_rate + 0.25 * raw_track_rate
        )
        projected_track_pos = (
            signed_track_pos + 18.0 * self.track_pos_rate
        )
        outside = track_pos >= 1.0 or min(track) < 0.0
        self.bad_ticks = self.bad_ticks + 1 if outside else 0
        self.stuck_ticks = self.stuck_ticks + 1 if speed < 2.0 else 0
        if self.state == NORMAL and (
            self.bad_ticks >= 3 or self.stuck_ticks >= 150
        ):
            self.state = STABILIZE
            self.state_ticks = 0

        interventions = []
        distance = safe_float(sensors.get("distFromStart"), -1.0)
        steer = base["steer"] + advisor["delta_steer"]
        target_speed = base["target_speed"] + advisor["delta_speed"]
        target_speed = clamp(target_speed, 55.0, MAX_OPERATIONAL_SPEED)

        track_block = track_block_at(distance)
        corkscrew_sector = distance_in_block(
            distance, "S07_CORKSCREW", include_end=True
        )
        last_corner_sector = distance_in_block(
            distance, "S09_LAST_CORNER", include_end=True
        )
        first_corner_sector = distance_in_block(
            distance, "S02_FIRST_CORNER", include_end=True
        )
        protected_sector = (
            first_corner_sector
            or corkscrew_sector
            or last_corner_sector
        )
        if not 2240.0 <= distance <= 2370.0:
            self.corkscrew_rescue = False
        elif (
            not self.corkscrew_rescue
            and distance <= 2305.0
            and projected_track_pos < -0.25
        ):
            self.corkscrew_rescue = True
        if not protected_sector:
            target_speed = min(
                MAX_OPERATIONAL_SPEED,
                target_speed + PERFORMANCE_SPEED_BONUS,
            )
            interventions.append("performance_bonus")

        profile_speed = 0.0
        profile_boost = False
        if (
            self.speed_profile is not None
            and not protected_sector
            and track_pos < 0.80
            and angle < 0.45
            and speed_y < 10.0
        ):
            profile_speed = self.speed_profile.speed_at(distance)
            profile_target = profile_speed * PERFORMANCE_PROFILE_FACTOR
            if profile_target > target_speed:
                target_speed = min(
                    MAX_OPERATIONAL_SPEED, profile_target
                )
                profile_boost = True
                interventions.append("expert_speed_floor")

        if 300.0 <= distance <= FIRST_CORNER_END:
            target_speed = min(
                target_speed, first_corner_speed_cap(distance)
            )
            interventions.append("first_corner_speed")

        if (
            660.0 <= distance <= 710.0
            and signed_track_pos < 0.20
            and speed > 172.0
        ):
            line_correction = clamp(
                0.30 * (0.38 - signed_track_pos),
                0.0,
                0.16,
            )
            steer += line_correction
            target_speed = min(target_speed, 176.0)
            interventions.append("s03_entry_guard")

        if self.corkscrew_rescue:
            rescue_correction = clamp(
                0.30 * (-0.12 - projected_track_pos),
                0.0,
                0.16,
            )
            steer += rescue_correction
            rescue_cap = clamp(
                225.0 - 0.80 * max(0.0, distance - 2300.0),
                185.0,
                225.0,
            )
            target_speed = min(target_speed, rescue_cap)
            interventions.append("corkscrew_rescue")

        if LAST_CORNER_PREPARE_START <= distance <= LAST_CORNER_LINE_END:
            target_track_pos = (
                -0.48
                if distance < LAST_CORNER_BRAKE_START
                else -0.55
            )
            line_correction = clamp(
                0.24 * (target_track_pos - signed_track_pos),
                -0.16,
                0.16,
            )
            steer += line_correction
            interventions.append("last_corner_line")
        if LAST_CORNER_BRAKE_START <= distance <= LAST_CORNER_END:
            target_speed = min(
                target_speed, last_corner_speed_cap(distance)
            )
            interventions.append("last_corner_speed")

        tight_curve = (
            base["front"] < 45.0
            and base["curve_urgency"] > 0.0
            and abs(base["aim_hint"]) >= 0.45
        )
        if tight_curve:
            if protected_sector:
                detected_cap = clamp(
                    65.0 + 0.90 * base["front"],
                    72.0,
                    105.0,
                )
                hold_ticks = TIGHT_CURVE_HOLD_TICKS
            else:
                detected_cap = clamp(
                    87.0 + 1.08 * base["front"],
                    95.0,
                    145.0,
                )
                hold_ticks = FAST_CURVE_HOLD_TICKS
            self.tight_curve_cap = min(
                self.tight_curve_cap, detected_cap
            )
            self.tight_curve_ticks = hold_ticks
        elif self.tight_curve_ticks > 0:
            self.tight_curve_ticks -= 1
            self.tight_curve_cap = min(
                MAX_OPERATIONAL_SPEED,
                self.tight_curve_cap
                + (
                    TIGHT_CURVE_RELEASE_PER_TICK
                    if protected_sector
                    else FAST_CURVE_RELEASE_PER_TICK
                ),
            )
        else:
            self.tight_curve_cap = MAX_OPERATIONAL_SPEED
        if self.tight_curve_ticks > 0 or tight_curve:
            effective_curve_cap = self.tight_curve_cap
            if (
                profile_speed > 0.0
                and not protected_sector
                and track_pos < 0.80
                and angle < 0.45
                and speed_y < 10.0
            ):
                profile_curve_cap = (
                    profile_speed * CURVE_PROFILE_FACTOR
                )
                if profile_curve_cap > effective_curve_cap:
                    effective_curve_cap = profile_curve_cap
                    interventions.append("expert_curve_cap")
            target_speed = min(target_speed, effective_curve_cap)
            interventions.append("tight_curve_hold")

        pedal = BaseSensorPolicy.pedal(speed, target_speed)

        base_pedal = base["pedal"]
        if (
            base_pedal < 0.0
            and pedal > base_pedal
            and not profile_boost
        ):
            pedal = base_pedal
            interventions.append("base_brake_priority")
        moving_outward = signed_track_pos * self.track_pos_rate > 0.0
        if (
            moving_outward
            and track_pos > 0.45
            and abs(projected_track_pos) > 0.72
        ):
            edge_risk = clamp(
                (abs(projected_track_pos) - 0.72) / 0.28,
                0.0,
                1.0,
            )
            steer += -math.copysign(0.10 + 0.18 * edge_risk, signed_track_pos)
            if pedal > 0.0:
                pedal *= 1.0 - 0.75 * edge_risk
            interventions.append("projected_edge")
        if (
            protected_sector
            and moving_outward
            and track_pos > 0.62
            and abs(projected_track_pos) > 0.78
        ):
            protected_risk = clamp(
                (abs(projected_track_pos) - 0.78) / 0.22,
                0.0,
                1.0,
            )
            steer += -math.copysign(
                0.06 + 0.10 * protected_risk,
                signed_track_pos,
            )
            if pedal > 0.0:
                pedal *= 1.0 - 0.35 * protected_risk
            interventions.append("protected_edge")
        if speed > 40.0 and speed_y > 6.0:
            lateral_correction = clamp(
                -0.014 * safe_float(sensors.get("speedY")),
                -0.16,
                0.16,
            )
            steer += lateral_correction
            interventions.append("lateral_steer")
        if track_pos > 0.85 and pedal > 0.0:
            pedal = 0.0
            interventions.append("edge_throttle_cut")
        if track_pos > 0.95:
            pedal = min(pedal, -clamp(0.18 + (track_pos - 0.95) * 2.0, 0.18, 0.55))
            interventions.append("edge_brake")
        if speed_y > 14.0 and pedal > 0.0:
            pedal *= clamp(1.0 - (speed_y - 14.0) / 18.0, 0.0, 1.0)
            interventions.append("lateral_cut")
        if angle > 0.32 and pedal > 0.0:
            pedal *= clamp(1.0 - (angle - 0.32) / 0.35, 0.0, 1.0)
            interventions.append("angle_cut")
        if speed > target_speed + 40.0:
            pedal = min(pedal, -clamp(0.18 + (speed - target_speed - 40.0) / 80.0, 0.18, 0.65))
            interventions.append("overspeed_brake")

        forced_gear = None
        if self.state != NORMAL:
            steer, pedal, forced_gear = self.recovery(sensors)
            interventions.append("recovery")
        return {
            "steer": clamp(steer, -1.0, 1.0),
            "accel": max(0.0, pedal),
            "brake": max(0.0, -pedal),
            "target_speed": target_speed,
            "forced_gear": forced_gear,
            "state": self.state,
            "projected_track_pos": projected_track_pos,
            "track_pos_rate": self.track_pos_rate,
            "tight_curve_cap": self.tight_curve_cap,
            "tight_curve_ticks": self.tight_curve_ticks,
            "profile_speed": profile_speed,
            "track_block": track_block.name,
            "track_block_role": track_block.role,
            "interventions": "+".join(interventions) or "none",
        }


class RuntimePolicy:
    def __init__(self, advisor=None, slow=False, speed_profile=None):
        self.base_policy = BaseSensorPolicy(slow=slow)
        self.advisor = advisor
        self.governor = SafetyGovernor(speed_profile=speed_profile)
        self.adas = SharedADAS()

    def action(self, sensors):
        base = self.base_policy.action_intent(sensors)
        if self.advisor is None:
            advice = {
                "raw_delta_steer": 0.0,
                "raw_delta_speed": 0.0,
                "delta_steer": 0.0,
                "delta_speed": 0.0,
                "neighbor_distance": 0.0,
                "confidence": 0.0,
                "inference_ms": 0.0,
            }
        else:
            advice = self.advisor.predict(sensor_features(sensors, base))
        governed = self.governor.apply(sensors, base, advice)
        action = self.adas.apply(
            sensors,
            governed,
            forced_gear=governed["forced_gear"],
        )
        diagnostics = {
            **base,
            **advice,
            **governed,
            **self.adas.last_diagnostics,
            "base_steer": base["steer"],
            "base_target_speed": base["target_speed"],
        }
        return action, diagnostics


class TraceLogger:
    FIELDS = (
        "step", "curLapTime", "distFromStart", "speedX", "speedY",
        "trackPos", "angle", "damage", "track_block",
        "track_block_role", "front", "near", "openness",
        "aim_hint", "curve_urgency", "track_pos_rate",
        "projected_track_pos",
        "tight_curve_cap", "tight_curve_ticks",
        "profile_speed",
        "base_steer", "base_target_speed", "raw_delta_steer",
        "raw_delta_speed", "delta_steer", "delta_speed", "confidence",
        "neighbor_distance", "target_speed", "governed_steer",
        "filtered_steer_target", "actuator_steer", "front_wheel_speed",
        "rear_wheel_speed", "abs_slip", "traction_slip",
        "final_steer", "final_accel", "final_brake", "final_gear",
        "state", "interventions", "inference_ms",
    )

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.file = open(TRACE_PATH, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()

    def write(self, step, sensors, action, data):
        if step % TRACE_EVERY:
            return
        self.writer.writerow({
            "step": step,
            "curLapTime": safe_float(sensors.get("curLapTime")),
            "distFromStart": safe_float(sensors.get("distFromStart")),
            "speedX": safe_float(sensors.get("speedX")),
            "speedY": safe_float(sensors.get("speedY")),
            "trackPos": safe_float(sensors.get("trackPos")),
            "angle": safe_float(sensors.get("angle")),
            "damage": safe_float(sensors.get("damage")),
            "track_block": data["track_block"],
            "track_block_role": data["track_block_role"],
            "front": data["front"],
            "near": data["near"],
            "openness": data["openness"],
            "aim_hint": data["aim_hint"],
            "curve_urgency": data["curve_urgency"],
            "track_pos_rate": data["track_pos_rate"],
            "projected_track_pos": data["projected_track_pos"],
            "tight_curve_cap": data["tight_curve_cap"],
            "tight_curve_ticks": data["tight_curve_ticks"],
            "profile_speed": data["profile_speed"],
            "base_steer": data["base_steer"],
            "base_target_speed": data["base_target_speed"],
            "raw_delta_steer": data["raw_delta_steer"],
            "raw_delta_speed": data["raw_delta_speed"],
            "delta_steer": data["delta_steer"],
            "delta_speed": data["delta_speed"],
            "confidence": data["confidence"],
            "neighbor_distance": data["neighbor_distance"],
            "target_speed": data["target_speed"],
            "governed_steer": data["governed_steer"],
            "filtered_steer_target": data["filtered_steer_target"],
            "actuator_steer": data["actuator_steer"],
            "front_wheel_speed": data["front_wheel_speed"],
            "rear_wheel_speed": data["rear_wheel_speed"],
            "abs_slip": data["abs_slip"],
            "traction_slip": data["traction_slip"],
            "final_steer": action["steer"],
            "final_accel": action["accel"],
            "final_brake": action["brake"],
            "final_gear": action["gear"],
            "state": data["state"],
            "interventions": data["interventions"],
            "inference_ms": data["inference_ms"],
        })

    def close(self):
        self.file.close()

    @staticmethod
    def archive(reason, clean):
        os.makedirs(TRACE_ARCHIVE_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        status = "clean" if clean else "error"
        safe_reason = "".join(
            character if character.isalnum() else "_"
            for character in reason
        )
        path = os.path.join(
            TRACE_ARCHIVE_DIR,
            "%s_%s_%s.csv" % (timestamp, status, safe_reason),
        )
        shutil.copyfile(TRACE_PATH, path)
        return path


class ValidationMetrics:
    def __init__(self):
        self.previous_lap_time = None
        self.sectors = {
            name: {
                "time": 0.0,
                "speed_sum": 0.0,
                "samples": 0,
                "max_track_pos": 0.0,
            }
            for name, _, _ in VALIDATION_SECTORS
        }

    @staticmethod
    def sector_name(distance):
        return track_block_at(distance).name

    def observe(self, sensors):
        lap_time = safe_float(sensors.get("curLapTime"), -1.0)
        distance = safe_float(sensors.get("distFromStart"))
        name = self.sector_name(distance)
        sector = self.sectors[name]
        sector["speed_sum"] += max(
            0.0, safe_float(sensors.get("speedX"))
        )
        sector["samples"] += 1
        sector["max_track_pos"] = max(
            sector["max_track_pos"],
            abs(safe_float(sensors.get("trackPos"))),
        )
        if self.previous_lap_time is not None:
            elapsed = lap_time - self.previous_lap_time
            if 0.0 < elapsed < 1.0:
                sector["time"] += elapsed
        self.previous_lap_time = lap_time

    def values(self):
        result = {}
        for name, _, _ in VALIDATION_SECTORS:
            sector = self.sectors[name]
            samples = sector["samples"]
            result["%s_time" % name] = sector["time"]
            result["%s_avg_speed" % name] = (
                sector["speed_sum"] / samples if samples else 0.0
            )
            result["%s_max_track_pos" % name] = sector["max_track_pos"]
        return result


def validation_fields():
    fields = [
        "timestamp", "version", "mode", "reason", "lap_time", "clean",
        "max_speed", "offtrack_steps", "recovery_steps",
        "p95_inference_ms",
    ]
    for name, _, _ in VALIDATION_SECTORS:
        fields.extend((
            "%s_time" % name,
            "%s_avg_speed" % name,
            "%s_max_track_pos" % name,
        ))
    return fields


def append_validation(row):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    exists = os.path.exists(VALIDATION_PATH)
    with open(VALIDATION_PATH, "a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=validation_fields())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def validation_summary(rows):
    def elapsed(row):
        official = safe_float(row.get("lap_time"))
        if official > 0.0:
            return official
        return sum(
            safe_float(row.get("%s_time" % name))
            for name, _, _ in VALIDATION_SECTORS
        )

    def completed_lap(row):
        if row.get("reason", "").startswith("lap_complete"):
            return elapsed(row) > 0.0
        return (
            row.get("reason") == "server_closed"
            and safe_float(row.get("offtrack_steps")) == 0.0
            and safe_float(row.get("recovery_steps")) == 0.0
            and elapsed(row) >= 80.0
        )

    completed = [row for row in rows if completed_lap(row)]
    clean = [
        row for row in completed
        if safe_float(row.get("offtrack_steps")) == 0.0
        and safe_float(row.get("recovery_steps")) == 0.0
    ]
    times = [elapsed(row) for row in clean]
    result = {
        "attempts": len(rows),
        "completed": len(completed),
        "clean": len(clean),
        "clean_rate": len(clean) / len(rows) if rows else 0.0,
        "best": min(times) if times else 0.0,
        "mean": statistics.mean(times) if times else 0.0,
        "median": statistics.median(times) if times else 0.0,
        "stdev": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "sectors": [],
    }
    for name, _, _ in VALIDATION_SECTORS:
        speeds = [
            safe_float(row.get("%s_avg_speed" % name))
            for row in clean
        ]
        edges = [
            safe_float(row.get("%s_max_track_pos" % name))
            for row in clean
        ]
        result["sectors"].append({
            "name": name,
            "avg_speed": statistics.median(speeds) if speeds else 0.0,
            "max_track_pos": max(edges) if edges else 0.0,
        })
    return result


def print_validation_report(limit):
    if not os.path.exists(VALIDATION_PATH):
        print("[VALIDATION] Nessun giro registrato.")
        return
    with open(VALIDATION_PATH, newline="", encoding="utf-8") as source:
        rows = [
            row for row in csv.DictReader(source)
            if row.get("version") == DRIVER_VERSION
            and row.get("mode") == "full"
        ][-limit:]
    report = validation_summary(rows)
    print(
        "[VALIDATION] tentativi=%d completati=%d puliti=%d "
        "affidabilita=%.1f%%"
        % (
            report["attempts"],
            report["completed"],
            report["clean"],
            report["clean_rate"] * 100.0,
        )
    )
    if not report["clean"]:
        return
    print(
        "[TEMPI] best=%.3f mediana=%.3f media=%.3f deviazione=%.3f"
        % (
            report["best"],
            report["median"],
            report["mean"],
            report["stdev"],
        )
    )
    print("[SETTORI] velocita_media max_abs_trackPos")
    for sector in report["sectors"]:
        print(
            "  %-18s %7.1f km/h %6.3f"
            % (
                sector["name"],
                sector["avg_speed"],
                sector["max_track_pos"],
            )
        )


def evaluate(dataset):
    base_policy = BaseSensorPolicy()
    rows = []
    all_latencies = []
    for held in dataset.runs:
        train_runs = [
            run for run in dataset.runs if run["run_id"] != held["run_id"]
        ]
        train_x, train_y = dataset.arrays(train_runs, base_policy)
        test_x, _ = dataset.arrays([held], base_policy)
        advisor = ResidualAdvisor(train_x, train_y)
        predictions = []
        for features in test_x:
            result = advisor.predict(features)
            predictions.append(result)
            all_latencies.append(result["inference_ms"])
        base_steer_error = []
        advisor_steer_error = []
        base_pedal_error = []
        advisor_pedal_error = []
        held_rows = held["rows"]
        for row, prediction in zip(held_rows, predictions):
            base = base_policy.action_intent(row)
            expert_pedal = clamp(
                row["accel_action"] - row["brake_action"], -1.0, 1.0
            )
            advised_steer = base["steer"] + prediction["delta_steer"]
            advised_target = (
                base["target_speed"] + prediction["delta_speed"]
            )
            advised_pedal = BaseSensorPolicy.pedal(
                row["speedX"], advised_target
            )
            base_steer_error.append(
                abs(base["steer"] - row["steer_action"])
            )
            advisor_steer_error.append(
                abs(advised_steer - row["steer_action"])
            )
            base_pedal_error.append(abs(base["pedal"] - expert_pedal))
            advisor_pedal_error.append(abs(advised_pedal - expert_pedal))
        rows.append({
            "run_id": held["run_id"],
            "samples": len(held_rows),
            "base_steer_mae": statistics.mean(base_steer_error),
            "advisor_steer_mae": statistics.mean(advisor_steer_error),
            "base_pedal_mae": statistics.mean(base_pedal_error),
            "advisor_pedal_mae": statistics.mean(advisor_pedal_error),
        })
    total = sum(row["samples"] for row in rows)
    aggregate = {"run_id": "ALL", "samples": total}
    for key in (
        "base_steer_mae", "advisor_steer_mae",
        "base_pedal_mae", "advisor_pedal_mae",
    ):
        aggregate[key] = sum(
            row[key] * row["samples"] for row in rows
        ) / total
    aggregate["inference_p95_ms"] = percentile(all_latencies, 0.95)
    rows.append(aggregate)

    sample = dataset.runs[0]["rows"][100]
    full_x, full_y = dataset.arrays(dataset.runs, base_policy)
    action, diagnostics = RuntimePolicy(
        ResidualAdvisor(full_x, full_y),
        speed_profile=ExpertSpeedProfile(dataset.runs),
    ).action(sample)
    assert abs(diagnostics["delta_steer"]) <= MAX_STEER_ADVICE
    assert abs(diagnostics["delta_speed"]) <= MAX_SPEED_ADVICE
    assert not (action["accel"] > 0.01 and action["brake"] > 0.01)
    assert -1.0 <= action["steer"] <= 1.0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(ANALYSIS_PATH, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[-1].keys())
        writer.writeheader()
        writer.writerows(rows)
    return aggregate


def create_client(arguments):
    original = list(sys.argv)
    try:
        sys.argv = [original[0]] + list(arguments)
        return snakeoil3.Client(p=PORT, vision=False)
    finally:
        sys.argv = original


def run_driver(dataset, base_only, slow, snakeoil_arguments):
    advisor = None
    speed_profile = None
    if not base_only:
        train_x, train_y = dataset.arrays(
            dataset.runs, BaseSensorPolicy(slow=slow)
        )
        advisor = ResidualAdvisor(train_x, train_y)
        if not slow:
            speed_profile = ExpertSpeedProfile(dataset.runs)
    policy = RuntimePolicy(
        advisor=advisor,
        slow=slow,
        speed_profile=speed_profile,
    )
    trace = TraceLogger()
    validation = ValidationMetrics()
    client = create_client(snakeoil_arguments)
    reason = "max_steps"
    summary = {
        "max_distance": 0.0,
        "max_speed": 0.0,
        "offtrack_steps": 0,
        "recovery_steps": 0,
        "latencies": [],
        "lap_time": 0.0,
    }
    try:
        for step in range(MAX_STEPS):
            client.get_servers_input()
            if not client.so:
                reason = "server_closed"
                break
            sensors = client.S.d
            last_lap_time = safe_float(sensors.get("lastLapTime"))
            if last_lap_time > 0.0:
                summary["lap_time"] = last_lap_time
                reason = "lap_complete"
                break
            validation.observe(sensors)
            action, diagnostics = policy.action(sensors)
            client.R.d.update(action)
            trace.write(step, sensors, action, diagnostics)
            summary["max_distance"] = max(
                summary["max_distance"],
                safe_float(sensors.get("distRaced")),
            )
            summary["max_speed"] = max(
                summary["max_speed"],
                safe_float(sensors.get("speedX")),
            )
            summary["offtrack_steps"] += int(
                abs(safe_float(sensors.get("trackPos"))) >= 1.0
            )
            summary["recovery_steps"] += int(
                diagnostics["state"] != NORMAL
            )
            summary["latencies"].append(diagnostics["inference_ms"])
            client.respond_to_server()
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    finally:
        trace.close()
        trace_clean = (
            summary["offtrack_steps"] == 0
            and summary["recovery_steps"] == 0
        )
        archived_trace = trace.archive(reason, trace_clean)
        client.shutdown()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        exists = os.path.exists(RUNS_PATH)
        with open(RUNS_PATH, "a", newline="", encoding="utf-8") as output:
            fields = (
                "timestamp", "version", "mode", "reason", "max_distance",
                "max_speed", "offtrack_steps", "recovery_steps",
                "p95_inference_ms",
            )
            writer = csv.DictWriter(output, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": DRIVER_VERSION,
                "mode": "slow" if slow else ("base" if base_only else "full"),
                "reason": reason,
                "max_distance": summary["max_distance"],
                "max_speed": summary["max_speed"],
                "offtrack_steps": summary["offtrack_steps"],
                "recovery_steps": summary["recovery_steps"],
                "p95_inference_ms": percentile(summary["latencies"], 0.95),
            })
        clean = int(
            reason == "lap_complete"
            and summary["offtrack_steps"] == 0
            and summary["recovery_steps"] == 0
        )
        append_validation({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": DRIVER_VERSION,
            "mode": "slow" if slow else ("base" if base_only else "full"),
            "reason": reason,
            "lap_time": summary["lap_time"],
            "clean": clean,
            "max_speed": summary["max_speed"],
            "offtrack_steps": summary["offtrack_steps"],
            "recovery_steps": summary["recovery_steps"],
            "p95_inference_ms": percentile(summary["latencies"], 0.95),
            **validation.values(),
        })
        print("[STOP] %s" % reason)
        print("[TRACE] %s" % TRACE_PATH)
        print("[TRACE ARCHIVE] %s" % archived_trace)
        print(
            "[RUN] lap=%.3f distance=%.1f speed=%.1f "
            "offtrack=%d recovery=%d"
            % (
                summary["lap_time"],
                summary["max_distance"],
                summary["max_speed"],
                summary["offtrack_steps"],
                summary["recovery_steps"],
            )
        )


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="V8 TORCS: base sensoriale + advisor KNN residuale."
    )
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--slow", action="store_true")
    parser.add_argument("--validation-report", action="store_true")
    parser.add_argument("--validation-runs", type=int, default=10)
    return parser.parse_known_args(argv)


def main():
    arguments, snakeoil_arguments = parse_arguments()
    if arguments.validation_report:
        print_validation_report(max(1, arguments.validation_runs))
        return
    dataset = DemonstrationDataset()
    print(
        "[DATASET] %d giri puliti, %d campioni."
        % (
            len(dataset.runs),
            sum(len(run["rows"]) for run in dataset.runs),
        )
    )
    if arguments.analyze_only:
        result = evaluate(dataset)
        print(
            "[LOO] steer base %.5f advisor %.5f; "
            "pedal base %.5f advisor %.5f; p95 %.3f ms"
            % (
                result["base_steer_mae"],
                result["advisor_steer_mae"],
                result["base_pedal_mae"],
                result["advisor_pedal_mae"],
                result["inference_p95_ms"],
            )
        )
        print("[REPORT] %s" % ANALYSIS_PATH)
        if result["inference_p95_ms"] >= 1.5:
            raise RuntimeError("Inferenza KNN oltre il gate di 1.5 ms.")
        return
    run_driver(
        dataset,
        base_only=arguments.base_only,
        slow=arguments.slow,
        snakeoil_arguments=snakeoil_arguments,
    )


if __name__ == "__main__":
    main()
