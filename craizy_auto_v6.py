"""V6: sensor-based hybrid KNN driver for TORCS.

The learned policy uses only live vehicle and track sensors.  Track position
along the lap is intentionally excluded, so the controller is not a spatial
replay of Corkscrew.
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
import warnings
from collections import defaultdict, deque

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

import snakeoil3_jm2 as snakeoil3


DRIVER_VERSION = "craizy_auto_v6_sensor_knn_brake_guard_v2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "torcs_ps4_dataset.csv")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRACE_PATH = os.path.join(LOGS_DIR, "auto_v6_latest.csv")
PREVIOUS_TRACE_PATH = os.path.join(LOGS_DIR, "auto_v6_previous.csv")
RUNS_PATH = os.path.join(RESULTS_DIR, "auto_v6_runs.csv")
ANALYSIS_PATH = os.path.join(RESULTS_DIR, "auto_v6_analysis.csv")

PORT = 3001
MAX_STEPS = 100000
KNN_NEIGHBORS = 7
CONFIDENCE_FULL_DISTANCE = 0.263
CONFIDENCE_ZERO_DISTANCE = 0.697
STEER_RATE_LIMIT = 0.04
PEDAL_RATE_LIMIT = 0.16
TRACE_EVERY = 5

BRAKE_GUARD_MIN_SPEED = 220.0
BRAKE_GUARD_MAX_VISIBILITY = 145.0
BRAKE_GUARD_CLOSING_RATE = 0.20
BRAKE_GUARD_OPEN_VISIBILITY = 94.0
BRAKE_GUARD_VISIBILITY_DROP = 8.0
BRAKE_GUARD_CORNER_SPEED = 90.0
BRAKE_GUARD_DECELERATION = 14.0
BRAKE_GUARD_MARGIN = 15.0
BRAKE_GUARD_SPEED_MARGIN = 10.0
BRAKE_GUARD_HOLD_TICKS = 8
BRAKE_GUARD_CONTEXT_TICKS = 220
BRAKE_RELEASE_MARGIN = 3.0
MAX_OPERATIONAL_SPEED = 215.0
CURVE_SAFE_SPEED_FACTOR = 0.90
SPEED_CAP_BRAKE_GAIN = 0.025

EDGE_GUARD_MIN_TRACKPOS = 0.35
EDGE_GUARD_PROJECT_TICKS = 12
EDGE_GUARD_PROJECT_LIMIT = 0.72
EDGE_GUARD_COUNTERSTEER_START = 0.45
EDGE_GUARD_MIN_SPEED = 60.0
EDGE_GUARD_MIN_STEER = 0.45
EDGE_GUARD_STEER_RATE = 0.07

MIN_CLEAN_LAPS = 3
MIN_LAP_ROWS = 800
MIN_LAP_DISTANCE = 3500.0
OFFTRACK_CONFIRM_TICKS = 3

GEAR_UP_SPEEDS = (27.0, 54.0, 90.0, 120.0, 150.0)
GEAR_DOWN_SPEEDS = (22.0, 48.0, 80.0, 110.0, 140.0)

NORMAL = "NORMAL"
STABILIZE = "STABILIZE"
REVERSE = "REVERSE"
FORWARD_ALIGN = "FORWARD_ALIGN"
REJOIN = "REJOIN"

WHEEL_RADIUS = 0.33

REQUIRED_COLUMNS = (
    "run_id",
    "step",
    "steer_action",
    "accel_action",
    "brake_action",
    "gear_action",
    "speedX",
    "speedY",
    "wheelSpinVel",
    "track",
    "trackPos",
    "angle",
    "rpm",
    "damage",
    "distFromStart",
)

NUMERIC_FIELDS = (
    "step",
    "curLapTime",
    "steer_action",
    "accel_action",
    "brake_action",
    "gear_action",
    "speedX",
    "speedY",
    "speedZ",
    "trackPos",
    "angle",
    "rpm",
    "damage",
    "distFromStart",
)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


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
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * clamp(fraction, 0.0, 1.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_row(raw):
    row = {"run_id": str(raw.get("run_id", "")).strip()}
    for field in NUMERIC_FIELDS:
        row[field] = safe_float(raw.get(field))
    row["track"] = safe_list(raw.get("track"), 19, -1.0)
    row["wheelSpinVel"] = safe_list(raw.get("wheelSpinVel"), 4, 0.0)
    return row


def wheel_slip(speed_x, wheel_spin):
    vehicle_ms = max(abs(speed_x) / 3.6, 1.0)
    wheel_ms = [abs(value) * WHEEL_RADIUS for value in wheel_spin]
    driven_ms = statistics.mean(wheel_ms[2:4])
    return (driven_ms - vehicle_ms) / vehicle_ms


def wheel_spin_spread(wheel_spin):
    if not wheel_spin:
        return 0.0
    return max(wheel_spin) - min(wheel_spin)


def feature_vector(sensors):
    track = safe_list(sensors.get("track"), 19, -1.0)
    wheels = safe_list(sensors.get("wheelSpinVel"), 4, 0.0)
    speed_x = safe_float(sensors.get("speedX"))
    features = [clamp(value, 0.0, 200.0) / 200.0 for value in track]
    features.extend(
        (
            clamp(safe_float(sensors.get("trackPos")), -2.0, 2.0),
            clamp(safe_float(sensors.get("angle")) / math.pi, -1.0, 1.0),
            clamp(speed_x / 300.0, -0.2, 1.2),
            clamp(safe_float(sensors.get("speedY")) / 100.0, -1.0, 1.0),
            clamp(safe_float(sensors.get("rpm")) / 10000.0, 0.0, 1.5),
            clamp(wheel_spin_spread(wheels) / 100.0, 0.0, 2.0),
        )
    )
    return np.asarray(features, dtype=np.float64)


def signed_pedal(row):
    return clamp(row["accel_action"] - row["brake_action"], -1.0, 1.0)


class DatasetError(RuntimeError):
    pass


class DemonstrationDataset:
    def __init__(self, path=DATASET_PATH):
        self.path = path
        self.runs = []
        self.rows = []
        self.features = None
        self.targets = None
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            raise DatasetError("Dataset non trovato: %s" % self.path)
        with open(self.path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = reader.fieldnames or []
            missing = [field for field in REQUIRED_COLUMNS if field not in fields]
            if missing:
                raise DatasetError(
                    "Schema post-ADAS non valido; colonne mancanti: %s"
                    % ", ".join(missing)
                )
            groups = defaultdict(list)
            for raw in reader:
                row = parse_row(raw)
                if row["run_id"]:
                    groups[row["run_id"]].append(row)

        rejected = []
        for run_id, rows in groups.items():
            rows.sort(key=lambda row: row["step"])
            reason = self._invalid_reason(rows)
            if reason:
                rejected.append("%s: %s" % (run_id, reason))
            else:
                self.runs.append({"run_id": run_id, "rows": rows})

        if len(self.runs) < MIN_CLEAN_LAPS:
            raise DatasetError(
                "Servono almeno %d giri puliti; trovati %d. %s"
                % (MIN_CLEAN_LAPS, len(self.runs), "; ".join(rejected))
            )

        self.rows = [
            row for run in self.runs for row in run["rows"]
        ]
        self.features = np.vstack(
            [feature_vector(row) for row in self.rows]
        )
        self.targets = np.asarray(
            [
                (clamp(row["steer_action"], -1.0, 1.0), signed_pedal(row))
                for row in self.rows
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _invalid_reason(rows):
        if len(rows) < MIN_LAP_ROWS:
            return "campioni insufficienti"
        if max(row["distFromStart"] for row in rows) < MIN_LAP_DISTANCE:
            return "distanza insufficiente"
        if max(row["damage"] for row in rows) > 0.0:
            return "danno non nullo"
        bad_ticks = 0
        for row in rows:
            bad = abs(row["trackPos"]) >= 1.0 or min(row["track"]) < 0.0
            bad_ticks = bad_ticks + 1 if bad else 0
            if bad_ticks >= OFFTRACK_CONFIRM_TICKS:
                return "uscita pista confermata"
        if any(
            abs(row["steer_action"]) > 1.000001
            or not 0.0 <= row["accel_action"] <= 1.000001
            or not 0.0 <= row["brake_action"] <= 1.000001
            for row in rows
        ):
            return "azione fuori limite"
        return ""


class SensorKNN:
    def __init__(self, features, targets):
        if len(features) < KNN_NEIGHBORS:
            raise DatasetError("Campioni insufficienti per KNN k=7.")
        self.model = KNeighborsRegressor(
            n_neighbors=KNN_NEIGHBORS,
            weights="distance",
            metric="euclidean",
            n_jobs=1,
        )
        self.model.fit(features, targets)
        # Pay sklearn/joblib initialization before the TORCS control loop.
        warmup = np.asarray(features[:1], dtype=np.float64)
        self.model.kneighbors(warmup, return_distance=True)
        self.model.predict(warmup)

    @staticmethod
    def confidence(mean_distance):
        if mean_distance <= CONFIDENCE_FULL_DISTANCE:
            return 1.0
        if mean_distance >= CONFIDENCE_ZERO_DISTANCE:
            return 0.0
        span = CONFIDENCE_ZERO_DISTANCE - CONFIDENCE_FULL_DISTANCE
        return 1.0 - (mean_distance - CONFIDENCE_FULL_DISTANCE) / span

    def predict(self, features):
        vector = np.asarray(features, dtype=np.float64).reshape(1, -1)
        start = time.perf_counter()
        distances, _ = self.model.kneighbors(vector, return_distance=True)
        prediction = self.model.predict(vector)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        mean_distance = float(np.mean(distances[0]))
        return {
            "steer": clamp(float(prediction[0]), -1.0, 1.0),
            "pedal": clamp(float(prediction[1]), -1.0, 1.0),
            "distance": mean_distance,
            "confidence": self.confidence(mean_distance),
            "inference_ms": elapsed_ms,
        }


class GearController:
    def __init__(self):
        self.gear = 1

    def update(self, speed_x, current_gear):
        current = int(safe_float(current_gear, self.gear))
        if 1 <= current <= 6:
            self.gear = current
        self.gear = int(clamp(self.gear, 1, 6))
        if self.gear < 6 and speed_x >= GEAR_UP_SPEEDS[self.gear - 1]:
            self.gear += 1
        elif self.gear > 1 and speed_x < GEAR_DOWN_SPEEDS[self.gear - 2]:
            self.gear -= 1
        return self.gear


class SafetyController:
    def __init__(self):
        self.previous_steer = 0.0
        self.previous_pedal = 0.0
        self.state = NORMAL
        self.state_ticks = 0
        self.bad_ticks = 0
        self.stuck_ticks = 0
        self.stable_ticks = 0
        self.recovery_cooldown = 0
        self.recovery_reason = ""
        self.previous_visibility = None
        self.visibility_closing_rate = 0.0
        self.visibility_history = deque(maxlen=50)
        self.brake_guard_ticks = 0
        self.brake_guard_pedal = 0.0
        self.brake_context_ticks = 0
        self.previous_track_pos = None
        self.track_pos_rate = 0.0
        self.gears = GearController()

    @staticmethod
    def fallback_action(sensors):
        track = safe_list(sensors.get("track"), 19, 0.0)
        angle = safe_float(sensors.get("angle"))
        track_pos = safe_float(sensors.get("trackPos"))
        speed_x = max(0.0, safe_float(sensors.get("speedX")))
        left = statistics.mean(track[0:9])
        right = statistics.mean(track[10:19])
        direction = clamp((right - left) / 200.0, -0.25, 0.25)
        steer = clamp(-1.10 * angle + 0.45 * track_pos + direction, -0.65, 0.65)
        front = max(0.0, track[9])
        side_min = max(0.0, min(track[7:12]))
        if front > 120.0 and side_min > 55.0:
            target_speed = 155.0
        elif front > 80.0:
            target_speed = 120.0
        elif front > 45.0:
            target_speed = 90.0
        else:
            target_speed = 65.0
        target_speed -= 25.0 * min(1.0, abs(steer))
        error = target_speed - speed_x
        if error > 15.0:
            pedal = 0.65
        elif error > 3.0:
            pedal = 0.30
        elif error > -12.0:
            pedal = 0.0
        else:
            pedal = -clamp((-error) / 80.0, 0.10, 0.45)
        return steer, pedal

    @staticmethod
    def recovery_steer(sensors):
        angle = safe_float(sensors.get("angle"))
        track_pos = safe_float(sensors.get("trackPos"))
        return clamp(-1.25 * angle + 0.70 * track_pos, -0.65, 0.65)

    def _predictive_brake_guard(self, sensors):
        track = safe_list(sensors.get("track"), 19, 0.0)
        speed_x = max(0.0, safe_float(sensors.get("speedX")))
        visibility = statistics.median(track[8:11])
        if self.previous_visibility is None:
            closing = 0.0
        else:
            closing = self.previous_visibility - visibility
        self.previous_visibility = visibility
        self.visibility_closing_rate = (
            0.85 * self.visibility_closing_rate + 0.15 * closing
        )
        self.visibility_history.append(visibility)
        recent_peak = max(self.visibility_history)
        visibility_drop = recent_peak - visibility

        usable_distance = max(0.0, visibility - BRAKE_GUARD_MARGIN)
        corner_speed_ms = BRAKE_GUARD_CORNER_SPEED / 3.6
        physical_safe_speed = 3.6 * math.sqrt(
            corner_speed_ms * corner_speed_ms
            + 2.0 * BRAKE_GUARD_DECELERATION * usable_distance
        )
        safe_speed = min(
            MAX_OPERATIONAL_SPEED,
            physical_safe_speed * CURVE_SAFE_SPEED_FACTOR,
        )
        overspeed = speed_x - safe_speed
        trigger = (
            speed_x >= BRAKE_GUARD_MIN_SPEED
            and visibility <= BRAKE_GUARD_MAX_VISIBILITY
            and self.visibility_closing_rate >= BRAKE_GUARD_CLOSING_RATE
            and recent_peak >= BRAKE_GUARD_OPEN_VISIBILITY
            and visibility_drop >= BRAKE_GUARD_VISIBILITY_DROP
            and overspeed >= BRAKE_GUARD_SPEED_MARGIN
        )
        if trigger:
            strength = clamp(0.18 + overspeed / 70.0, 0.18, 0.85)
            self.brake_guard_pedal = -strength
            self.brake_guard_ticks = BRAKE_GUARD_HOLD_TICKS
            self.brake_context_ticks = BRAKE_GUARD_CONTEXT_TICKS
        elif self.brake_guard_ticks > 0:
            self.brake_guard_ticks -= 1
            if speed_x > safe_speed - 5.0:
                strength = clamp(0.12 + max(0.0, overspeed) / 80.0, 0.12, 0.70)
                self.brake_guard_pedal = min(
                    self.brake_guard_pedal,
                    -strength,
                )
            else:
                self.brake_guard_ticks = 0
                self.brake_guard_pedal = 0.0
        else:
            self.brake_guard_pedal = 0.0
        if self.brake_context_ticks > 0:
            self.brake_context_ticks -= 1
        return {
            "visibility": visibility,
            "closing_rate": self.visibility_closing_rate,
            "recent_peak": recent_peak,
            "visibility_drop": visibility_drop,
            "safe_speed": safe_speed,
            "physical_safe_speed": physical_safe_speed,
            "active": self.brake_guard_ticks > 0,
            "context_active": self.brake_context_ticks > 0,
            "pedal": self.brake_guard_pedal,
        }

    def _edge_guard(self, sensors, requested_steer):
        track_pos = safe_float(sensors.get("trackPos"))
        speed_x = max(0.0, safe_float(sensors.get("speedX")))
        if self.previous_track_pos is None:
            raw_rate = 0.0
        else:
            raw_rate = track_pos - self.previous_track_pos
        self.previous_track_pos = track_pos
        self.track_pos_rate = 0.65 * self.track_pos_rate + 0.35 * raw_rate
        projected = (
            track_pos + EDGE_GUARD_PROJECT_TICKS * self.track_pos_rate
        )
        moving_outward = track_pos * self.track_pos_rate > 0.0
        steering_outward = requested_steer * track_pos < 0.0
        active = (
            speed_x >= EDGE_GUARD_MIN_SPEED
            and abs(track_pos) >= EDGE_GUARD_MIN_TRACKPOS
            and moving_outward
            and steering_outward
            and abs(requested_steer) >= EDGE_GUARD_MIN_STEER
            and abs(projected) >= EDGE_GUARD_PROJECT_LIMIT
        )
        target = requested_steer
        if active:
            strength = clamp(
                (
                    abs(track_pos) - EDGE_GUARD_MIN_TRACKPOS
                ) / (
                    1.0 - EDGE_GUARD_MIN_TRACKPOS
                ),
                0.0,
                1.0,
            )
            target = track_pos * (0.35 + 0.20 * strength)
        return {
            "active": active,
            "track_pos_rate": self.track_pos_rate,
            "projected_track_pos": projected,
            "target": clamp(target, -0.45, 0.45) if active else target,
        }

    def _set_state(self, state):
        if state != self.state:
            self.state = state
            self.state_ticks = 0
            self.stable_ticks = 0

    def _update_detection(self, sensors):
        speed_x = abs(safe_float(sensors.get("speedX")))
        speed_y = abs(safe_float(sensors.get("speedY")))
        track_pos = abs(safe_float(sensors.get("trackPos")))
        angle = abs(safe_float(sensors.get("angle")))
        track = safe_list(sensors.get("track"), 19, -1.0)
        bad = (
            track_pos > 1.0
            or angle > 0.75
            or speed_y > 30.0
            or min(track) < 0.0
        )
        extreme = track_pos > 1.35 or angle > 1.15
        self.bad_ticks = self.bad_ticks + 1 if bad else 0
        if speed_x < 3.0:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0
        if self.recovery_cooldown > 0:
            self.recovery_cooldown -= 1
        if (
            self.state == NORMAL
            and self.recovery_cooldown == 0
            and (extreme or self.bad_ticks >= 5 or self.stuck_ticks >= 125)
        ):
            if self.stuck_ticks >= 125:
                self.recovery_reason = "stuck"
            elif extreme:
                self.recovery_reason = "extreme"
            else:
                self.recovery_reason = "unstable"
            self._set_state(STABILIZE)

    def _recovery_action(self, sensors):
        speed_x = safe_float(sensors.get("speedX"))
        angle = abs(safe_float(sensors.get("angle")))
        track_pos = abs(safe_float(sensors.get("trackPos")))
        forward_steer = self.recovery_steer(sensors)
        self.state_ticks += 1

        if self.state == STABILIZE:
            steer = forward_steer
            pedal = -0.18 if abs(speed_x) > 18.0 else 0.0
            gear = self.gears.update(max(0.0, speed_x), sensors.get("gear"))
            stable = track_pos < 0.92 and angle < 0.35 and abs(speed_x) > 5.0
            self.stable_ticks = self.stable_ticks + 1 if stable else 0
            if self.stable_ticks >= 15:
                self._set_state(REJOIN)
            elif (
                self.state_ticks >= 35
                and abs(speed_x) < 6.0
                and (
                    self.recovery_reason == "stuck"
                    or track_pos > 0.95
                    or angle > 0.45
                )
            ):
                self._set_state(REVERSE)
            return steer, pedal, gear

        if self.state == REVERSE:
            steer = -forward_steer
            pedal = 0.35
            gear = -1
            if (
                self.state_ticks >= 35
                and (angle < 0.42 or self.state_ticks >= 110)
            ):
                self._set_state(FORWARD_ALIGN)
            return steer, pedal, gear

        if self.state == FORWARD_ALIGN:
            steer = forward_steer
            pedal = 0.28 if angle < 0.75 else 0.16
            gear = 1
            stable = track_pos < 0.92 and angle < 0.30 and speed_x > 10.0
            self.stable_ticks = self.stable_ticks + 1 if stable else 0
            if self.stable_ticks >= 15:
                self._set_state(REJOIN)
            elif self.state_ticks >= 140:
                self._set_state(STABILIZE)
            return steer, pedal, gear

        # REJOIN blends back into the learned policy in action().
        return forward_steer, 0.22, self.gears.update(
            max(0.0, speed_x), sensors.get("gear")
        )

    def action(self, sensors, prediction):
        self._update_detection(sensors)
        fallback_steer, fallback_pedal = self.fallback_action(sensors)
        confidence = prediction["confidence"]
        learned_steer = prediction["steer"]
        learned_pedal = prediction["pedal"]
        requested_steer = (
            confidence * learned_steer + (1.0 - confidence) * fallback_steer
        )
        requested_pedal = (
            confidence * learned_pedal + (1.0 - confidence) * fallback_pedal
        )

        interventions = []
        speed_x = safe_float(sensors.get("speedX"))
        speed_y = safe_float(sensors.get("speedY"))
        angle = safe_float(sensors.get("angle"))
        track_pos = safe_float(sensors.get("trackPos"))
        slip = wheel_slip(
            speed_x, safe_list(sensors.get("wheelSpinVel"), 4, 0.0)
        )
        brake_guard = self._predictive_brake_guard(sensors)
        if brake_guard["active"] and self.state == NORMAL:
            requested_pedal = min(requested_pedal, brake_guard["pedal"])
            interventions.append("predictive_brake")
        elif (
            brake_guard["context_active"]
            and self.state == NORMAL
            and requested_pedal < 0.0
            and speed_x <= brake_guard["safe_speed"] + BRAKE_RELEASE_MARGIN
        ):
            requested_pedal = 0.0
            interventions.append("brake_release")

        if speed_x > MAX_OPERATIONAL_SPEED and self.state == NORMAL:
            overspeed = speed_x - MAX_OPERATIONAL_SPEED
            requested_pedal = min(
                requested_pedal,
                -clamp(
                    0.10 + overspeed * SPEED_CAP_BRAKE_GAIN,
                    0.10,
                    0.45,
                ),
            )
            interventions.append("speed_cap")

        # Mild sensor feedback protects the learned action without replacing it.
        danger = max(
            clamp((abs(track_pos) - 0.70) / 0.30, 0.0, 1.0),
            clamp((abs(angle) - 0.25) / 0.45, 0.0, 1.0),
            clamp((abs(speed_y) - 12.0) / 22.0, 0.0, 1.0),
        )
        if danger > 0.0:
            correction = clamp(
                -0.55 * angle + 0.30 * track_pos - 0.002 * speed_y,
                -0.10,
                0.10,
            )
            requested_steer += danger * correction
            if requested_pedal > 0.0:
                requested_pedal *= 1.0 - 0.65 * danger
            interventions.append("stability")

        edge_guard = self._edge_guard(sensors, requested_steer)
        if edge_guard["active"] and self.state == NORMAL:
            requested_steer = edge_guard["target"]
            if requested_pedal > 0.0:
                requested_pedal = 0.0
            interventions.append("edge_guard")

        if requested_pedal > 0.0 and speed_x > 20.0 and slip > 0.18:
            requested_pedal *= clamp(1.0 - (slip - 0.18), 0.25, 1.0)
            interventions.append("tcs")

        gear = self.gears.update(max(0.0, speed_x), sensors.get("gear"))
        if self.state != NORMAL:
            recovery_steer, recovery_pedal, gear = self._recovery_action(sensors)
            if self.state == REJOIN:
                blend = clamp(self.state_ticks / 60.0, 0.0, 1.0)
                requested_steer = (
                    (1.0 - blend) * recovery_steer + blend * requested_steer
                )
                requested_pedal = (
                    (1.0 - blend) * recovery_pedal + blend * requested_pedal
                )
                if (
                    self.state_ticks >= 60
                    and abs(track_pos) < 0.80
                    and abs(angle) < 0.22
                ):
                    self._set_state(NORMAL)
                    self.recovery_cooldown = 100
                    self.bad_ticks = 0
                    self.stuck_ticks = 0
                    self.recovery_reason = ""
            else:
                requested_steer = recovery_steer
                requested_pedal = recovery_pedal
            interventions.append("recovery")

        track = safe_list(sensors.get("track"), 19, 0.0)
        straight_high_speed = (
            speed_x > 100.0
            and track[9] > 100.0
            and abs(requested_steer) < 0.20
            and self.state == NORMAL
        )
        steer_smoothing = 0.25 if straight_high_speed else 0.55
        requested_steer = self.previous_steer + steer_smoothing * (
            requested_steer - self.previous_steer
        )
        if straight_high_speed:
            interventions.append("straight_smoothing")

        steer_rate_limit = (
            EDGE_GUARD_STEER_RATE
            if edge_guard["active"] and self.state == NORMAL
            else STEER_RATE_LIMIT
        )
        steer_delta = clamp(
            requested_steer - self.previous_steer,
            -steer_rate_limit,
            steer_rate_limit,
        )
        pedal_delta = clamp(
            requested_pedal - self.previous_pedal,
            -PEDAL_RATE_LIMIT,
            PEDAL_RATE_LIMIT,
        )
        final_steer = clamp(self.previous_steer + steer_delta, -1.0, 1.0)
        final_pedal = clamp(self.previous_pedal + pedal_delta, -1.0, 1.0)
        self.previous_steer = final_steer
        self.previous_pedal = final_pedal

        accel = max(0.0, final_pedal)
        brake = max(0.0, -final_pedal)
        if brake > 0.01:
            accel = 0.0
        action = {
            "steer": final_steer,
            "accel": clamp(accel, 0.0, 1.0),
            "brake": clamp(brake, 0.0, 1.0),
            "gear": int(clamp(gear, -1, 6)),
        }
        diagnostics = {
            **prediction,
            "fallback_steer": fallback_steer,
            "fallback_pedal": fallback_pedal,
            "requested_steer": requested_steer,
            "requested_pedal": requested_pedal,
            "steer_smoothing": steer_smoothing,
            "final_pedal": final_pedal,
            "slip": slip,
            "danger": danger,
            "track_pos_rate": edge_guard["track_pos_rate"],
            "projected_track_pos": edge_guard["projected_track_pos"],
            "edge_guard": int(edge_guard["active"]),
            "edge_guard_target": edge_guard["target"],
            "visibility": brake_guard["visibility"],
            "closing_rate": brake_guard["closing_rate"],
            "recent_peak": brake_guard["recent_peak"],
            "visibility_drop": brake_guard["visibility_drop"],
            "safe_speed": brake_guard["safe_speed"],
            "physical_safe_speed": brake_guard["physical_safe_speed"],
            "brake_guard": int(brake_guard["active"]),
            "brake_context": int(brake_guard["context_active"]),
            "brake_guard_pedal": brake_guard["pedal"],
            "state": self.state,
            "recovery_reason": self.recovery_reason,
            "state_ticks": self.state_ticks,
            "bad_ticks": self.bad_ticks,
            "stuck_ticks": self.stuck_ticks,
            "interventions": "+".join(interventions) or "none",
        }
        return action, diagnostics


class TraceLogger:
    FIELDS = (
        "step",
        "curLapTime",
        "distFromStart",
        "speedX",
        "speedY",
        "trackPos",
        "angle",
        "damage",
        "knn_steer",
        "knn_pedal",
        "neighbor_distance",
        "confidence",
        "fallback_steer",
        "fallback_pedal",
        "requested_steer",
        "requested_pedal",
        "steer_smoothing",
        "final_steer",
        "final_accel",
        "final_brake",
        "final_gear",
        "slip",
        "danger",
        "track_pos_rate",
        "projected_track_pos",
        "edge_guard",
        "edge_guard_target",
        "visibility",
        "closing_rate",
        "recent_peak",
        "visibility_drop",
        "safe_speed",
        "physical_safe_speed",
        "brake_guard",
        "brake_context",
        "brake_guard_pedal",
        "state",
        "recovery_reason",
        "state_ticks",
        "bad_ticks",
        "stuck_ticks",
        "interventions",
        "inference_ms",
        "offtrack",
    )

    def __init__(self):
        os.makedirs(LOGS_DIR, exist_ok=True)
        if os.path.exists(TRACE_PATH):
            os.replace(TRACE_PATH, PREVIOUS_TRACE_PATH)
        self.file = open(TRACE_PATH, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()

    def write(self, step, sensors, action, data):
        if step % TRACE_EVERY:
            return
        row = {
            "step": step,
            "curLapTime": safe_float(sensors.get("curLapTime")),
            "distFromStart": safe_float(sensors.get("distFromStart")),
            "speedX": safe_float(sensors.get("speedX")),
            "speedY": safe_float(sensors.get("speedY")),
            "trackPos": safe_float(sensors.get("trackPos")),
            "angle": safe_float(sensors.get("angle")),
            "damage": safe_float(sensors.get("damage")),
            "knn_steer": data["steer"],
            "knn_pedal": data["pedal"],
            "neighbor_distance": data["distance"],
            "confidence": data["confidence"],
            "fallback_steer": data["fallback_steer"],
            "fallback_pedal": data["fallback_pedal"],
            "requested_steer": data["requested_steer"],
            "requested_pedal": data["requested_pedal"],
            "steer_smoothing": data["steer_smoothing"],
            "final_steer": action["steer"],
            "final_accel": action["accel"],
            "final_brake": action["brake"],
            "final_gear": action["gear"],
            "slip": data["slip"],
            "danger": data["danger"],
            "track_pos_rate": data["track_pos_rate"],
            "projected_track_pos": data["projected_track_pos"],
            "edge_guard": data["edge_guard"],
            "edge_guard_target": data["edge_guard_target"],
            "visibility": data["visibility"],
            "closing_rate": data["closing_rate"],
            "recent_peak": data["recent_peak"],
            "visibility_drop": data["visibility_drop"],
            "safe_speed": data["safe_speed"],
            "physical_safe_speed": data["physical_safe_speed"],
            "brake_guard": data["brake_guard"],
            "brake_context": data["brake_context"],
            "brake_guard_pedal": data["brake_guard_pedal"],
            "state": data["state"],
            "recovery_reason": data["recovery_reason"],
            "state_ticks": data["state_ticks"],
            "bad_ticks": data["bad_ticks"],
            "stuck_ticks": data["stuck_ticks"],
            "interventions": data["interventions"],
            "inference_ms": data["inference_ms"],
            "offtrack": int(abs(safe_float(sensors.get("trackPos"))) >= 1.0),
        }
        self.writer.writerow(row)

    def close(self):
        self.file.close()


class RunSummary:
    FIELDS = (
        "timestamp",
        "version",
        "reason",
        "steps",
        "max_distance",
        "max_speed",
        "damage_final",
        "offtrack_steps",
        "recovery_steps",
        "mean_confidence",
        "p95_inference_ms",
        "last_lap_time",
    )

    def __init__(self):
        self.steps = 0
        self.max_distance = 0.0
        self.max_speed = 0.0
        self.damage = 0.0
        self.offtrack = 0
        self.recovery = 0
        self.confidences = []
        self.latencies = []
        self.last_lap_time = 0.0

    def record(self, sensors, diagnostics):
        self.steps += 1
        self.max_distance = max(
            self.max_distance, safe_float(sensors.get("distRaced"))
        )
        self.max_speed = max(self.max_speed, safe_float(sensors.get("speedX")))
        self.damage = safe_float(sensors.get("damage"))
        self.offtrack += int(abs(safe_float(sensors.get("trackPos"))) >= 1.0)
        self.recovery += int(diagnostics["state"] != NORMAL)
        self.confidences.append(diagnostics["confidence"])
        self.latencies.append(diagnostics["inference_ms"])
        self.last_lap_time = safe_float(sensors.get("lastLapTime"))

    def write(self, reason):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        existing = []
        if os.path.exists(RUNS_PATH):
            with open(RUNS_PATH, newline="", encoding="utf-8") as source:
                existing = list(csv.DictReader(source))[-19:]
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": DRIVER_VERSION,
            "reason": reason,
            "steps": self.steps,
            "max_distance": self.max_distance,
            "max_speed": self.max_speed,
            "damage_final": self.damage,
            "offtrack_steps": self.offtrack,
            "recovery_steps": self.recovery,
            "mean_confidence": (
                statistics.mean(self.confidences) if self.confidences else 0.0
            ),
            "p95_inference_ms": percentile(self.latencies, 0.95),
            "last_lap_time": self.last_lap_time,
        }
        with open(RUNS_PATH, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(existing + [row])
        return row


def evaluate_leave_one_lap_out(dataset):
    report = []
    all_latencies = []
    for held in dataset.runs:
        train_rows = [
            row
            for run in dataset.runs
            if run["run_id"] != held["run_id"]
            for row in run["rows"]
        ]
        train_x = np.vstack([feature_vector(row) for row in train_rows])
        train_y = np.asarray(
            [(row["steer_action"], signed_pedal(row)) for row in train_rows]
        )
        model = SensorKNN(train_x, train_y)
        steer_error = 0.0
        pedal_error = 0.0
        fallback_steer_error = 0.0
        fallback_pedal_error = 0.0
        distances = []
        for row in held["rows"]:
            prediction = model.predict(feature_vector(row))
            fallback_steer, fallback_pedal = SafetyController.fallback_action(row)
            steer_error += abs(prediction["steer"] - row["steer_action"])
            pedal_error += abs(prediction["pedal"] - signed_pedal(row))
            fallback_steer_error += abs(fallback_steer - row["steer_action"])
            fallback_pedal_error += abs(fallback_pedal - signed_pedal(row))
            distances.append(prediction["distance"])
            all_latencies.append(prediction["inference_ms"])
        count = len(held["rows"])
        report.append(
            {
                "run_id": held["run_id"],
                "samples": count,
                "steer_mae": steer_error / count,
                "pedal_mae": pedal_error / count,
                "fallback_steer_mae": fallback_steer_error / count,
                "fallback_pedal_mae": fallback_pedal_error / count,
                "neighbor_distance_p95": percentile(distances, 0.95),
                "inference_p95_ms": percentile(all_latencies, 0.95),
            }
        )
    totals = sum(row["samples"] for row in report)
    aggregate = {
        "run_id": "ALL",
        "samples": totals,
        "steer_mae": sum(
            row["steer_mae"] * row["samples"] for row in report
        ) / totals,
        "pedal_mae": sum(
            row["pedal_mae"] * row["samples"] for row in report
        ) / totals,
        "fallback_steer_mae": sum(
            row["fallback_steer_mae"] * row["samples"] for row in report
        ) / totals,
        "fallback_pedal_mae": sum(
            row["fallback_pedal_mae"] * row["samples"] for row in report
        ) / totals,
        "neighbor_distance_p95": max(
            row["neighbor_distance_p95"] for row in report
        ),
        "inference_p95_ms": percentile(all_latencies, 0.95),
    }
    report.append(aggregate)
    return report


def run_invariant_checks(dataset):
    model = SensorKNN(dataset.features, dataset.targets)
    controller = SafetyController()
    sample = dataset.rows[len(dataset.rows) // 2]
    prediction = model.predict(feature_vector(sample))
    action, _ = controller.action(sample, prediction)
    assert len(feature_vector(sample)) == 25
    assert -1.0 <= action["steer"] <= 1.0
    assert 0.0 <= action["accel"] <= 1.0
    assert 0.0 <= action["brake"] <= 1.0
    assert not (action["accel"] > 0.01 and action["brake"] > 0.01)
    assert action["gear"] in (-1, 1, 2, 3, 4, 5, 6)
    assert SensorKNN.confidence(CONFIDENCE_FULL_DISTANCE) == 1.0
    assert SensorKNN.confidence(CONFIDENCE_ZERO_DISTANCE) == 0.0
    assert abs(controller.previous_steer) <= STEER_RATE_LIMIT + 1e-12
    assert abs(controller.previous_pedal) <= PEDAL_RATE_LIMIT + 1e-12

    edge = SafetyController()
    edge.previous_track_pos = -0.45
    edge.track_pos_rate = -0.025
    edge_result = edge._edge_guard(
        {"trackPos": -0.48, "speedX": 124.0},
        0.58,
    )
    assert edge_result["active"]
    assert edge_result["target"] < 0.0

    quiet_edge = SafetyController()
    quiet_edge.previous_track_pos = -0.45
    quiet_edge.track_pos_rate = -0.025
    quiet_result = quiet_edge._edge_guard(
        {"trackPos": -0.48, "speedX": 124.0},
        0.30,
    )
    assert not quiet_result["active"]

    # Worst case: the learned pedal keeps requesting full throttle.
    # The predictive guard must still be braking by the first demonstrated
    # braking point after the start.
    for run in dataset.runs:
        guard = SafetyController()
        first_brake_checked = False
        guard_seen = False
        maximum_guard_brake = 0.0
        release_seen = False
        for row in run["rows"]:
            if not 200.0 < row["distFromStart"] < 500.0:
                continue
            prediction = {
                "steer": row["steer_action"],
                "pedal": (
                    -1.0 if guard_seen else 1.0
                ),
                "distance": 0.0,
                "confidence": 1.0,
                "inference_ms": 0.0,
            }
            action, diagnostics = guard.action(row, prediction)
            guard_seen = guard_seen or bool(diagnostics["brake_guard"])
            maximum_guard_brake = max(maximum_guard_brake, action["brake"])
            release_seen = release_seen or (
                "brake_release" in diagnostics["interventions"]
                and diagnostics["requested_pedal"] == 0.0
            )
            if row["brake_action"] > 0.10:
                assert guard_seen
                assert maximum_guard_brake > 0.10
                first_brake_checked = True
            if first_brake_checked and release_seen:
                break
        assert first_brake_checked
        assert release_seen


def analyze(dataset):
    run_invariant_checks(dataset)
    report = evaluate_leave_one_lap_out(dataset)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(ANALYSIS_PATH, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    return report[-1]


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="V6 ibrida sensoriale con behavioral cloning KNN.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Valida dataset, KNN, invarianti e latenza senza TORCS.",
    )
    return parser.parse_known_args(argv)


def create_torcs_client(snakeoil_arguments):
    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]] + list(snakeoil_arguments)
        return snakeoil3.Client(p=PORT)
    finally:
        sys.argv = original_argv


def main():
    arguments, snakeoil_arguments = parse_arguments()
    dataset = DemonstrationDataset()
    print(
        "[DATASET] %d giri, %d campioni, %d feature sensoriali."
        % (len(dataset.runs), len(dataset.rows), dataset.features.shape[1])
    )
    if arguments.analyze_only:
        result = analyze(dataset)
        print("[REPORT] %s" % ANALYSIS_PATH)
        print(
            "[LOO] steer MAE %.6f; pedal MAE %.6f; "
            "fallback %.6f/%.6f; p95 %.3f ms"
            % (
                result["steer_mae"],
                result["pedal_mae"],
                result["fallback_steer_mae"],
                result["fallback_pedal_mae"],
                result["inference_p95_ms"],
            )
        )
        if result["inference_p95_ms"] >= 5.0:
            raise RuntimeError("Inferenza KNN oltre il gate di 5 ms.")
        return

    model = SensorKNN(dataset.features, dataset.targets)
    safety = SafetyController()
    trace = TraceLogger()
    summary = RunSummary()
    client = create_torcs_client(snakeoil_arguments)
    reason = "max_steps"
    try:
        for step in range(MAX_STEPS):
            client.get_servers_input()
            if not client.so:
                reason = "server_closed"
                break
            sensors = client.S.d
            if safe_float(sensors.get("lastLapTime")) > 0.0:
                reason = "lap_complete"
                break
            prediction = model.predict(feature_vector(sensors))
            action, diagnostics = safety.action(sensors, prediction)
            client.R.d.update(action)
            trace.write(step, sensors, action, diagnostics)
            summary.record(sensors, diagnostics)
            client.respond_to_server()
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    finally:
        trace.close()
        result = summary.write(reason)
        client.shutdown()
        print("[STOP] %s" % reason)
        print("[TRACE] %s" % TRACE_PATH)
        print(
            "[RUN] distance=%.1f speed=%.1f damage=%.1f "
            "offtrack=%d recovery=%d"
            % (
                result["max_distance"],
                result["max_speed"],
                result["damage_final"],
                result["offtrack_steps"],
                result["recovery_steps"],
            )
        )


if __name__ == "__main__":
    main()
