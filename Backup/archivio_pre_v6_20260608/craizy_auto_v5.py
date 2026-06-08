import argparse
import bisect
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict

import snakeoil3_jm2 as snakeoil3


PORT = 3001
MAX_STEPS = 100000
DRIVER_VERSION = "craizy_auto_v5_ibm_knn_residual"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "torcs_ps4_dataset.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RUNS_PATH = os.path.join(BASE_DIR, "auto_v5_runs.csv")

KNN_NEIGHBORS = 7
KNN_FULL_CONFIDENCE_DISTANCE = 0.10
KNN_ZERO_CONFIDENCE_DISTANCE = 0.70
TRACE_EVERY = 5
REFERENCE_STEP_METERS = 5.0

STEER_RATE_LIMIT = 0.04
STEER_FEEDBACK_POSITION_GAIN = 0.30
STEER_FEEDBACK_ANGLE_GAIN = 0.30
STEER_FEEDBACK_LATERAL_GAIN = 0.003
STEER_FEEDBACK_LIMIT_LOW_SPEED = 0.18
STEER_FEEDBACK_LIMIT_HIGH_SPEED = 0.14
LINE_GUARD_START = 0.15
LINE_GUARD_FULL = 0.55
ANGLE_GUARD_START = 0.08
ANGLE_GUARD_FULL = 0.30
LATERAL_GUARD_START = 4.0
LATERAL_GUARD_FULL = 16.0
MAX_PROFILE_STEER_BLEND = 0.70
GUARD_MAX_THROTTLE_CUT = 0.60
GUARD_BRAKE_MAX = 0.12

MIN_COMPLETE_LAP_ROWS = 800
MIN_COMPLETE_LAP_DISTANCE = 3500.0
MIN_CLEAN_LAPS = 3
MAX_TRACK_LENGTH_DEVIATION = 10.0
OFFTRACK_CONFIRM_TICKS = 3

# Parameters copied from the professors' IBM fastest.py F1 driver.
TARGET_SPEED = 160.0
STEER_GAIN = 30.0
CENTERING_GAIN = 0.2
BRAKE_THRESHOLD = 0.4
GEAR_SPEEDS = (0.0, 50.0, 80.0, 120.0, 150.0, 200.0)
SAFE_GENTLE_CORNER_SPEED = 140.0
SAFE_SHARP_CORNER_SPEED = 65.0
TARGET_STRAIGHT_SPEED = 194.0
CORNER_READING = 2.0
SLOW_DOWN_DISTANCE = 60.0
STRAIGHT_DISTANCE = 120.0
BRAKING_INTENSITY = 0.3
STEERING_EFFECT = 1.6

RECOVERY_TRACK_POS = 1.0
RECOVERY_EXTREME_TRACK_POS = 1.20
RECOVERY_EXTREME_ANGLE = 0.90
RECOVERY_CONFIRM_TICKS = 3
RECOVERY_EXIT_TICKS = 10
RECOVERY_STEER_RATE = 0.04
RECOVERY_MAX_STEER = 0.65
STUCK_SPEED = 5.0
STUCK_TICKS = 100
RECOVERY_UNSTABLE_ANGLE = 0.45
RECOVERY_UNSTABLE_SPEEDY = 10.0

REQUIRED_COLUMNS = (
    "run_id",
    "step",
    "curLapTime",
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
    "z",
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
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
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
    ordered = sorted(values)
    position = (len(ordered) - 1) * clamp(fraction, 0.0, 1.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_dataset_row(raw):
    row = {"run_id": str(raw.get("run_id", "")).strip()}
    for field in NUMERIC_FIELDS:
        row[field] = safe_float(raw.get(field))
    row["track"] = safe_list(raw.get("track"), 19, -1.0)
    row["wheelSpinVel"] = safe_list(raw.get("wheelSpinVel"), 4, 0.0)
    return row


def consecutive_offtrack_ticks(rows):
    current = 0
    maximum = 0
    for row in rows:
        offtrack = (
            abs(row["trackPos"]) >= 1.0
            or min(row["track"]) < 0.0
        )
        current = current + 1 if offtrack else 0
        maximum = max(maximum, current)
    return maximum


class DatasetError(RuntimeError):
    pass


class PostAdasDataset:
    def __init__(self, path=DATASET_PATH):
        self.path = path
        self.runs = []
        self.track_length = 0.0
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            raise DatasetError("Dataset non trovato: %s" % self.path)
        with open(self.path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            missing = [
                field for field in REQUIRED_COLUMNS
                if field not in (reader.fieldnames or [])
            ]
            if missing:
                raise DatasetError(
                    "Schema post-ADAS non valido; colonne mancanti: %s"
                    % ", ".join(missing)
                )
            groups = defaultdict(list)
            for raw in reader:
                row = parse_dataset_row(raw)
                if row["run_id"]:
                    groups[row["run_id"]].append(row)

        candidates = []
        rejected = []
        for run_id, rows in groups.items():
            prepared, reason = self._prepare_run(run_id, rows)
            if prepared is None:
                rejected.append("%s: %s" % (run_id, reason))
            else:
                candidates.append(prepared)

        if len(candidates) < MIN_CLEAN_LAPS:
            detail = "; ".join(rejected) if rejected else "nessun run_id"
            raise DatasetError(
                "Servono almeno %d giri puliti post-ADAS; trovati %d. %s"
                % (MIN_CLEAN_LAPS, len(candidates), detail)
            )

        median_length = statistics.median(
            run["length"] for run in candidates
        )
        self.runs = [
            run for run in candidates
            if abs(run["length"] - median_length)
            <= MAX_TRACK_LENGTH_DEVIATION
        ]
        if len(self.runs) < MIN_CLEAN_LAPS:
            raise DatasetError(
                "Meno di %d giri restano dopo il controllo lunghezza pista."
                % MIN_CLEAN_LAPS
            )
        self.track_length = statistics.median(
            run["length"] for run in self.runs
        )

    @staticmethod
    def _prepare_run(run_id, rows):
        rows = sorted(rows, key=lambda row: row["step"])
        if len(rows) < MIN_COMPLETE_LAP_ROWS:
            return None, "campioni insufficienti"
        if max(row["damage"] for row in rows) > 0.0:
            return None, "danno non nullo"
        if any(min(row["track"]) < 0.0 for row in rows):
            return None, "sensore pista negativo"
        if consecutive_offtrack_ticks(rows) >= OFFTRACK_CONFIRM_TICKS:
            return None, "uscita pista confermata"

        wrap_index = None
        for index in range(1, len(rows)):
            if (
                rows[index - 1]["distFromStart"] > 3000.0
                and rows[index]["distFromStart"] < 500.0
            ):
                wrap_index = index
                break
        if wrap_index is None:
            return None, "wrap launch/racing non trovato"

        launch_rows = rows[:wrap_index]
        racing_rows = rows[wrap_index:]
        racing_distance = max(
            row["distFromStart"] for row in racing_rows
        )
        length = max(row["distFromStart"] for row in rows)
        if (
            len(racing_rows) < MIN_COMPLETE_LAP_ROWS
            or racing_distance < MIN_COMPLETE_LAP_DISTANCE
        ):
            return None, "giro racing incompleto"

        launch_start = launch_rows[0]["curLapTime"]
        for row in launch_rows:
            row["_phase"] = "launch"
            row["_phase_time"] = max(
                0.0,
                row["curLapTime"] - launch_start,
            )
        for row in racing_rows:
            row["_phase"] = "racing"
            row["_phase_time"] = 0.0

        return {
            "run_id": run_id,
            "rows": rows,
            "launch_rows": launch_rows,
            "racing_rows": racing_rows,
            "length": length,
            "lap_time": rows[-1]["curLapTime"],
        }, ""


class IBMBasePolicy:
    """Corrected standalone port of the professors' fastest.py policy."""

    def __init__(self):
        self.previous_accel = 0.2

    def reset(self):
        self.previous_accel = 0.2

    @staticmethod
    def gear_for_speed(speed):
        gear = 1
        for index, threshold in enumerate(GEAR_SPEEDS):
            if speed > threshold:
                gear = index + 1
        return min(gear, 6)

    @staticmethod
    def is_corner(sensors, track):
        side_minimum = min(min(track[:9]), min(track[10:]))
        return (
            side_minimum < CORNER_READING
            or track[9] < sensors["speedX"] * 0.65
        )

    @staticmethod
    def corner_speed(track):
        return (
            SAFE_SHARP_CORNER_SPEED
            if max(track[8:11]) < SLOW_DOWN_DISTANCE
            else SAFE_GENTLE_CORNER_SPEED
        )

    def action(self, raw_sensors):
        sensors = {
            "speedX": safe_float(raw_sensors.get("speedX")),
            "speedY": safe_float(raw_sensors.get("speedY")),
            "trackPos": safe_float(raw_sensors.get("trackPos")),
            "angle": safe_float(raw_sensors.get("angle")),
        }
        track = safe_list(raw_sensors.get("track"), 19, 200.0)
        wheel_spin = safe_list(
            raw_sensors.get("wheelSpinVel"),
            4,
            0.0,
        )
        corner = self.is_corner(sensors, track)

        steer = (
            sensors["angle"] * STEER_GAIN / math.pi
            - sensors["trackPos"] * CENTERING_GAIN
        )
        if corner:
            # fastest.py divided nine sensor values by eight.
            left_average = sum(track[:9]) / len(track[:9])
            right_average = sum(track[10:]) / len(track[10:])
            bias = right_average - left_average
            if bias < 0.0:
                steer += 0.46
            elif bias > 0.0:
                steer -= 0.46
        steer = clamp(steer, -1.0, 1.0)

        target_speed = TARGET_SPEED
        if (
            sensors["speedX"] >= TARGET_SPEED - 5.0
            and track[9] > STRAIGHT_DISTANCE
        ):
            target_speed = TARGET_STRAIGHT_SPEED

        # abs(steer) removes the original left/right throttle asymmetry.
        if (
            sensors["speedX"]
            < target_speed - abs(steer) * STEERING_EFFECT
        ):
            accel = min(1.0, self.previous_accel + 0.4)
        else:
            accel = max(0.0, self.previous_accel - 0.2)

        if corner and sensors["speedX"] > self.corner_speed(track):
            accel = max(0.0, accel - 0.2)
        if sensors["speedX"] < 10.0:
            accel = 1.0

        brake = 0.0
        if abs(sensors["angle"]) > BRAKE_THRESHOLD:
            brake = BRAKING_INTENSITY
        if max(track[7:12]) < sensors["speedX"] * 0.60:
            brake += 0.1
        brake = clamp(brake, 0.0, 1.0)

        slip = (
            wheel_spin[2] + wheel_spin[3]
            - wheel_spin[0] - wheel_spin[1]
        )
        if slip > 2.0:
            accel -= 0.1
        accel = clamp(accel, 0.0, 1.0)
        if brake > 0.05:
            accel = 0.0

        self.previous_accel = accel
        return {
            "steer": steer,
            "accel": accel,
            "brake": brake,
            "gear": self.gear_for_speed(sensors["speedX"]),
            "clutch": 0.0,
            "meta": 0,
        }


def racing_features(sensors, base_action, track_length):
    distance = safe_float(sensors.get("distFromStart")) % track_length
    phase = 2.0 * math.pi * distance / track_length
    track = safe_list(sensors.get("track"), 19, 200.0)
    return [
        math.sin(phase),
        math.cos(phase),
        safe_float(sensors.get("trackPos")),
        safe_float(sensors.get("angle")) / math.pi,
        safe_float(sensors.get("speedX")) / 300.0,
        safe_float(sensors.get("speedY")) / 100.0,
        track[8] / 200.0,
        track[9] / 200.0,
        track[10] / 200.0,
        base_action["steer"],
        base_action["accel"],
        base_action["brake"],
    ]


def launch_features(sensors, base_action, elapsed):
    track = safe_list(sensors.get("track"), 19, 200.0)
    return [
        safe_float(elapsed) / 3.0,
        safe_float(sensors.get("trackPos")),
        safe_float(sensors.get("angle")) / math.pi,
        safe_float(sensors.get("speedX")) / 300.0,
        safe_float(sensors.get("speedY")) / 100.0,
        track[8] / 200.0,
        track[9] / 200.0,
        track[10] / 200.0,
        base_action["steer"],
        base_action["accel"],
        base_action["brake"],
    ]


def residual_target(row, base_action):
    return [
        row["steer_action"] - base_action["steer"],
        row["accel_action"] - base_action["accel"],
        row["brake_action"] - base_action["brake"],
    ]


class ResidualKNN:
    def __init__(self, phase, neighbours=KNN_NEIGHBORS):
        self.phase = phase
        self.requested_neighbours = neighbours
        self.neighbours = 0
        self.samples = 0
        self.model = None

    def fit(self, features, targets):
        if not features:
            raise DatasetError("Nessun campione KNN per la fase %s." % self.phase)
        try:
            from sklearn.neighbors import KNeighborsRegressor
        except ImportError as error:
            raise RuntimeError(
                "V5 richiede scikit-learn: "
                "python -m pip install -r requirements.txt"
            ) from error

        self.neighbours = min(
            max(1, int(self.requested_neighbours)),
            len(features),
        )
        self.model = KNeighborsRegressor(
            n_neighbors=self.neighbours,
            weights="distance",
            metric="euclidean",
            n_jobs=1,
        )
        self.model.fit(features, targets)
        self.samples = len(features)
        return self

    @property
    def available(self):
        return self.model is not None

    def predict_features(self, features):
        if not self.available:
            return [0.0, 0.0, 0.0], 0.0, float("inf"), float("inf")
        query = [features]
        distances, _ = self.model.kneighbors(
            query,
            n_neighbors=self.neighbours,
            return_distance=True,
        )
        values = [float(value) for value in distances[0]]
        nearest = min(values)
        mean_distance = statistics.fmean(values)
        if nearest <= 1e-12:
            confidence = 1.0
        else:
            confidence = 1.0 - clamp(
                (
                    mean_distance
                    - KNN_FULL_CONFIDENCE_DISTANCE
                )
                / (
                    KNN_ZERO_CONFIDENCE_DISTANCE
                    - KNN_FULL_CONFIDENCE_DISTANCE
                ),
                0.0,
                1.0,
            )
        prediction = self.model.predict(query)[0]
        return (
            [float(value) for value in prediction],
            confidence,
            nearest,
            mean_distance,
        )


class ResidualModelSet:
    def __init__(self, dataset, runs=None):
        self.dataset = dataset
        self.runs = list(runs if runs is not None else dataset.runs)
        self.launch = ResidualKNN("launch")
        self.racing = ResidualKNN("racing")
        self._fit()

    def _fit(self):
        features = {"launch": [], "racing": []}
        targets = {"launch": [], "racing": []}
        for run in self.runs:
            base_policy = IBMBasePolicy()
            for row in run["rows"]:
                base_action = base_policy.action(row)
                phase = row["_phase"]
                if phase == "launch":
                    sample = launch_features(
                        row,
                        base_action,
                        row["_phase_time"],
                    )
                else:
                    sample = racing_features(
                        row,
                        base_action,
                        self.dataset.track_length,
                    )
                features[phase].append(sample)
                targets[phase].append(residual_target(row, base_action))
        self.launch.fit(features["launch"], targets["launch"])
        self.racing.fit(features["racing"], targets["racing"])

    def predict(self, phase, sensors, base_action, elapsed=0.0):
        if phase == "launch":
            features = launch_features(sensors, base_action, elapsed)
            model = self.launch
        else:
            features = racing_features(
                sensors,
                base_action,
                self.dataset.track_length,
            )
            model = self.racing
        return model.predict_features(features)


class DemonstrationProfile:
    """Robust spatial reference built from all clean racing laps."""

    FIELDS = (
        "trackPos",
        "angle",
        "speedY",
        "speedX",
        "steer_action",
    )

    def __init__(self, dataset):
        self.track_length = safe_float(
            getattr(dataset, "track_length", 0.0)
        )
        self.samples = []
        self.distances = []
        runs = list(getattr(dataset, "runs", []))
        if not runs or self.track_length <= 0.0:
            return
        lap_data = [
            (
                run["racing_rows"],
                [
                    row["distFromStart"]
                    for row in run["racing_rows"]
                ],
            )
            for run in runs
        ]
        count = int(
            math.ceil(self.track_length / REFERENCE_STEP_METERS)
        )
        for index in range(count):
            distance = index * REFERENCE_STEP_METERS
            if distance >= self.track_length:
                break
            per_lap = [
                self._sample_rows(rows, distances, distance)
                for rows, distances in lap_data
            ]
            sample = {"distance": distance}
            for field in self.FIELDS:
                sample[field] = statistics.median(
                    row[field] for row in per_lap
                )
            self.samples.append(sample)
            self.distances.append(distance)

    @staticmethod
    def _sample_rows(rows, distances, distance):
        upper = bisect.bisect_left(distances, distance)
        if upper <= 0:
            return {field: rows[0][field] for field in DemonstrationProfile.FIELDS}
        if upper >= len(rows):
            return {field: rows[-1][field] for field in DemonstrationProfile.FIELDS}
        lower_row = rows[upper - 1]
        upper_row = rows[upper]
        span = upper_row["distFromStart"] - lower_row["distFromStart"]
        fraction = (
            0.0
            if span <= 1e-9
            else (distance - lower_row["distFromStart"]) / span
        )
        return {
            field: (
                lower_row[field]
                + (upper_row[field] - lower_row[field]) * fraction
            )
            for field in DemonstrationProfile.FIELDS
        }

    @property
    def available(self):
        return bool(self.samples)

    def reference_at(self, distance):
        if not self.available:
            return None
        distance = clamp(
            safe_float(distance),
            self.distances[0],
            self.distances[-1],
        )
        upper = bisect.bisect_left(self.distances, distance)
        if upper <= 0:
            return dict(self.samples[0])
        if upper >= len(self.samples):
            return dict(self.samples[-1])
        lower = self.samples[upper - 1]
        higher = self.samples[upper]
        span = higher["distance"] - lower["distance"]
        fraction = (
            0.0
            if span <= 1e-9
            else (distance - lower["distance"]) / span
        )
        result = {"distance": distance}
        for field in self.FIELDS:
            result[field] = (
                lower[field]
                + (higher[field] - lower[field]) * fraction
            )
        return result


class V5Controller:
    def __init__(self, dataset, models, base_only=False):
        self.dataset = dataset
        self.models = models
        self.base_only = base_only
        self.base_policy = IBMBasePolicy()
        self.reference_profile = DemonstrationProfile(dataset)
        self.phase = None
        self.previous_distance = None
        self.launch_start_time = None
        self.recovery_active = False
        self.recovery_ticks = 0
        self.recovery_exit_ticks = 0
        self.stuck_ticks = 0
        self.previous_recovery_steer = 0.0
        self.previous_final_steer = None

    @staticmethod
    def _guard_score(line_error, angle_error, lateral_error):
        line_score = clamp(
            (abs(line_error) - LINE_GUARD_START)
            / (LINE_GUARD_FULL - LINE_GUARD_START),
            0.0,
            1.0,
        )
        angle_score = clamp(
            (abs(angle_error) - ANGLE_GUARD_START)
            / (ANGLE_GUARD_FULL - ANGLE_GUARD_START),
            0.0,
            1.0,
        )
        lateral_score = clamp(
            (abs(lateral_error) - LATERAL_GUARD_START)
            / (LATERAL_GUARD_FULL - LATERAL_GUARD_START),
            0.0,
            1.0,
        )
        return max(line_score, angle_score, lateral_score)

    @staticmethod
    def _feedback_limit(speed):
        return (
            STEER_FEEDBACK_LIMIT_HIGH_SPEED
            if abs(speed) >= 160.0
            else STEER_FEEDBACK_LIMIT_LOW_SPEED
        )

    def _limit_steer_rate(self, desired):
        desired = clamp(desired, -1.0, 1.0)
        if self.previous_final_steer is None:
            self.previous_final_steer = desired
            return desired, 0.0
        delta = clamp(
            desired - self.previous_final_steer,
            -STEER_RATE_LIMIT,
            STEER_RATE_LIMIT,
        )
        limited = clamp(
            self.previous_final_steer + delta,
            -1.0,
            1.0,
        )
        self.previous_final_steer = limited
        return limited, delta

    def _update_phase(self, sensors):
        distance = safe_float(sensors.get("distFromStart"))
        speed = safe_float(sensors.get("speedX"))
        current_time = safe_float(sensors.get("curLapTime"))
        transitioned = False
        if self.phase is None:
            self.phase = (
                "launch"
                if distance > 3000.0 and abs(speed) < 30.0
                else "racing"
            )
            self.launch_start_time = current_time
        if (
            self.phase == "launch"
            and self.previous_distance is not None
            and self.previous_distance > 3000.0
            and distance < 500.0
        ):
            self.phase = "racing"
            transitioned = True
        self.previous_distance = distance
        elapsed = (
            max(0.0, current_time - self.launch_start_time)
            if self.phase == "launch"
            else 0.0
        )
        return elapsed, transitioned

    @staticmethod
    def _recovery_target(angle, track_pos):
        return clamp(
            angle * 1.25 - track_pos * 0.70,
            -RECOVERY_MAX_STEER,
            RECOVERY_MAX_STEER,
        )

    def _recovery_action(self, sensors, stuck):
        speed = safe_float(sensors.get("speedX"))
        speed_y = safe_float(sensors.get("speedY"))
        angle = safe_float(sensors.get("angle"))
        track_pos = safe_float(sensors.get("trackPos"))
        target = self._recovery_target(
            angle,
            track_pos,
        )
        delta = clamp(
            target - self.previous_recovery_steer,
            -RECOVERY_STEER_RATE,
            RECOVERY_STEER_RATE,
        )
        steer = clamp(
            self.previous_recovery_steer + delta,
            -RECOVERY_MAX_STEER,
            RECOVERY_MAX_STEER,
        )
        self.previous_recovery_steer = steer
        if stuck:
            accel, brake, gear = 0.45, 0.0, -1
        elif (
            abs(angle) > RECOVERY_UNSTABLE_ANGLE
            or abs(speed_y) > RECOVERY_UNSTABLE_SPEEDY
        ):
            accel = 0.0
            brake = 0.12 if abs(speed) > 40.0 else 0.05
            gear = IBMBasePolicy.gear_for_speed(speed)
        elif speed > 140.0:
            accel, brake = 0.0, 0.0
            gear = IBMBasePolicy.gear_for_speed(speed)
        elif speed > 80.0:
            accel, brake = 0.0, 0.10
            gear = IBMBasePolicy.gear_for_speed(speed)
        else:
            accel, brake = 0.35, 0.0
            gear = IBMBasePolicy.gear_for_speed(speed)
        return {
            "steer": steer,
            "accel": accel,
            "brake": brake,
            "gear": gear,
            "clutch": 0.0,
            "meta": 0,
        }, target

    def action(self, sensors):
        elapsed, transitioned = self._update_phase(sensors)
        base_action = self.base_policy.action(sensors)
        residual = [0.0, 0.0, 0.0]
        confidence = 0.0
        nearest = float("inf")
        mean_distance = float("inf")
        if not self.base_only and self.models is not None:
            residual, confidence, nearest, mean_distance = (
                self.models.predict(
                    self.phase,
                    sensors,
                    base_action,
                    elapsed,
                )
            )

        learned_action = {
            "steer": clamp(
                base_action["steer"] + residual[0] * confidence,
                -1.0,
                1.0,
            ),
            "accel": clamp(
                base_action["accel"] + residual[1] * confidence,
                0.0,
                1.0,
            ),
            "brake": clamp(
                base_action["brake"] + residual[2] * confidence,
                0.0,
                1.0,
            ),
            # KNN never predicts or modifies gear.
            "gear": base_action["gear"],
            "clutch": 0.0,
            "meta": 0,
        }
        if learned_action["brake"] > 0.05:
            learned_action["accel"] = 0.0

        track = safe_list(sensors.get("track"), 19, 200.0)
        track_pos = safe_float(sensors.get("trackPos"))
        angle = safe_float(sensors.get("angle"))
        speed = safe_float(sensors.get("speedX"))
        speed_y = safe_float(sensors.get("speedY"))
        reference = None
        line_error = 0.0
        angle_error = 0.0
        lateral_error = 0.0
        guard_score = 0.0
        profile_blend = 0.0
        steering_feedback = 0.0
        desired_steer = learned_action["steer"]
        steer_rate_delta = 0.0

        if (
            not self.base_only
            and self.phase == "racing"
            and self.reference_profile.available
        ):
            reference = self.reference_profile.reference_at(
                sensors.get("distFromStart")
            )
            line_error = track_pos - reference["trackPos"]
            angle_error = angle - reference["angle"]
            lateral_error = speed_y - reference["speedY"]
            guard_score = self._guard_score(
                line_error,
                angle_error,
                lateral_error,
            )
            profile_blend = MAX_PROFILE_STEER_BLEND * guard_score
            profile_guided_steer = (
                learned_action["steer"] * (1.0 - profile_blend)
                + reference["steer_action"] * profile_blend
            )
            steering_feedback = clamp(
                (
                    angle_error * STEER_FEEDBACK_ANGLE_GAIN
                    - line_error * STEER_FEEDBACK_POSITION_GAIN
                    - lateral_error * STEER_FEEDBACK_LATERAL_GAIN
                ),
                -self._feedback_limit(speed),
                self._feedback_limit(speed),
            )
            desired_steer = profile_guided_steer + steering_feedback
            learned_action["accel"] *= (
                1.0 - GUARD_MAX_THROTTLE_CUT * guard_score
            )
            overspeed = speed - reference["speedX"]
            if guard_score > 0.50 and overspeed > 5.0:
                learned_action["brake"] = max(
                    learned_action["brake"],
                    GUARD_BRAKE_MAX
                    * guard_score
                    * clamp(overspeed / 20.0, 0.40, 1.0),
                )
            if learned_action["brake"] > 0.05:
                learned_action["accel"] = 0.0

        learned_action["steer"], steer_rate_delta = (
            self._limit_steer_rate(desired_steer)
        )
        offtrack = (
            abs(track_pos) > RECOVERY_TRACK_POS
            or min(track) < 0.0
        )
        if self.phase == "racing" and offtrack:
            self.recovery_ticks += 1
        else:
            self.recovery_ticks = 0
        if self.phase == "racing" and abs(speed) < STUCK_SPEED:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0

        stuck = self.stuck_ticks >= STUCK_TICKS
        extreme = (
            abs(track_pos) > RECOVERY_EXTREME_TRACK_POS
            or abs(angle) > RECOVERY_EXTREME_ANGLE
        )
        recovery_cause = ""
        if stuck:
            recovery_cause = "stuck"
        elif extreme:
            recovery_cause = "extreme"
        elif self.recovery_ticks >= RECOVERY_CONFIRM_TICKS:
            recovery_cause = "offtrack"
        if recovery_cause:
            self.recovery_active = True
            self.recovery_exit_ticks = 0
        elif self.recovery_active:
            stable = (
                abs(track_pos) < 0.90
                and abs(angle) < 0.35
                and min(track) >= 0.0
            )
            self.recovery_exit_ticks = (
                self.recovery_exit_ticks + 1 if stable else 0
            )
            if self.recovery_exit_ticks >= RECOVERY_EXIT_TICKS:
                self.recovery_active = False
                self.recovery_exit_ticks = 0

        recovery_action, recovery_target = self._recovery_action(
            sensors,
            stuck,
        )
        if self.recovery_active:
            final_action = recovery_action
            mode = "recovery"
            self.previous_final_steer = final_action["steer"]
        else:
            final_action = learned_action
            mode = (
                "base"
                if self.base_only or confidence <= 0.0
                else "guard"
                if guard_score > 0.0
                else "knn"
            )
            self.previous_recovery_steer = final_action["steer"]

        diagnostics = {
            "phase": self.phase,
            "phase_transition": int(transitioned),
            "phase_time": elapsed,
            "mode": mode,
            "confidence": confidence,
            "nearest_distance": nearest,
            "mean_distance": mean_distance,
            "reference": reference,
            "line_error": line_error,
            "angle_error": angle_error,
            "lateral_error": lateral_error,
            "guard_score": guard_score,
            "profile_blend": profile_blend,
            "steering_feedback": steering_feedback,
            "desired_steer": clamp(desired_steer, -1.0, 1.0),
            "steer_rate_delta": steer_rate_delta,
            "residual_steer": residual[0],
            "residual_accel": residual[1],
            "residual_brake": residual[2],
            "base_action": base_action,
            "learned_action": learned_action,
            "recovery_active": int(self.recovery_active),
            "recovery_cause": recovery_cause,
            "recovery_ticks": self.recovery_ticks,
            "stuck_ticks": self.stuck_ticks,
            "recovery_target": recovery_target,
        }
        return final_action, diagnostics


class TraceLogger:
    FIELDS = [
        "step",
        "phase",
        "phase_transition",
        "phase_time",
        "mode",
        "curLapTime",
        "distFromStart",
        "speedX",
        "speedY",
        "trackPos",
        "angle",
        "front_track",
        "base_steer",
        "base_accel",
        "base_brake",
        "base_gear",
        "residual_steer",
        "residual_accel",
        "residual_brake",
        "knn_confidence",
        "knn_nearest_distance",
        "knn_mean_distance",
        "reference_trackPos",
        "reference_angle",
        "reference_speedY",
        "reference_speedX",
        "reference_steer",
        "line_error",
        "angle_error",
        "lateral_error",
        "guard_score",
        "profile_blend",
        "steering_feedback",
        "desired_steer",
        "steer_rate_delta",
        "learned_steer",
        "learned_accel",
        "learned_brake",
        "final_steer",
        "final_accel",
        "final_brake",
        "final_gear",
        "recovery_active",
        "recovery_cause",
        "recovery_ticks",
        "stuck_ticks",
        "damage",
        "offtrack",
    ]

    def __init__(self, path=None):
        os.makedirs(LOGS_DIR, exist_ok=True)
        if path is None:
            stamp = "%s_%07d" % (
                time.strftime("%Y%m%d_%H%M%S"),
                time.time_ns() % 10000000,
            )
            path = os.path.join(
                LOGS_DIR,
                "auto_v5_trace_%s.csv" % stamp,
            )
        self.path = path
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()

    def write(self, step, sensors, action, diagnostics):
        if step % TRACE_EVERY:
            return
        base = diagnostics["base_action"]
        learned = diagnostics["learned_action"]
        reference = diagnostics["reference"] or {}
        track = safe_list(sensors.get("track"), 19, 200.0)
        self.writer.writerow({
            "step": step,
            "phase": diagnostics["phase"],
            "phase_transition": diagnostics["phase_transition"],
            "phase_time": diagnostics["phase_time"],
            "mode": diagnostics["mode"],
            "curLapTime": safe_float(sensors.get("curLapTime")),
            "distFromStart": safe_float(sensors.get("distFromStart")),
            "speedX": safe_float(sensors.get("speedX")),
            "speedY": safe_float(sensors.get("speedY")),
            "trackPos": safe_float(sensors.get("trackPos")),
            "angle": safe_float(sensors.get("angle")),
            "front_track": track[9],
            "base_steer": base["steer"],
            "base_accel": base["accel"],
            "base_brake": base["brake"],
            "base_gear": base["gear"],
            "residual_steer": diagnostics["residual_steer"],
            "residual_accel": diagnostics["residual_accel"],
            "residual_brake": diagnostics["residual_brake"],
            "knn_confidence": diagnostics["confidence"],
            "knn_nearest_distance": diagnostics["nearest_distance"],
            "knn_mean_distance": diagnostics["mean_distance"],
            "reference_trackPos": reference.get("trackPos", ""),
            "reference_angle": reference.get("angle", ""),
            "reference_speedY": reference.get("speedY", ""),
            "reference_speedX": reference.get("speedX", ""),
            "reference_steer": reference.get("steer_action", ""),
            "line_error": diagnostics["line_error"],
            "angle_error": diagnostics["angle_error"],
            "lateral_error": diagnostics["lateral_error"],
            "guard_score": diagnostics["guard_score"],
            "profile_blend": diagnostics["profile_blend"],
            "steering_feedback": diagnostics["steering_feedback"],
            "desired_steer": diagnostics["desired_steer"],
            "steer_rate_delta": diagnostics["steer_rate_delta"],
            "learned_steer": learned["steer"],
            "learned_accel": learned["accel"],
            "learned_brake": learned["brake"],
            "final_steer": action["steer"],
            "final_accel": action["accel"],
            "final_brake": action["brake"],
            "final_gear": action["gear"],
            "recovery_active": diagnostics["recovery_active"],
            "recovery_cause": diagnostics["recovery_cause"],
            "recovery_ticks": diagnostics["recovery_ticks"],
            "stuck_ticks": diagnostics["stuck_ticks"],
            "damage": safe_float(sensors.get("damage")),
            "offtrack": int(
                abs(safe_float(sensors.get("trackPos"))) > 1.0
                or min(track) < 0.0
            ),
        })

    def close(self):
        self.file.close()


def mean_absolute_error(total, count):
    return total / count if count else 0.0


def evaluate_run(dataset, train_runs, held_run):
    models = ResidualModelSet(dataset, train_runs)
    base_policy = IBMBasePolicy()
    totals = {
        "base_steer": 0.0,
        "base_accel": 0.0,
        "base_brake": 0.0,
        "knn_steer": 0.0,
        "knn_accel": 0.0,
        "knn_brake": 0.0,
    }
    confidences = []
    durations = []
    count = 0
    for row in held_run["rows"]:
        base = base_policy.action(row)
        start = time.perf_counter()
        residual, confidence, _, _ = models.predict(
            row["_phase"],
            row,
            base,
            row["_phase_time"],
        )
        durations.append((time.perf_counter() - start) * 1000.0)
        predicted = (
            clamp(base["steer"] + residual[0] * confidence, -1.0, 1.0),
            clamp(base["accel"] + residual[1] * confidence, 0.0, 1.0),
            clamp(base["brake"] + residual[2] * confidence, 0.0, 1.0),
        )
        expected = (
            row["steer_action"],
            row["accel_action"],
            row["brake_action"],
        )
        for index, name in enumerate(("steer", "accel", "brake")):
            totals["base_" + name] += abs(base[name] - expected[index])
            totals["knn_" + name] += abs(predicted[index] - expected[index])
        confidences.append(confidence)
        count += 1
    return {
        "held_run_id": held_run["run_id"],
        "samples": count,
        "base_steer_mae": mean_absolute_error(totals["base_steer"], count),
        "base_accel_mae": mean_absolute_error(totals["base_accel"], count),
        "base_brake_mae": mean_absolute_error(totals["base_brake"], count),
        "knn_steer_mae": mean_absolute_error(totals["knn_steer"], count),
        "knn_accel_mae": mean_absolute_error(totals["knn_accel"], count),
        "knn_brake_mae": mean_absolute_error(totals["knn_brake"], count),
        "avg_confidence": statistics.fmean(confidences),
        "inference_p95_ms": percentile(durations, 0.95),
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as output:
        if not rows:
            return
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze(dataset):
    rows = []
    for held_run in dataset.runs:
        train_runs = [
            run for run in dataset.runs
            if run["run_id"] != held_run["run_id"]
        ]
        rows.append(evaluate_run(dataset, train_runs, held_run))
    total_samples = sum(row["samples"] for row in rows)
    aggregate = {"held_run_id": "ALL", "samples": total_samples}
    metric_fields = [
        "base_steer_mae",
        "base_accel_mae",
        "base_brake_mae",
        "knn_steer_mae",
        "knn_accel_mae",
        "knn_brake_mae",
        "avg_confidence",
        "inference_p95_ms",
    ]
    for field in metric_fields:
        aggregate[field] = sum(
            row[field] * row["samples"] for row in rows
        ) / total_samples
    rows.append(aggregate)
    loo_path = os.path.join(RESULTS_DIR, "corkscrew_v5_loo.csv")
    write_csv(loo_path, rows)

    models = ResidualModelSet(dataset)
    summary = [{
        "driver_version": DRIVER_VERSION,
        "dataset_runs": len(dataset.runs),
        "dataset_rows": sum(len(run["rows"]) for run in dataset.runs),
        "track_length": dataset.track_length,
        "knn_neighbors": KNN_NEIGHBORS,
        "launch_samples": models.launch.samples,
        "racing_samples": models.racing.samples,
        "full_confidence_distance": KNN_FULL_CONFIDENCE_DISTANCE,
        "zero_confidence_distance": KNN_ZERO_CONFIDENCE_DISTANCE,
        "loo_base_steer_mae": aggregate["base_steer_mae"],
        "loo_knn_steer_mae": aggregate["knn_steer_mae"],
        "loo_base_accel_mae": aggregate["base_accel_mae"],
        "loo_knn_accel_mae": aggregate["knn_accel_mae"],
        "loo_base_brake_mae": aggregate["base_brake_mae"],
        "loo_knn_brake_mae": aggregate["knn_brake_mae"],
        "loo_inference_p95_ms": aggregate["inference_p95_ms"],
    }]
    summary_path = os.path.join(
        RESULTS_DIR,
        "corkscrew_v5_model_summary.csv",
    )
    write_csv(summary_path, summary)
    return loo_path, summary_path, rows, models


class RunSummary:
    FIELDS = [
        "timestamp",
        "driver_version",
        "mode",
        "steps",
        "last_lap_time",
        "damage",
        "offtrack_steps",
        "recovery_steps",
        "avg_speed",
        "max_speed",
        "avg_knn_confidence",
        "reason",
    ]

    def __init__(self, base_only):
        self.base_only = base_only
        self.steps = 0
        self.speed_sum = 0.0
        self.max_speed = 0.0
        self.confidence_sum = 0.0
        self.offtrack_steps = 0
        self.recovery_steps = 0
        self.final_sensors = {}

    def record(self, sensors, diagnostics):
        speed = safe_float(sensors.get("speedX"))
        track = safe_list(sensors.get("track"), 19, 200.0)
        self.steps += 1
        self.speed_sum += speed
        self.max_speed = max(self.max_speed, speed)
        self.confidence_sum += diagnostics["confidence"]
        self.offtrack_steps += int(
            abs(safe_float(sensors.get("trackPos"))) > 1.0
            or min(track) < 0.0
        )
        self.recovery_steps += diagnostics["recovery_active"]
        self.final_sensors = dict(sensors)

    def write(self, reason):
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "driver_version": DRIVER_VERSION,
            "mode": "base" if self.base_only else "knn",
            "steps": self.steps,
            "last_lap_time": safe_float(
                self.final_sensors.get("lastLapTime")
            ),
            "damage": safe_float(self.final_sensors.get("damage")),
            "offtrack_steps": self.offtrack_steps,
            "recovery_steps": self.recovery_steps,
            "avg_speed": (
                self.speed_sum / self.steps if self.steps else 0.0
            ),
            "max_speed": self.max_speed,
            "avg_knn_confidence": (
                self.confidence_sum / self.steps if self.steps else 0.0
            ),
            "reason": reason,
        }
        exists = os.path.exists(RUNS_PATH) and os.path.getsize(RUNS_PATH) > 0
        with open(RUNS_PATH, "a", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=self.FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="V5 IBM base con KNN regressivo residuale per Corkscrew.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Usa soltanto il pilota IBM, senza applicare residui KNN.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Genera leave-one-lap-out e benchmark senza collegarsi a TORCS.",
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
    dataset = PostAdasDataset()
    print(
        "[DATASET] %d giri, %d campioni, pista %.2f m."
        % (
            len(dataset.runs),
            sum(len(run["rows"]) for run in dataset.runs),
            dataset.track_length,
        )
    )
    if arguments.analyze_only:
        loo_path, summary_path, rows, _ = analyze(dataset)
        aggregate = rows[-1]
        print("[REPORT] %s" % loo_path)
        print("[REPORT] %s" % summary_path)
        print(
            "[LOO] steer %.6f -> %.6f; accel %.6f -> %.6f; "
            "brake %.6f -> %.6f; p95 %.3f ms"
            % (
                aggregate["base_steer_mae"],
                aggregate["knn_steer_mae"],
                aggregate["base_accel_mae"],
                aggregate["knn_accel_mae"],
                aggregate["base_brake_mae"],
                aggregate["knn_brake_mae"],
                aggregate["inference_p95_ms"],
            )
        )
        return

    models = (
        None
        if arguments.base_only
        else ResidualModelSet(dataset)
    )
    controller = V5Controller(
        dataset,
        models,
        base_only=arguments.base_only,
    )
    trace = TraceLogger()
    summary = RunSummary(arguments.base_only)
    client = create_torcs_client(snakeoil_arguments)
    reason = "max_steps"
    print(
        "[DRIVER] %s mode=%s launch=%d racing=%d"
        % (
            DRIVER_VERSION,
            "base" if arguments.base_only else "knn",
            models.launch.samples if models is not None else 0,
            models.racing.samples if models is not None else 0,
        )
    )
    try:
        for step in range(MAX_STEPS):
            client.get_servers_input()
            if not client.so:
                reason = "server_closed"
                break
            sensors = client.S.d
            if safe_float(sensors.get("lastLapTime")) > 0.0:
                summary.final_sensors = dict(sensors)
                reason = "lap_complete"
                break
            action, diagnostics = controller.action(sensors)
            client.R.d.update(action)
            summary.record(sensors, diagnostics)
            trace.write(step, sensors, action, diagnostics)
            client.respond_to_server()
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    finally:
        trace.close()
        summary.write(reason)
        client.shutdown()
        print("[STOP] %s" % reason)
        print("[TRACE] %s" % trace.path)


if __name__ == "__main__":
    main()
