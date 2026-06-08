"""V7: behavioral cloning and DAgger driver for TORCS.

The normal driving policy is a PyTorch MLP trained on post-ADAS expert
actions. Deterministic code is limited to transmission, command clipping and
an emergency recovery state machine.
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import torch
from sklearn.neighbors import KNeighborsRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import snakeoil3_jm2 as snakeoil3

pygame = None


DRIVER_VERSION = "craizy_auto_v7_robust_behavioral_cloning_v2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "torcs_ps4_dataset.csv")
DAGGER_PATH = os.path.join(BASE_DIR, "torcs_v7_dagger.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "craizy_v7_bc.pt")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRACE_PATH = os.path.join(LOG_DIR, "auto_v7_latest.csv")
ANALYSIS_PATH = os.path.join(RESULTS_DIR, "auto_v7_analysis.csv")
CURRENT_ANALYSIS_PATH = os.path.join(
    RESULTS_DIR, "auto_v7_current_analysis.csv"
)

PORT = 3001
MAX_STEPS = 100000
FEATURE_COUNT = 25
DEFAULT_EPOCHS = 100
LOO_EPOCHS = 45
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 7

MIN_CLEAN_LAPS = 3
MIN_LAP_ROWS = 800
MIN_LAP_DISTANCE = 3500.0
OFFTRACK_CONFIRM_TICKS = 3

GEAR_UP_SPEEDS = (27.0, 54.0, 90.0, 120.0, 150.0)
GEAR_DOWN_SPEEDS = (22.0, 48.0, 80.0, 110.0, 140.0)

NORMAL = "NORMAL"
RETURN_FORWARD = "RETURN_FORWARD"
REVERSE = "REVERSE"
REJOIN = "REJOIN"

TAKEOVER_BUTTON = 9  # L1 on the DualShock 4 pygame mapping.
STEER_AXIS = 0
L2_AXIS = 4
R2_AXIS = 5
INVERT_STEERING = True
STEER_DEADZONE = 0.08
TRIGGER_DEADZONE = 0.08
STEER_PROGRESSION = 2.20
TRIGGER_PROGRESSION = 1.70

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

DATASET_COLUMNS = (
    "run_id", "step", "curLapTime",
    "steer_intent", "accel_intent", "brake_intent",
    "steer_action", "accel_action", "brake_action", "gear_action",
    "speedX", "speedY", "speedZ", "wheelSpinVel", "z", "track",
    "trackPos", "angle", "rpm", "damage", "distFromStart",
    "previous_steer", "previous_pedal",
)

REQUIRED_COLUMNS = (
    "run_id", "step", "steer_action", "accel_action", "brake_action",
    "speedX", "speedY", "wheelSpinVel", "track", "trackPos", "angle",
    "rpm", "damage", "distFromStart",
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
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction * 100))


def signed_pedal(row):
    return clamp(
        safe_float(row.get("accel_action"))
        - safe_float(row.get("brake_action")),
        -1.0,
        1.0,
    )


def wheel_slip_feature(speed_x, wheel_spin):
    wheel_speed = [
        abs(wheel_spin[index]) * WHEEL_RADII[index] for index in range(4)
    ]
    vehicle_speed = abs(speed_x) / 3.6
    driven_speed = statistics.mean(wheel_speed[2:4])
    return clamp((driven_speed - vehicle_speed) / 30.0, -1.0, 1.0)


def feature_vector(sensors):
    track = safe_list(sensors.get("track"), 19, -1.0)
    wheels = safe_list(sensors.get("wheelSpinVel"), 4, 0.0)
    speed_x = safe_float(sensors.get("speedX"))
    values = [clamp(value, 0.0, 200.0) / 200.0 for value in track]
    values.extend(
        (
            clamp(safe_float(sensors.get("trackPos")), -2.0, 2.0),
            clamp(safe_float(sensors.get("angle")) / math.pi, -1.0, 1.0),
            clamp(speed_x / 300.0, -0.2, 1.2),
            clamp(safe_float(sensors.get("speedY")) / 100.0, -1.0, 1.0),
            clamp(safe_float(sensors.get("rpm")) / 10000.0, 0.0, 1.5),
            wheel_slip_feature(speed_x, wheels),
        )
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (FEATURE_COUNT,):
        raise ValueError("Numero feature inatteso: %s" % (result.shape,))
    return result


def action_feature_vector(sensors, previous_steer, previous_pedal):
    return feature_vector(sensors)


def sequence_features(rows, use_stored_history=False):
    features = []
    previous_steer = 0.0
    previous_pedal = 0.0
    previous_step = None
    for row in rows:
        if use_stored_history:
            previous_steer = safe_float(row.get("previous_steer"))
            previous_pedal = safe_float(row.get("previous_pedal"))
        elif previous_step is not None and row["step"] != previous_step + 1:
            previous_steer = 0.0
            previous_pedal = 0.0
        features.append(feature_vector(row))
        previous_steer = clamp(row["steer_action"], -1.0, 1.0)
        previous_pedal = signed_pedal(row)
        previous_step = row["step"]
    return np.vstack(features)


def parse_row(raw):
    row = dict(raw)
    row["run_id"] = str(raw.get("run_id", "")).strip()
    row["track"] = safe_list(raw.get("track"), 19, -1.0)
    row["wheelSpinVel"] = safe_list(raw.get("wheelSpinVel"), 4, 0.0)
    for field in (
        "step", "curLapTime", "steer_intent", "accel_intent",
        "brake_intent", "steer_action", "accel_action", "brake_action",
        "gear_action", "speedX", "speedY", "speedZ", "z", "trackPos",
        "angle", "rpm", "damage", "distFromStart",
        "previous_steer", "previous_pedal",
    ):
        row[field] = safe_float(raw.get(field))
    return row


class DatasetError(RuntimeError):
    pass


class DemonstrationDataset:
    def __init__(self, include_dagger=True):
        self.runs = []
        self.rows = []
        self.features = None
        self.targets = None
        self.sample_weights = None
        self._load_file(DATASET_PATH, require_clean_laps=True)
        if include_dagger and os.path.exists(DAGGER_PATH):
            self._load_file(DAGGER_PATH, require_clean_laps=False)
        clean_runs = [run for run in self.runs if run["source"] == "expert"]
        if len(clean_runs) < MIN_CLEAN_LAPS:
            raise DatasetError(
                "Servono almeno %d giri puliti; trovati %d."
                % (MIN_CLEAN_LAPS, len(clean_runs))
            )
        feature_groups = []
        self.rows = []
        for run in self.runs:
            run["features"] = sequence_features(
                run["rows"],
                use_stored_history=run["source"] == "dagger",
            )
            feature_groups.append(run["features"])
            self.rows.extend(run["rows"])
        self.features = np.vstack(feature_groups)
        self.targets = np.asarray(
            [
                (
                    clamp(row["steer_action"], -1.0, 1.0),
                    signed_pedal(row),
                )
                for row in self.rows
            ],
            dtype=np.float32,
        )
        self.sample_weights = self._weights()

    def _load_file(self, path, require_clean_laps):
        with open(path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            missing = [
                field for field in REQUIRED_COLUMNS
                if field not in (reader.fieldnames or [])
            ]
            if missing:
                raise DatasetError(
                    "%s: colonne mancanti: %s"
                    % (os.path.basename(path), ", ".join(missing))
                )
            groups = defaultdict(list)
            for raw in reader:
                row = parse_row(raw)
                if row["run_id"]:
                    groups[row["run_id"]].append(row)
        source_name = "expert" if require_clean_laps else "dagger"
        for run_id, rows in groups.items():
            rows.sort(key=lambda item: item["step"])
            if require_clean_laps:
                reason = self._invalid_lap(rows)
                if reason:
                    continue
            else:
                rows = [
                    row for row in rows
                    if self._valid_dagger_row(row)
                ]
                if not rows:
                    continue
            self.runs.append(
                {"run_id": run_id, "rows": rows, "source": source_name}
            )

    @staticmethod
    def _invalid_lap(rows):
        if len(rows) < MIN_LAP_ROWS:
            return "campioni insufficienti"
        if max(row["distFromStart"] for row in rows) < MIN_LAP_DISTANCE:
            return "distanza insufficiente"
        if max(row["damage"] for row in rows) > 0.0:
            return "danno"
        bad_ticks = 0
        for row in rows:
            bad = abs(row["trackPos"]) >= 1.0 or min(row["track"]) < 0.0
            bad_ticks = bad_ticks + 1 if bad else 0
            if bad_ticks >= OFFTRACK_CONFIRM_TICKS:
                return "offtrack"
        return ""

    @staticmethod
    def _valid_dagger_row(row):
        intent_pedal = (
            safe_float(row.get("accel_intent"))
            - safe_float(row.get("brake_intent"))
        )
        action_pedal = (
            safe_float(row.get("accel_action"))
            - safe_float(row.get("brake_action"))
        )
        return (
            safe_float(row.get("damage")) <= 0.0
            and abs(safe_float(row.get("trackPos"))) < 1.0
            and min(safe_list(row.get("track"), 19, -1.0)) >= 0.0
            and safe_float(row.get("speedX")) >= 0.0
            and abs(intent_pedal - action_pedal) <= 0.30
        )

    def _weights(self):
        steer = np.abs(self.targets[:, 0])
        pedal = self.targets[:, 1]
        weights = np.ones(len(self.targets), dtype=np.float32)
        weights += 1.5 * np.clip(steer / 0.6, 0.0, 1.0)
        weights += 3.0 * (pedal < -0.05).astype(np.float32)
        weights += 1.0 * (pedal < -0.30).astype(np.float32)
        weights += 2.0 * (
            np.abs(self.features[:, 21]) < (5.0 / 300.0)
        ).astype(np.float32)
        return weights

    @property
    def expert_runs(self):
        return [run for run in self.runs if run["source"] == "expert"]


class BehavioralCloningMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(FEATURE_COUNT, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, features):
        return self.layers(features)


def set_random_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(features, targets, weights, epochs, quiet=False):
    set_random_seed()
    model = BehavioralCloningMLP()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    rng = np.random.default_rng(RANDOM_SEED)
    augmented_features = [features]
    augmented_targets = [targets]
    augmented_weights = [weights]
    for _ in range(2):
        noisy = features.copy()
        noisy[:, :19] += rng.normal(0.0, 0.010, noisy[:, :19].shape)
        noisy[:, 19] += rng.normal(0.0, 0.015, len(noisy))
        noisy[:, 20] += rng.normal(0.0, 0.010, len(noisy))
        noisy[:, 21:23] += rng.normal(
            0.0, 0.010, noisy[:, 21:23].shape
        )
        noisy[:, :19] = np.clip(noisy[:, :19], 0.0, 1.0)
        noisy[:, 19:21] = np.clip(noisy[:, 19:21], -2.0, 2.0)
        augmented_features.append(noisy)
        augmented_targets.append(targets)
        augmented_weights.append(weights)
    dataset = TensorDataset(
        torch.from_numpy(np.vstack(augmented_features)),
        torch.from_numpy(np.vstack(augmented_targets)),
        torch.from_numpy(np.concatenate(augmented_weights)),
    )
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y, batch_weight in loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            per_output = (prediction - batch_y) ** 2
            loss = (per_output.mean(dim=1) * batch_weight).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch_x)
        if not quiet and (epoch == 0 or (epoch + 1) % 20 == 0):
            print(
                "[TRAIN] epoch %d/%d loss=%.6f"
                % (epoch + 1, epochs, epoch_loss / len(dataset))
            )
    model.eval()
    return model


def predict_array(model, features):
    with torch.inference_mode():
        tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
        return model(tensor).cpu().numpy()


def metrics(prediction, target):
    error = np.abs(prediction - target)
    brake_mask = target[:, 1] < -0.05
    return {
        "steer_mae": float(error[:, 0].mean()),
        "pedal_mae": float(error[:, 1].mean()),
        "brake_mae": (
            float(error[brake_mask, 1].mean()) if brake_mask.any() else 0.0
        ),
        "brake_samples": int(brake_mask.sum()),
    }


def leave_one_lap_out(dataset):
    rows = []
    for held in dataset.expert_runs:
        train_runs = [
            run for run in dataset.runs if run["run_id"] != held["run_id"]
        ]
        train_rows = [row for run in train_runs for row in run["rows"]]
        train_x = np.vstack([run["features"] for run in train_runs])
        train_y = np.asarray(
            [(row["steer_action"], signed_pedal(row)) for row in train_rows],
            dtype=np.float32,
        )
        steer = np.abs(train_y[:, 0])
        train_w = (
            1.0
            + 1.5 * np.clip(steer / 0.6, 0.0, 1.0)
            + 3.0 * (train_y[:, 1] < -0.05)
            + 1.0 * (train_y[:, 1] < -0.30)
            + 2.0 * (np.abs(train_x[:, 21]) < (5.0 / 300.0))
        ).astype(np.float32)
        test_x = held["features"]
        test_y = np.asarray(
            [
                (row["steer_action"], signed_pedal(row))
                for row in held["rows"]
            ],
            dtype=np.float32,
        )
        model = train_model(
            train_x, train_y, train_w, LOO_EPOCHS, quiet=True
        )
        result = metrics(predict_array(model, test_x), test_y)
        result.update({"run_id": held["run_id"], "samples": len(test_y)})
        rows.append(result)
    total = sum(row["samples"] for row in rows)
    aggregate = {"run_id": "MLP_ALL", "samples": total}
    for key in ("steer_mae", "pedal_mae", "brake_mae"):
        aggregate[key] = sum(
            row[key] * row["samples"] for row in rows
        ) / total
    aggregate["brake_samples"] = sum(row["brake_samples"] for row in rows)
    rows.append(aggregate)
    return rows


def knn_comparison(dataset):
    predictions = []
    targets = []
    for held in dataset.expert_runs:
        train_rows = [
            row for run in dataset.runs
            if run["run_id"] != held["run_id"]
            for row in run["rows"]
        ]
        train_x = np.vstack([feature_vector(row) for row in train_rows])
        train_y = np.asarray(
            [(row["steer_action"], signed_pedal(row)) for row in train_rows],
            dtype=np.float32,
        )
        model = KNeighborsRegressor(
            n_neighbors=7, weights="distance", metric="euclidean", n_jobs=1
        )
        model.fit(train_x, train_y)
        test_x = np.vstack([feature_vector(row) for row in held["rows"]])
        predictions.append(model.predict(test_x))
        targets.append(
            np.asarray(
                [
                    (row["steer_action"], signed_pedal(row))
                    for row in held["rows"]
                ],
                dtype=np.float32,
            )
        )
    result = metrics(np.vstack(predictions), np.vstack(targets))
    result.update(
        {
            "run_id": "KNN_ALL",
            "samples": sum(len(values) for values in targets),
        }
    )
    return result


def save_model(model, dataset, epochs):
    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {
        "version": DRIVER_VERSION,
        "feature_count": FEATURE_COUNT,
        "architecture": [FEATURE_COUNT, 128, 64, 2],
        "state_dict": model.state_dict(),
        "expert_samples": sum(
            len(run["rows"]) for run in dataset.expert_runs
        ),
        "dagger_samples": sum(
            len(run["rows"]) for run in dataset.runs
            if run["source"] == "dagger"
        ),
        "epochs": epochs,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(payload, MODEL_PATH)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Modello V7 non trovato. Eseguire: python craizy_auto_v7.py --train"
        )
    payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if payload.get("feature_count") != FEATURE_COUNT:
        raise RuntimeError("Modello V7 incompatibile con le feature correnti.")
    model = BehavioralCloningMLP()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


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


class RecoveryController:
    def __init__(self):
        self.state = NORMAL
        self.state_ticks = 0
        self.bad_ticks = 0
        self.stuck_ticks = 0
        self.no_progress_ticks = 0
        self.best_track_pos = float("inf")
        self.reverse_attempted = False
        self.gears = GearController()

    def _set_state(self, state):
        if state != self.state:
            self.state = state
            self.state_ticks = 0

    def reset(self):
        self.state = NORMAL
        self.state_ticks = 0
        self.bad_ticks = 0
        self.stuck_ticks = 0
        self.no_progress_ticks = 0
        self.best_track_pos = float("inf")
        self.reverse_attempted = False

    def update_detection(self, sensors):
        track_pos = abs(safe_float(sensors.get("trackPos")))
        speed_x = abs(safe_float(sensors.get("speedX")))
        track = safe_list(sensors.get("track"), 19, -1.0)
        outside = track_pos >= 1.0 or min(track) < 0.0
        self.bad_ticks = self.bad_ticks + 1 if outside else 0
        self.stuck_ticks = self.stuck_ticks + 1 if speed_x < 2.0 else 0
        if self.state == NORMAL and (
            self.bad_ticks >= 3 or self.stuck_ticks >= 150
        ):
            self.best_track_pos = track_pos
            self.no_progress_ticks = 0
            self.reverse_attempted = False
            self._set_state(RETURN_FORWARD)

    def action(self, sensors):
        self.state_ticks += 1
        track_pos_signed = safe_float(sensors.get("trackPos"))
        track_pos = abs(track_pos_signed)
        angle = math.atan2(
            math.sin(safe_float(sensors.get("angle"))),
            math.cos(safe_float(sensors.get("angle"))),
        )
        speed_x = safe_float(sensors.get("speedX"))
        steer = clamp(-1.10 * angle + 0.75 * track_pos_signed, -0.65, 0.65)

        if track_pos < self.best_track_pos - 0.02:
            self.best_track_pos = track_pos
            self.no_progress_ticks = 0
        else:
            self.no_progress_ticks += 1

        if self.state == RETURN_FORWARD:
            target_speed = 30.0
            if speed_x > target_speed + 10.0:
                pedal = -0.12
            elif speed_x < target_speed:
                pedal = 0.25
            else:
                pedal = 0.0
            gear = self.gears.update(max(0.0, speed_x), sensors.get("gear"))
            if track_pos < 0.85 and abs(angle) < 0.30 and speed_x > 5.0:
                self._set_state(REJOIN)
            elif (
                self.no_progress_ticks > 100
                and speed_x < 6.0
                and not self.reverse_attempted
            ):
                self.reverse_attempted = True
                self._set_state(REVERSE)
            return steer, pedal, gear

        if self.state == REVERSE:
            if self.state_ticks >= 55:
                self.best_track_pos = track_pos
                self.no_progress_ticks = 0
                self._set_state(RETURN_FORWARD)
            return -steer, 0.28, -1

        if self.state == REJOIN:
            if self.state_ticks >= 30:
                self.state = NORMAL
                self.state_ticks = 0
                self.bad_ticks = 0
                self.stuck_ticks = 0
            return steer, 0.20, self.gears.update(
                max(0.0, speed_x), sensors.get("gear")
            )
        raise RuntimeError("Stato recovery non valido.")


class RuntimePolicy:
    def __init__(self, model):
        self.model = model
        self.recovery = RecoveryController()
        self.gears = GearController()
        self.previous_steer = 0.0
        self.previous_pedal = 0.0

    def predict(self, sensors):
        features = feature_vector(sensors)
        started = time.perf_counter()
        prediction = predict_array(self.model, features.reshape(1, -1))[0]
        inference_ms = (time.perf_counter() - started) * 1000.0
        return features, float(prediction[0]), float(prediction[1]), inference_ms

    def action(self, sensors):
        features, network_steer, network_pedal, inference_ms = self.predict(
            sensors
        )
        self.recovery.update_detection(sensors)
        if self.recovery.state == NORMAL:
            speed_x = abs(safe_float(sensors.get("speedX")))
            steer_rate = (
                0.018 if speed_x > 170.0
                else (0.026 if speed_x > 100.0 else 0.040)
            )
            steer = self.previous_steer + clamp(
                network_steer - self.previous_steer,
                -steer_rate,
                steer_rate,
            )
            pedal = self.previous_pedal + clamp(
                network_pedal - self.previous_pedal,
                -0.20,
                0.20,
            )
            gear = self.gears.update(
                max(0.0, safe_float(sensors.get("speedX"))),
                sensors.get("gear"),
            )
        else:
            steer, pedal, gear = self.recovery.action(sensors)
        accel = max(0.0, pedal)
        brake = max(0.0, -pedal)
        if brake > 0.0:
            accel = 0.0
        action = {
            "steer": clamp(steer, -1.0, 1.0),
            "accel": clamp(accel, 0.0, 1.0),
            "brake": clamp(brake, 0.0, 1.0),
            "gear": int(clamp(gear, -1, 6)),
            "clutch": 0.0,
            "meta": 0,
        }
        diagnostics = {
            "features": features,
            "network_steer": network_steer,
            "network_pedal": network_pedal,
            "inference_ms": inference_ms,
            "recovery": self.recovery.state,
        }
        self.sync_action(action)
        return action, diagnostics

    def sync_action(self, action):
        self.previous_steer = clamp(
            safe_float(action.get("steer")), -1.0, 1.0
        )
        self.previous_pedal = clamp(
            safe_float(action.get("accel"))
            - safe_float(action.get("brake")),
            -1.0,
            1.0,
        )


def axis_value(joystick, index):
    if index < 0 or index >= joystick.get_numaxes():
        return 0.0
    return safe_float(joystick.get_axis(index))


def curve_axis(value, deadzone, progression):
    if abs(value) <= deadzone:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    normalized = (abs(value) - deadzone) / (1.0 - deadzone)
    return sign * clamp(normalized, 0.0, 1.0) ** progression


def normalize_trigger(value):
    value = (value + 1.0) / 2.0 if value < -0.05 else value
    return clamp(value, 0.0, 1.0)


class DaggerController:
    def __init__(self):
        global pygame
        try:
            import pygame as pygame_module
        except ImportError as error:
            raise RuntimeError("pygame necessario per --dagger.") from error
        pygame = pygame_module
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= 0:
            raise RuntimeError("Nessun controller rilevato.")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.adas = ManualADAS()
        self.was_active = False

    def read(self, sensors, current_action):
        pygame.event.pump()
        active = (
            TAKEOVER_BUTTON < self.joystick.get_numbuttons()
            and bool(self.joystick.get_button(TAKEOVER_BUTTON))
        )
        if active and not self.was_active:
            self.adas.begin_takeover(current_action)
            print("\n[DAGGER] Takeover L1 attivo.")
        if not active and self.was_active:
            print("\n[DAGGER] Controllo restituito alla rete.")
        self.was_active = active
        if not active:
            return False, None, None
        raw_steer = axis_value(self.joystick, STEER_AXIS)
        if INVERT_STEERING:
            raw_steer *= -1.0
        intention = {
            "steer": curve_axis(
                raw_steer, STEER_DEADZONE, STEER_PROGRESSION
            ),
            "accel": curve_axis(
                normalize_trigger(axis_value(self.joystick, R2_AXIS)),
                TRIGGER_DEADZONE,
                TRIGGER_PROGRESSION,
            ),
            "brake": curve_axis(
                normalize_trigger(axis_value(self.joystick, L2_AXIS)),
                TRIGGER_DEADZONE,
                TRIGGER_PROGRESSION,
            ),
        }
        intention["accel"] = max(0.0, intention["accel"])
        intention["brake"] = max(0.0, intention["brake"])
        return True, intention, self.adas.apply(sensors, intention)

    def close(self):
        pygame.joystick.quit()
        pygame.quit()


class ManualADAS:
    def __init__(self):
        self.gears = GearController()
        self.reset_from_sensors()

    def reset_from_sensors(self):
        self.steer = 0.0
        self.steer_target = 0.0
        self.accel = 0.0
        self.brake = 0.0

    def reset_from_action(self, action):
        self.steer = safe_float(action.get("steer"))
        self.steer_target = self.steer
        self.accel = safe_float(action.get("accel"))
        self.brake = safe_float(action.get("brake"))

    def begin_takeover(self, action):
        # Preserve steering continuity but remove the network pedal
        # immediately so the expert correction is not contaminated by it.
        self.steer = safe_float(action.get("steer"))
        self.steer_target = self.steer
        self.accel = 0.0
        self.brake = 0.0

    def apply(self, sensors, intent):
        speed = abs(safe_float(sensors.get("speedX")))
        target_steer = clamp(intent["steer"], -1.0, 1.0)
        speed_ratio = clamp(speed / SPEED_FOR_MIN_STEER, 0.0, 1.0)
        steer_limit = MAX_STEER_LOW_SPEED + (
            MAX_STEER_HIGH_SPEED - MAX_STEER_LOW_SPEED
        ) * speed_ratio
        target_steer = clamp(target_steer, -steer_limit, steer_limit)
        self.steer_target += STEER_TARGET_FILTER * (
            target_steer - self.steer_target
        )
        target_steer = self.steer_target
        throttle_factor = 1.0
        if abs(target_steer) > THROTTLE_STEER_START:
            ratio = clamp(
                (abs(target_steer) - THROTTLE_STEER_START)
                / (THROTTLE_STEER_FULL - THROTTLE_STEER_START),
                0.0,
                1.0,
            )
            throttle_factor = 1.0 + (
                THROTTLE_STEER_MIN_ACCEL - 1.0
            ) * ratio
        target_accel = clamp(intent["accel"], 0.0, 1.0) * throttle_factor
        target_brake = clamp(intent["brake"], 0.0, 1.0)
        steer_rate = 0.018 if speed > 170 else (0.026 if speed > 100 else 0.04)
        self.steer += clamp(
            (target_steer - self.steer) * STEER_SMOOTHING,
            -steer_rate,
            steer_rate,
        )
        self.accel += PEDAL_SMOOTHING * (target_accel - self.accel)
        brake_alpha = 0.38 if target_brake < self.brake else PEDAL_SMOOTHING
        self.brake += brake_alpha * (target_brake - self.brake)
        output_accel = self.accel
        output_brake = self.brake
        wheels = safe_list(sensors.get("wheelSpinVel"), 4, 0.0)
        wheel_speed = [
            abs(wheels[index]) * WHEEL_RADII[index] for index in range(4)
        ]
        vehicle_speed = speed / 3.6
        mean_speed = statistics.mean(wheel_speed)
        driven_speed = statistics.mean(wheel_speed[2:4])
        abs_slip = max(0.0, vehicle_speed - mean_speed)
        if speed >= ABS_MIN_SPEED_KMH and abs_slip > ABS_SLIP_START_MPS:
            release = clamp(
                (abs_slip - ABS_SLIP_START_MPS)
                / (ABS_SLIP_FULL_MPS - ABS_SLIP_START_MPS),
                0.0,
                1.0,
            ) * ABS_MAX_RELEASE
            output_brake *= 1.0 - release
        traction_slip = max(0.0, driven_speed - vehicle_speed)
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
            "gear": self.gears.update(speed, sensors.get("gear")),
            "clutch": 0.0,
            "meta": 0,
        }


def dataset_row(
    sensors,
    intention,
    action,
    run_id,
    step,
    previous_steer=0.0,
    previous_pedal=0.0,
):
    return {
        "run_id": run_id,
        "step": step,
        "curLapTime": safe_float(sensors.get("curLapTime")),
        "steer_intent": intention["steer"],
        "accel_intent": intention["accel"],
        "brake_intent": intention["brake"],
        "steer_action": action["steer"],
        "accel_action": action["accel"],
        "brake_action": action["brake"],
        "gear_action": action["gear"],
        "speedX": safe_float(sensors.get("speedX")),
        "speedY": safe_float(sensors.get("speedY")),
        "speedZ": safe_float(sensors.get("speedZ")),
        "wheelSpinVel": json.dumps(
            safe_list(sensors.get("wheelSpinVel"), 4, 0.0)
        ),
        "z": safe_float(sensors.get("z")),
        "track": json.dumps(safe_list(sensors.get("track"), 19, -1.0)),
        "trackPos": safe_float(sensors.get("trackPos")),
        "angle": safe_float(sensors.get("angle")),
        "rpm": safe_float(sensors.get("rpm")),
        "damage": safe_float(sensors.get("damage")),
        "distFromStart": safe_float(sensors.get("distFromStart")),
        "previous_steer": previous_steer,
        "previous_pedal": previous_pedal,
    }


class DaggerWriter:
    def __init__(self):
        self.run_id = "dagger_%s" % time.time_ns()
        self.rows = []

    def append(
        self,
        sensors,
        intention,
        action,
        step,
        previous_steer,
        previous_pedal,
    ):
        if not DemonstrationDataset._valid_dagger_row(sensors):
            return
        self.rows.append(
            dataset_row(
                sensors,
                intention,
                action,
                self.run_id,
                step,
                previous_steer,
                previous_pedal,
            )
        )

    def close(self):
        if not self.rows:
            return 0
        exists = os.path.exists(DAGGER_PATH) and os.path.getsize(DAGGER_PATH) > 0
        with open(DAGGER_PATH, "a", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=DATASET_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerows(self.rows)
        return len(self.rows)


class TraceLogger:
    FIELDS = (
        "step", "curLapTime", "distFromStart", "speedX", "speedY",
        "trackPos", "angle", "damage", "network_steer", "network_pedal",
        "final_steer", "final_accel", "final_brake", "final_gear",
        "takeover", "recovery", "inference_ms", "offtrack",
    )

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.file = open(TRACE_PATH, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()

    def write(self, step, sensors, action, data, takeover):
        self.writer.writerow(
            {
                "step": step,
                "curLapTime": safe_float(sensors.get("curLapTime")),
                "distFromStart": safe_float(sensors.get("distFromStart")),
                "speedX": safe_float(sensors.get("speedX")),
                "speedY": safe_float(sensors.get("speedY")),
                "trackPos": safe_float(sensors.get("trackPos")),
                "angle": safe_float(sensors.get("angle")),
                "damage": safe_float(sensors.get("damage")),
                "network_steer": data["network_steer"],
                "network_pedal": data["network_pedal"],
                "final_steer": action["steer"],
                "final_accel": action["accel"],
                "final_brake": action["brake"],
                "final_gear": action["gear"],
                "takeover": int(takeover),
                "recovery": data["recovery"],
                "inference_ms": data["inference_ms"],
                "offtrack": int(
                    abs(safe_float(sensors.get("trackPos"))) >= 1.0
                ),
            }
        )

    def close(self):
        self.file.close()


def analyze_model(dataset, model):
    prediction = predict_array(model, dataset.features)
    result = metrics(prediction, dataset.targets)
    latencies = []
    for features in dataset.features[:2000]:
        started = time.perf_counter()
        predict_array(model, features.reshape(1, -1))
        latencies.append((time.perf_counter() - started) * 1000.0)
    result["inference_p95_ms"] = percentile(latencies, 0.95)
    result["samples"] = len(dataset.rows)
    result["expert_runs"] = len(dataset.expert_runs)
    result["dagger_samples"] = sum(
        len(run["rows"]) for run in dataset.runs if run["source"] == "dagger"
    )
    return result


def write_analysis(rows, path=ANALYSIS_PATH):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_training(epochs):
    dataset = DemonstrationDataset(include_dagger=True)
    print(
        "[DATASET] %d giri esperti, %d campioni totali."
        % (len(dataset.expert_runs), len(dataset.rows))
    )
    loo = leave_one_lap_out(dataset)
    knn = knn_comparison(dataset)
    model = train_model(
        dataset.features,
        dataset.targets,
        dataset.sample_weights,
        epochs,
    )
    save_model(model, dataset, epochs)
    final = analyze_model(dataset, model)
    final.update({"run_id": "MLP_TRAIN", "samples": len(dataset.rows)})
    write_analysis(loo + [knn, final])
    aggregate = loo[-1]
    print(
        "[LOO MLP] steer %.5f pedal %.5f brake %.5f"
        % (
            aggregate["steer_mae"],
            aggregate["pedal_mae"],
            aggregate["brake_mae"],
        )
    )
    print(
        "[LOO KNN] steer %.5f pedal %.5f brake %.5f"
        % (knn["steer_mae"], knn["pedal_mae"], knn["brake_mae"])
    )
    print("[MODEL] %s" % MODEL_PATH)
    print("[REPORT] %s" % ANALYSIS_PATH)


def run_analysis():
    dataset = DemonstrationDataset(include_dagger=True)
    model, metadata = load_model()
    result = analyze_model(dataset, model)
    write_analysis(
        [{"run_id": "MLP_CURRENT", **result}],
        path=CURRENT_ANALYSIS_PATH,
    )
    print(
        "[MODEL] version=%s expert=%s dagger=%s epochs=%s"
        % (
            metadata.get("version"),
            metadata.get("expert_samples"),
            metadata.get("dagger_samples"),
            metadata.get("epochs"),
        )
    )
    print(
        "[ANALYSIS] steer %.5f pedal %.5f brake %.5f p95 %.3f ms"
        % (
            result["steer_mae"],
            result["pedal_mae"],
            result["brake_mae"],
            result["inference_p95_ms"],
        )
    )
    if result["inference_p95_ms"] >= 5.0:
        raise RuntimeError("Inferenza V7 oltre il gate di 5 ms.")
    print("[REPORT] %s" % CURRENT_ANALYSIS_PATH)


def create_client(arguments):
    original = list(sys.argv)
    try:
        sys.argv = [original[0]] + list(arguments)
        return snakeoil3.Client(p=PORT, vision=False)
    finally:
        sys.argv = original


def run_driver(dagger, snakeoil_arguments):
    model, metadata = load_model()
    policy = RuntimePolicy(model)
    controller = DaggerController() if dagger else None
    dagger_writer = DaggerWriter() if dagger else None
    trace = TraceLogger()
    client = create_client(snakeoil_arguments)
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
            previous_steer = policy.previous_steer
            previous_pedal = policy.previous_pedal
            network_action, diagnostics = policy.action(sensors)
            takeover = False
            action = network_action
            if controller is not None:
                takeover, intention, expert_action = controller.read(
                    sensors, network_action
                )
                if takeover:
                    policy.recovery.reset()
                    action = expert_action
                    dagger_writer.append(
                        sensors,
                        intention,
                        action,
                        step,
                        previous_steer,
                        previous_pedal,
                    )
                    policy.sync_action(action)
                    diagnostics["recovery"] = "TAKEOVER"
            client.R.d.update(action)
            trace.write(step, sensors, action, diagnostics, takeover)
            client.respond_to_server()
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    finally:
        trace.close()
        saved = dagger_writer.close() if dagger_writer is not None else 0
        if controller is not None:
            controller.close()
        client.shutdown()
        print("[STOP] %s" % reason)
        print("[TRACE] %s" % TRACE_PATH)
        if dagger:
            print("[DAGGER] %d correzioni salvate in %s" % (saved, DAGGER_PATH))
        print(
            "[MODEL] expert=%s dagger=%s"
            % (
                metadata.get("expert_samples"),
                metadata.get("dagger_samples"),
            )
        )


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="V7 TORCS behavioral cloning PyTorch con DAgger."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    mode.add_argument("--dagger", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    return parser.parse_known_args(argv)


def main():
    arguments, snakeoil_arguments = parse_arguments()
    if arguments.train:
        run_training(max(1, arguments.epochs))
    elif arguments.analyze_only:
        run_analysis()
    else:
        run_driver(arguments.dagger, snakeoil_arguments)


if __name__ == "__main__":
    main()
