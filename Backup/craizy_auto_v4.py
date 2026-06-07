import csv
import math
import os
import statistics
import time

import snakeoil3_jm2 as snakeoil3


# ============================================================
# crAIzy pilots - automatic driver V4 with KNN
# ============================================================

PORT = 3001
MAX_STEPS = 100000
DRIVER_VERSION = "craizy_auto_v4_knn"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "torcs_ps4_dataset.csv")
AUTO_RESULTS_PATH = os.path.join(BASE_DIR, "auto_v4_runs.csv")
TRACE_PATH = os.path.join(BASE_DIR, "auto_v4_trace.csv")
TRACE_EVERY = 5

PROFILE_BIN_METERS = 5.0
MIN_COMPLETE_LAP_ROWS = 800
MIN_COMPLETE_LAP_DISTANCE = 3500.0
EXPERT_SPEED_SCALE = 1.00
EXPERT_STEER_FEEDFORWARD = 0.48
EXPERT_ANGLE_GAIN = 1.70
EXPERT_POSITION_GAIN = 0.58
EXPERT_LATERAL_DAMPING = 0.004
EXPERT_MAX_LINE_ERROR = 1.05
EXPERT_MAX_ANGLE_ERROR = 0.58
BRAKING_DECEL_MPS2 = 7.2

KNN_NEIGHBORS = 7
KNN_FEATURE_SCALES = (35.0, 0.35, 0.18, 45.0, 8.0)
KNN_FULL_CONFIDENCE_DISTANCE = 0.55
KNN_ZERO_CONFIDENCE_DISTANCE = 2.20
KNN_RECOVERY_CONFIDENCE = 0.35

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

STRAIGHT_TARGET_SPEED = 235.0
MIN_CORNER_SPEED = 90.0
CURVE_FILTER = 0.18
CURVE_SIGNAL_FILTER = 0.14

SHIFT_COOLDOWN = 0.35
UPSHIFT_RPM = 7600.0
DOWNSHIFT_RPM = 3300.0
PANIC_DOWNSHIFT_RPM = 2300.0
UPSHIFT_SPEED = {1: 45.0, 2: 78.0, 3: 112.0, 4: 148.0, 5: 184.0}
DOWNSHIFT_SPEED = {2: 30.0, 3: 58.0, 4: 90.0, 5: 122.0, 6: 155.0}
MIN_SPEED_FOR_UPSHIFT = {1: 22.0, 2: 48.0, 3: 78.0, 4: 110.0, 5: 145.0}

DATASET_COLUMNS = [
    "steer", "accel", "brake", "gear",
    "speedX", "speedY", "speedZ",
    "wheelSpinVel", "z", "track", "trackPos", "angle",
    "rpm", "damage", "distFromStart",
]


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def safe_float(value, default=0.0):
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def safe_list(value, length, default_value=0.0):
    if not isinstance(value, (list, tuple)):
        return [default_value] * length

    values = [safe_float(item, default_value) for item in value[:length]]
    if len(values) < length:
        values += [default_value] * (length - len(values))
    return values


def linear_limit(value, start, full, minimum):
    value = abs(value)
    if value <= start:
        return 1.0
    if value >= full:
        return minimum
    ratio = (value - start) / max(full - start, 0.0001)
    return 1.0 + (minimum - 1.0) * ratio


class AutomaticGearbox:
    def __init__(self):
        self.gear = 1
        self.last_shift_time = 0.0

    def reset(self):
        self.gear = 1
        self.last_shift_time = 0.0

    def update(self, sensors):
        now = time.monotonic()
        speed = abs(safe_float(sensors.get("speedX")))
        rpm = safe_float(sensors.get("rpm"))
        current = int(clamp(self.gear, 1, 6))

        if now - self.last_shift_time < SHIFT_COOLDOWN:
            return current

        new_gear = current
        if current > 1:
            too_slow = speed < DOWNSHIFT_SPEED.get(current, 0.0)
            rpm_too_low = 0.0 < rpm < PANIC_DOWNSHIFT_RPM
            rpm_low_and_slow = (
                0.0 < rpm < DOWNSHIFT_RPM
                and speed < DOWNSHIFT_SPEED.get(current, 0.0) + 12.0
            )
            if too_slow or rpm_too_low or rpm_low_and_slow:
                new_gear = current - 1

        if new_gear == current and current < 6:
            enough_speed = speed >= MIN_SPEED_FOR_UPSHIFT.get(current, 999.0)
            if (rpm >= UPSHIFT_RPM and enough_speed) or speed >= UPSHIFT_SPEED.get(current, 999.0):
                new_gear = current + 1

        new_gear = int(clamp(new_gear, 1, 6))
        if new_gear != current:
            self.last_shift_time = now
        self.gear = new_gear
        return new_gear


class SharedADAS:
    """Applies the same physical assists to human and automatic intentions."""

    def __init__(self):
        self.gearbox = AutomaticGearbox()
        self.reset()

    def reset(self):
        self.steer = 0.0
        self.steer_target = 0.0
        self.accel = 0.0
        self.brake = 0.0
        self.gearbox.reset()

    def apply(self, sensors, intent):
        speed = abs(safe_float(sensors.get("speedX")))
        target_steer = clamp(safe_float(intent.get("steer")), -1.0, 1.0)
        target_accel = clamp(safe_float(intent.get("accel")), 0.0, 1.0)
        target_brake = clamp(safe_float(intent.get("brake")), 0.0, 1.0)

        speed_ratio = clamp(speed / SPEED_FOR_MIN_STEER, 0.0, 1.0)
        steer_limit = (
            MAX_STEER_LOW_SPEED
            + (MAX_STEER_HIGH_SPEED - MAX_STEER_LOW_SPEED) * speed_ratio
        )
        target_steer = clamp(target_steer, -steer_limit, steer_limit)

        self.steer_target += STEER_TARGET_FILTER * (
            target_steer - self.steer_target
        )
        target_steer = self.steer_target
        target_deadzone = 0.012 if speed > 100.0 else 0.006
        if abs(target_steer) < target_deadzone:
            target_steer = 0.0

        throttle_limit = linear_limit(
            target_steer,
            THROTTLE_STEER_START,
            THROTTLE_STEER_FULL,
            THROTTLE_STEER_MIN_ACCEL,
        )
        target_accel *= throttle_limit

        if speed > 170.0:
            steer_rate = 0.018
        elif speed > 100.0:
            steer_rate = 0.026
        else:
            steer_rate = 0.040

        steer_step = (target_steer - self.steer) * STEER_SMOOTHING
        self.steer += clamp(steer_step, -steer_rate, steer_rate)
        self.accel += PEDAL_SMOOTHING * (target_accel - self.accel)
        brake_smoothing = 0.38 if target_brake < self.brake else PEDAL_SMOOTHING
        self.brake += brake_smoothing * (target_brake - self.brake)
        output_accel = self.accel
        output_brake = self.brake

        wheel_spin = safe_list(sensors.get("wheelSpinVel"), 4)
        wheel_speed = [
            abs(wheel_spin[index]) * WHEEL_RADII[index]
            for index in range(4)
        ]
        vehicle_speed = speed / 3.6
        mean_wheel_speed = sum(wheel_speed) / len(wheel_speed)
        driven_wheel_speed = (wheel_speed[2] + wheel_speed[3]) / 2.0

        abs_slip = max(0.0, vehicle_speed - mean_wheel_speed)
        abs_release = 0.0
        if speed >= ABS_MIN_SPEED_KMH and abs_slip > ABS_SLIP_START_MPS:
            abs_release = clamp(
                (abs_slip - ABS_SLIP_START_MPS)
                / max(ABS_SLIP_FULL_MPS - ABS_SLIP_START_MPS, 0.001),
                0.0,
                1.0,
            ) * ABS_MAX_RELEASE
            output_brake *= 1.0 - abs_release

        traction_slip = max(0.0, driven_wheel_speed - vehicle_speed)
        traction_cut = 0.0
        if traction_slip > TCS_SLIP_START_MPS:
            traction_cut = clamp(
                (traction_slip - TCS_SLIP_START_MPS)
                / max(TCS_SLIP_FULL_MPS - TCS_SLIP_START_MPS, 0.001),
                0.0,
                1.0,
            ) * TCS_MAX_CUT
            output_accel *= 1.0 - traction_cut

        if target_brake > 0.05 or output_brake > 0.05:
            output_accel = 0.0

        action = {
            "steer": clamp(self.steer, -1.0, 1.0),
            "accel": clamp(output_accel, 0.0, 1.0),
            "brake": clamp(output_brake, 0.0, 1.0),
            "gear": self.gearbox.update(sensors),
            "clutch": 0.0,
            "meta": 0,
        }
        diagnostics = {
            "steer_limit": steer_limit,
            "abs_slip": abs_slip,
            "abs_release": abs_release,
            "traction_slip": traction_slip,
            "traction_cut": traction_cut,
        }
        return action, diagnostics


class ExpertProfile:
    def __init__(self, path=DATASET_PATH, laps=None):
        self.path = path
        self.bins = {}
        self.rows = 0
        self.laps = 0
        self.track_length = 0.0
        self.complete_laps = []
        if laps is None:
            self.load()
        else:
            self._build(laps)

    @property
    def available(self):
        return bool(self.bins)

    @staticmethod
    def _split_laps(rows):
        if not rows:
            return []

        laps = []
        start = 0
        previous_distance = safe_float(rows[0].get("distFromStart"))
        for index in range(1, len(rows)):
            distance = safe_float(rows[index].get("distFromStart"))
            distance_jump = distance - previous_distance
            if abs(distance_jump) > 250.0 or distance_jump < -2.0:
                laps.append(rows[start:index])
                start = index
            previous_distance = distance
        laps.append(rows[start:])
        return laps

    @staticmethod
    def _is_complete_lap(rows):
        if len(rows) < MIN_COMPLETE_LAP_ROWS:
            return False
        distances = [safe_float(row.get("distFromStart")) for row in rows]
        return min(distances) < 80.0 and max(distances) >= MIN_COMPLETE_LAP_DISTANCE

    @staticmethod
    def _weighted_average(values):
        total = sum(value * weight for value, weight in values)
        weight_total = sum(weight for _, weight in values)
        return total / max(weight_total, 0.0001)

    def load(self):
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return

        try:
            with open(self.path, newline="", encoding="utf-8-sig") as dataset_file:
                reader = csv.DictReader(dataset_file)
                if reader.fieldnames != DATASET_COLUMNS:
                    print("[DATASET] Schema non compatibile, profilo umano disattivato.")
                    return
                all_rows = list(reader)
        except (OSError, csv.Error) as error:
            print("[DATASET] Impossibile leggere il profilo umano: %s" % error)
            return

        complete_laps = [
            lap for lap in self._split_laps(all_rows)
            if self._is_complete_lap(lap)
        ]
        if not complete_laps:
            print("[DATASET] Nessun giro completo trovato nel CSV.")
            return
        self._build(complete_laps)

    def _build(self, complete_laps):
        self.bins = {}
        self.rows = 0
        self.laps = 0
        self.track_length = 0.0
        self.complete_laps = []
        fields = ("steer", "accel", "brake", "speedX", "angle", "trackPos")
        lap_profiles = []
        for lap in complete_laps:
            lap_samples = {}
            clean_lap = []
            for row in lap:
                if abs(safe_float(row.get("trackPos"))) > 1.05:
                    continue
                clean_lap.append(row)
                distance = max(0.0, safe_float(row.get("distFromStart")))
                bin_id = int(distance // PROFILE_BIN_METERS)
                values = lap_samples.setdefault(
                    bin_id,
                    {key: [] for key in fields},
                )
                for key in fields:
                    values[key].append(safe_float(row.get(key)))
                self.rows += 1

            if clean_lap:
                self.complete_laps.append(clean_lap)
            lap_profile = {}
            for bin_id, values in lap_samples.items():
                if len(values["speedX"]) < 2:
                    continue
                lap_profile[bin_id] = {
                    key: statistics.median(values[key])
                    for key in fields
                }
            lap_profiles.append(lap_profile)

        raw_bins = {}
        all_bin_ids = sorted({
            bin_id
            for lap_profile in lap_profiles
            for bin_id in lap_profile
        })
        for bin_id in all_bin_ids:
            raw_bins[bin_id] = {
                key: statistics.median([
                    lap_profile[bin_id][key]
                    for lap_profile in lap_profiles
                    if bin_id in lap_profile
                ])
                for key in fields
            }

        if not raw_bins:
            return

        for bin_id in raw_bins:
            line_neighbours = []
            pedal_neighbours = []
            weights = ((-2, 1.0), (-1, 2.0), (0, 3.0), (1, 2.0), (2, 1.0))
            for offset, weight in weights:
                neighbour = raw_bins.get(bin_id + offset)
                if neighbour is not None:
                    line_neighbours.append((neighbour, weight))
                    if abs(offset) <= 1:
                        pedal_neighbours.append((neighbour, weight))

            self.bins[bin_id] = {
                "steer": self._weighted_average([
                    (clamp(item["steer"], -1.0, 1.0), weight)
                    for item, weight in line_neighbours
                ]),
                "angle": self._weighted_average([
                    (item["angle"], weight)
                    for item, weight in line_neighbours
                ]),
                "trackPos": self._weighted_average([
                    (item["trackPos"], weight)
                    for item, weight in line_neighbours
                ]),
                "speedX": self._weighted_average([
                    (item["speedX"], weight)
                    for item, weight in pedal_neighbours
                ]),
                "accel": self._weighted_average([
                    (item["accel"], weight)
                    for item, weight in pedal_neighbours
                ]),
                "brake": self._weighted_average([
                    (item["brake"], weight)
                    for item, weight in pedal_neighbours
                ]),
            }
        self.laps = len(self.complete_laps)
        self.track_length = (max(self.bins) + 1) * PROFILE_BIN_METERS

    def reference_at(self, distance):
        if not self.available:
            return None

        distance = safe_float(distance) % self.track_length
        bin_position = distance / PROFILE_BIN_METERS
        lower_id = int(bin_position)
        upper_id = lower_id + 1
        if upper_id > max(self.bins):
            upper_id = min(self.bins)
        lower = self.bins.get(lower_id)
        upper = self.bins.get(upper_id)

        if lower is None:
            return upper
        if upper is None:
            return lower

        fraction = bin_position - lower_id
        return {
            key: lower[key] + (upper[key] - lower[key]) * fraction
            for key in lower
        }

    def target_speed(self, distance, current_speed):
        reference = self.reference_at(distance)
        if reference is None:
            return None

        target = reference["speedX"] * EXPERT_SPEED_SCALE
        lookahead = clamp(35.0 + current_speed * 0.28, 45.0, 105.0)
        offset = PROFILE_BIN_METERS
        while offset <= lookahead:
            future_distance = (distance + offset) % self.track_length
            future = self.reference_at(future_distance)
            if future is not None:
                future_speed = max(0.0, future["speedX"] * EXPERT_SPEED_SCALE)
                allowed = math.sqrt(
                    (future_speed / 3.6) ** 2
                    + 2.0 * BRAKING_DECEL_MPS2 * offset
                ) * 3.6
                target = min(target, allowed)
            offset += PROFILE_BIN_METERS
        return target


class KNNImitationModel:
    """Local imitation model with fixed, domain-specific feature scaling."""

    def __init__(
        self,
        laps,
        track_length,
        neighbours=KNN_NEIGHBORS,
    ):
        self.track_length = safe_float(track_length)
        self.neighbours = 0
        self.samples = 0
        self.model = None

        rows = [row for lap in laps for row in lap]
        if not rows or self.track_length <= 0.0:
            return

        try:
            from sklearn.neighbors import KNeighborsRegressor
        except ImportError as error:
            raise RuntimeError(
                "La policy KNN richiede scikit-learn. "
                "Installa le dipendenze con: python -m pip install -r requirements.txt"
            ) from error

        features = []
        targets = []
        for row in rows:
            distance = safe_float(row.get("distFromStart"))
            target = [
                clamp(safe_float(row.get("steer")), -1.0, 1.0),
                clamp(safe_float(row.get("accel")), 0.0, 1.0),
                clamp(safe_float(row.get("brake")), 0.0, 1.0),
            ]
            for lap_offset in (-self.track_length, 0.0, self.track_length):
                features.append(self._features(
                    distance + lap_offset,
                    safe_float(row.get("trackPos")),
                    safe_float(row.get("angle")),
                    safe_float(row.get("speedX")),
                    safe_float(row.get("speedY")),
                ))
                targets.append(target)

        self.neighbours = min(max(1, int(neighbours)), len(features))
        self.model = KNeighborsRegressor(
            n_neighbors=self.neighbours,
            weights="distance",
            metric="euclidean",
        )
        self.model.fit(features, targets)
        self.samples = len(rows)

    @property
    def available(self):
        return self.model is not None

    @staticmethod
    def _features(distance, track_pos, angle, speed_x, speed_y):
        values = (distance, track_pos, angle, speed_x, speed_y)
        return [
            safe_float(value) / scale
            for value, scale in zip(values, KNN_FEATURE_SCALES)
        ]

    def predict_values(self, distance, track_pos, angle, speed_x, speed_y):
        if not self.available:
            return None, 0.0, float("inf")

        query = [self._features(
            safe_float(distance) % self.track_length,
            track_pos,
            angle,
            speed_x,
            speed_y,
        )]
        distances, _ = self.model.kneighbors(
            query,
            n_neighbors=self.neighbours,
            return_distance=True,
        )
        mean_distance = sum(float(value) for value in distances[0]) / self.neighbours
        confidence = 1.0 - clamp(
            (mean_distance - KNN_FULL_CONFIDENCE_DISTANCE)
            / (
                KNN_ZERO_CONFIDENCE_DISTANCE
                - KNN_FULL_CONFIDENCE_DISTANCE
            ),
            0.0,
            1.0,
        )
        prediction = self.model.predict(query)[0]
        return {
            "steer": clamp(float(prediction[0]), -1.0, 1.0),
            "accel": clamp(float(prediction[1]), 0.0, 1.0),
            "brake": clamp(float(prediction[2]), 0.0, 1.0),
        }, confidence, mean_distance

    def predict(self, sensors):
        return self.predict_values(
            safe_float(sensors.get("distFromStart")),
            safe_float(sensors.get("trackPos")),
            safe_float(sensors.get("angle")),
            safe_float(sensors.get("speedX")),
            safe_float(sensors.get("speedY")),
        )


class FallbackPolicy:
    def __init__(self):
        self.curve_strength = 0.0
        self.curve_signal = 0.0

    @staticmethod
    def _front_speed_limit(front):
        if front >= 160.0:
            return STRAIGHT_TARGET_SPEED
        if front >= 120.0:
            return 190.0 + (front - 120.0) * 1.125
        if front >= 85.0:
            return 150.0 + (front - 85.0) * (40.0 / 35.0)
        if front >= 55.0:
            return 115.0 + (front - 55.0) * (35.0 / 30.0)
        if front >= 30.0:
            return 80.0 + (front - 30.0) * 1.4
        return 72.0

    def intention(self, sensors):
        track = [
            max(0.0, value)
            for value in safe_list(sensors.get("track"), 19, 200.0)
        ]
        front = track[9]
        left_front = sum(track[6:9]) / 3.0
        right_front = sum(track[10:13]) / 3.0
        measured_curve_signal = right_front - left_front
        self.curve_signal += CURVE_SIGNAL_FILTER * (
            measured_curve_signal - self.curve_signal
        )
        curve_signal = self.curve_signal

        front_pressure = clamp((150.0 - front) / 110.0, 0.0, 1.0)
        signal_pressure = clamp((abs(curve_signal) - 8.0) / 82.0, 0.0, 1.0)
        measured_curve = max(front_pressure, signal_pressure)
        self.curve_strength += CURVE_FILTER * (measured_curve - self.curve_strength)
        curve_strength = clamp(self.curve_strength, 0.0, 1.0)

        speed = safe_float(sensors.get("speedX"))
        speed_y = safe_float(sensors.get("speedY"))
        angle = safe_float(sensors.get("angle"))
        track_pos = safe_float(sensors.get("trackPos"))

        angle_deadzone = 0.026 - curve_strength * 0.016
        position_deadzone = 0.055 - curve_strength * 0.030
        steering_angle = 0.0 if abs(angle) < angle_deadzone else angle
        steering_position = 0.0 if abs(track_pos) < position_deadzone else track_pos

        angle_gain = 1.55 + curve_strength * 1.00
        centering_gain = 0.14 - curve_strength * 0.03
        preview_gain = 0.08 + curve_strength * 0.15
        preview = (curve_signal / 200.0) * preview_gain
        lateral_damping = speed_y * (0.0055 - curve_strength * 0.0035)

        steer = (
            steering_angle * angle_gain
            - steering_position * centering_gain
            + preview
            - lateral_damping
        )

        if abs(track_pos) > 0.82:
            steer -= track_pos * 0.22
        if abs(speed_y) > 8.0:
            steer *= 0.88

        policy_steer_limit = 0.22 + curve_strength * 0.58
        steer = clamp(steer, -policy_steer_limit, policy_steer_limit)

        curve_target = (
            STRAIGHT_TARGET_SPEED
            - (STRAIGHT_TARGET_SPEED - MIN_CORNER_SPEED) * (curve_strength ** 0.72)
        )
        target_speed = min(curve_target, self._front_speed_limit(front))

        if abs(track_pos) > 0.90:
            target_speed = min(target_speed, 70.0)
        elif abs(track_pos) > 0.72:
            target_speed = min(target_speed, 105.0)

        if abs(speed_y) > 14.0:
            target_speed = min(target_speed, 90.0)
        elif abs(speed_y) > 9.0:
            target_speed = min(target_speed, 125.0)

        if abs(angle) > 0.42:
            target_speed = min(target_speed, 78.0)
        elif abs(angle) > 0.26:
            target_speed = min(target_speed, 115.0)

        speed_error = target_speed - speed
        if speed_error > 25.0:
            accel = 1.0
        elif speed_error > 10.0:
            accel = 0.85
        elif speed_error > 0.0:
            accel = 0.58
        elif speed_error > -5.0:
            accel = 0.22
        else:
            accel = 0.0

        overspeed = speed - target_speed
        if overspeed > 45.0:
            brake = 0.50
        elif overspeed > 25.0:
            brake = 0.30
        elif overspeed > 12.0:
            brake = 0.15
        else:
            brake = 0.0

        if front < max(32.0, speed * 0.32) and overspeed > 5.0:
            brake = max(brake, 0.30)
        if brake > 0.05:
            accel = 0.0
        elif curve_strength > 0.78 and speed_error < 18.0:
            accel = min(accel, 0.42)

        track_info = {
            "front": front,
            "curve_signal": curve_signal,
            "curve_strength": curve_strength,
            "target_speed": target_speed,
        }
        return {"steer": steer, "accel": accel, "brake": brake}, track_info


class ExpertGuidedPolicy:
    def __init__(self, profile, policy_mode="profile", knn_model=None):
        self.profile = profile
        self.policy_mode = policy_mode
        self.knn_model = knn_model
        self.fallback = FallbackPolicy()
        self.previous_distance = None
        self.launch_mode = None

    def intention(self, sensors):
        fallback_intent, track_info = self.fallback.intention(sensors)
        distance = max(0.0, safe_float(sensors.get("distFromStart")))
        speed = safe_float(sensors.get("speedX"))

        if self.launch_mode is None:
            self.launch_mode = distance > 3000.0 and speed < 30.0
        if (
            self.launch_mode
            and self.previous_distance is not None
            and self.previous_distance - distance > 1000.0
        ):
            self.launch_mode = False
        self.previous_distance = distance

        reference = self.profile.reference_at(distance)
        if self.launch_mode or reference is None:
            if self.launch_mode and speed > 135.0:
                fallback_intent = fallback_intent.copy()
                fallback_intent["accel"] = 0.0
                fallback_intent["brake"] = max(fallback_intent["brake"], 0.12)
                track_info["target_speed"] = min(
                    track_info["target_speed"],
                    135.0,
                )
            track_info.update({
                "mode": "launch" if self.launch_mode else "fallback",
                "policy": self.policy_mode,
                "expert_speed": 0.0,
                "expert_track_pos": 0.0,
                "expert_angle": 0.0,
                "line_error": 0.0,
                "angle_error": 0.0,
                "safety_blend": 1.0,
                "recovery_blend": 1.0,
                "knn_confidence": 0.0,
                "knn_distance": 0.0,
                "knn_steer": 0.0,
                "knn_accel": 0.0,
                "knn_brake": 0.0,
            })
            return fallback_intent, track_info, fallback_intent, None, False

        track_pos = safe_float(sensors.get("trackPos"))
        angle = safe_float(sensors.get("angle"))
        speed_y = safe_float(sensors.get("speedY"))
        line_error = track_pos - reference["trackPos"]
        angle_error = angle - reference["angle"]

        knn_intent = None
        knn_confidence = 0.0
        knn_distance = 0.0
        if self.policy_mode == "knn" and self.knn_model is not None:
            knn_intent, knn_confidence, knn_distance = self.knn_model.predict(
                sensors
            )

        guidance_blend = knn_confidence if knn_intent is not None else 0.0
        feedforward = reference["steer"]
        reference_accel = reference["accel"]
        reference_brake = reference["brake"]
        if knn_intent is not None:
            feedforward += (
                knn_intent["steer"] - feedforward
            ) * guidance_blend
            reference_accel += (
                knn_intent["accel"] - reference_accel
            ) * guidance_blend
            reference_brake += (
                knn_intent["brake"] - reference_brake
            ) * guidance_blend

        feedforward = clamp(feedforward, -0.65, 0.65)
        expert_steer = (
            feedforward * EXPERT_STEER_FEEDFORWARD
            - angle_error * EXPERT_ANGLE_GAIN
            + line_error * EXPERT_POSITION_GAIN
            - speed_y * EXPERT_LATERAL_DAMPING
        )

        line_slowdown = clamp(
            (abs(line_error) - 0.45)
            / max(EXPERT_MAX_LINE_ERROR - 0.45, 0.001),
            0.0,
            1.0,
        )
        angle_danger = clamp(
            (abs(angle_error) - 0.32)
            / max(EXPERT_MAX_ANGLE_ERROR - 0.32, 0.001),
            0.0,
            1.0,
        )
        edge_danger = clamp((abs(track_pos) - 0.94) / 0.16, 0.0, 1.0)
        safety_blend = max(angle_danger, edge_danger)
        line_recovery = clamp(
            (abs(line_error) - 0.75)
            / max(EXPERT_MAX_LINE_ERROR - 0.75, 0.001),
            0.0,
            1.0,
        )
        knn_recovery = 0.0
        if self.policy_mode == "knn":
            knn_recovery = clamp(
                (KNN_RECOVERY_CONFIDENCE - knn_confidence)
                / KNN_RECOVERY_CONFIDENCE,
                0.0,
                1.0,
            )
        recovery_blend = max(safety_blend, line_recovery, knn_recovery)
        if abs(track_pos) > 1.0:
            recovery_blend = 1.0

        target_speed = self.profile.target_speed(distance, speed)
        target_speed *= 1.0 - 0.18 * line_slowdown
        fallback_speed = track_info["target_speed"]
        if recovery_blend > 0.0:
            safety_target = min(target_speed, fallback_speed)
            target_speed += (
                safety_target - target_speed
            ) * recovery_blend

        front = track_info["front"]
        if abs(track_pos) > 1.0:
            target_speed = min(target_speed, 65.0)
        elif abs(track_pos) > 0.90:
            target_speed = min(target_speed, 105.0)
        if abs(speed_y) > 15.0:
            target_speed = min(target_speed, 95.0)
        if front < 18.0 and speed > 100.0:
            target_speed = min(target_speed, 85.0)

        speed_error = target_speed - speed
        if speed_error > 18.0:
            accel = 1.0
        elif speed_error > 8.0:
            accel = max(0.88, reference_accel)
        elif speed_error > 2.0:
            accel = max(0.72, reference_accel * 0.95)
        elif speed_error > -3.0:
            accel = reference_accel * 0.90
        else:
            accel = 0.0

        overspeed = speed - target_speed
        if overspeed > 28.0:
            brake = 0.52
        elif overspeed > 16.0:
            brake = 0.34
        elif overspeed > 7.0:
            brake = 0.18
        elif overspeed > 4.0:
            brake = 0.08
        else:
            brake = 0.0

        if speed > target_speed - 4.0:
            brake = max(brake, reference_brake * 0.70)
        if brake > 0.05:
            accel = 0.0

        expert_intent = {
            "steer": clamp(expert_steer, -1.0, 1.0),
            "accel": clamp(accel, 0.0, 1.0),
            "brake": clamp(brake, 0.0, 1.0),
        }
        steer = expert_intent["steer"] + (
            fallback_intent["steer"] - expert_intent["steer"]
        ) * recovery_blend
        output_accel = expert_intent["accel"] + (
            fallback_intent["accel"] - expert_intent["accel"]
        ) * recovery_blend
        output_brake = expert_intent["brake"] + (
            fallback_intent["brake"] - expert_intent["brake"]
        ) * recovery_blend
        if output_brake > 0.05:
            output_accel = 0.0

        intent = {
            "steer": clamp(steer, -1.0, 1.0),
            "accel": clamp(output_accel, 0.0, 1.0),
            "brake": clamp(output_brake, 0.0, 1.0),
        }
        if recovery_blend >= 0.5:
            mode = "recovery"
        elif self.policy_mode == "knn":
            mode = "knn" if knn_confidence >= 0.85 else "blend"
        else:
            mode = "profile"

        track_info.update({
            "mode": mode,
            "policy": self.policy_mode,
            "target_speed": target_speed,
            "expert_speed": reference["speedX"],
            "expert_track_pos": reference["trackPos"],
            "expert_angle": reference["angle"],
            "line_error": line_error,
            "angle_error": angle_error,
            "safety_blend": safety_blend,
            "recovery_blend": recovery_blend,
            "knn_confidence": knn_confidence,
            "knn_distance": knn_distance,
            "knn_steer": knn_intent["steer"] if knn_intent else 0.0,
            "knn_accel": knn_intent["accel"] if knn_intent else 0.0,
            "knn_brake": knn_intent["brake"] if knn_intent else 0.0,
        })
        return intent, track_info, fallback_intent, expert_intent, True


class LapCompletionDetector:
    """Detects a completed lap even if TORCS closes before lastLapTime arrives."""

    def __init__(self):
        self.previous_distance = None
        self.required_crossings = None
        self.crossings = 0

    def update(self, sensors):
        distance = max(0.0, safe_float(sensors.get("distFromStart")))
        speed = abs(safe_float(sensors.get("speedX")))
        if self.required_crossings is None:
            starts_before_line = distance > 3000.0 and speed < 30.0
            self.required_crossings = 2 if starts_before_line else 1

        if (
            self.previous_distance is not None
            and self.previous_distance - distance > 1000.0
        ):
            self.crossings += 1
        self.previous_distance = distance
        return self.crossings >= self.required_crossings


class TraceLogger:
    FIELDS = [
        "step", "distFromStart", "speedX", "speedY", "angle", "trackPos",
        "policy", "mode", "front", "curve_strength", "target_speed", "expert_speed",
        "expert_track_pos", "expert_angle", "line_error", "angle_error",
        "safety_blend", "recovery_blend", "knn_confidence", "knn_distance",
        "knn_steer", "knn_accel", "knn_brake",
        "base_steer", "expert_steer", "intent_steer", "final_steer",
        "final_accel", "final_brake", "gear", "steer_limit",
        "abs_release", "traction_cut", "profile_used",
    ]

    def __init__(self):
        os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
        self.file = open(TRACE_PATH, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()

    def write(
        self,
        step,
        sensors,
        track_info,
        base_intent,
        expert_intent,
        intent,
        action,
        diagnostics,
        used_profile,
    ):
        if step % TRACE_EVERY != 0:
            return

        row = {
            "step": step,
            "distFromStart": round(safe_float(sensors.get("distFromStart")), 3),
            "speedX": round(safe_float(sensors.get("speedX")), 3),
            "speedY": round(safe_float(sensors.get("speedY")), 3),
            "angle": round(safe_float(sensors.get("angle")), 5),
            "trackPos": round(safe_float(sensors.get("trackPos")), 5),
            "policy": track_info["policy"],
            "mode": track_info["mode"],
            "front": round(track_info["front"], 3),
            "curve_strength": round(track_info["curve_strength"], 4),
            "target_speed": round(track_info["target_speed"], 3),
            "expert_speed": round(track_info["expert_speed"], 3),
            "expert_track_pos": round(track_info["expert_track_pos"], 5),
            "expert_angle": round(track_info["expert_angle"], 5),
            "line_error": round(track_info["line_error"], 5),
            "angle_error": round(track_info["angle_error"], 5),
            "safety_blend": round(track_info["safety_blend"], 4),
            "recovery_blend": round(track_info["recovery_blend"], 4),
            "knn_confidence": round(track_info["knn_confidence"], 5),
            "knn_distance": round(track_info["knn_distance"], 5),
            "knn_steer": round(track_info["knn_steer"], 5),
            "knn_accel": round(track_info["knn_accel"], 5),
            "knn_brake": round(track_info["knn_brake"], 5),
            "base_steer": round(base_intent["steer"], 5),
            "expert_steer": (
                round(expert_intent["steer"], 5)
                if expert_intent is not None
                else ""
            ),
            "intent_steer": round(intent["steer"], 5),
            "final_steer": round(action["steer"], 5),
            "final_accel": round(action["accel"], 5),
            "final_brake": round(action["brake"], 5),
            "gear": action["gear"],
            "steer_limit": round(diagnostics["steer_limit"], 5),
            "abs_release": round(diagnostics["abs_release"], 5),
            "traction_cut": round(diagnostics["traction_cut"], 5),
            "profile_used": int(used_profile),
        }
        self.writer.writerow(row)

    def close(self):
        self.file.close()


class RunSummary:
    FIELDS = [
        "timestamp", "driver_version", "dataset_rows", "profile_bins",
        "profile_steps", "steps", "elapsed_s", "lap_time",
        "avg_speed", "max_speed", "final_damage", "offtrack_steps",
        "avg_abs_steer", "max_abs_steer", "steer_direction_changes",
        "abs_steps", "tcs_steps", "reason",
    ]

    def __init__(self, profile, driver_version):
        self.profile = profile
        self.driver_version = driver_version
        self.start_time = time.time()
        self.steps = 0
        self.profile_steps = 0
        self.speed_sum = 0.0
        self.max_speed = 0.0
        self.offtrack_steps = 0
        self.abs_steer_sum = 0.0
        self.max_abs_steer = 0.0
        self.steer_direction_changes = 0
        self.last_steer_direction = 0
        self.abs_steps = 0
        self.tcs_steps = 0
        self.final_sensors = {}

    def record(self, sensors, action, diagnostics, used_profile):
        speed = safe_float(sensors.get("speedX"))
        track = safe_list(sensors.get("track"), 19, 200.0)
        steer = safe_float(action.get("steer"))
        self.steps += 1
        self.profile_steps += int(used_profile)
        self.speed_sum += speed
        self.max_speed = max(self.max_speed, speed)
        self.abs_steer_sum += abs(steer)
        self.max_abs_steer = max(self.max_abs_steer, abs(steer))

        steer_direction = 0
        if steer > 0.025:
            steer_direction = 1
        elif steer < -0.025:
            steer_direction = -1
        if (
            steer_direction != 0
            and self.last_steer_direction != 0
            and steer_direction != self.last_steer_direction
        ):
            self.steer_direction_changes += 1
        if steer_direction != 0:
            self.last_steer_direction = steer_direction

        self.offtrack_steps += int(
            abs(safe_float(sensors.get("trackPos"))) > 1.0 or min(track) < 0.0
        )
        self.abs_steps += int(diagnostics["abs_release"] > 0.0)
        self.tcs_steps += int(diagnostics["traction_cut"] > 0.0)
        self.final_sensors = sensors.copy()

    def write(self, reason):
        os.makedirs(os.path.dirname(AUTO_RESULTS_PATH), exist_ok=True)
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "driver_version": self.driver_version,
            "dataset_rows": self.profile.rows,
            "profile_bins": len(self.profile.bins),
            "profile_steps": self.profile_steps,
            "steps": self.steps,
            "elapsed_s": round(time.time() - self.start_time, 3),
            "lap_time": round(safe_float(self.final_sensors.get("lastLapTime")), 3),
            "avg_speed": round(self.speed_sum / self.steps, 3) if self.steps else 0.0,
            "max_speed": round(self.max_speed, 3),
            "final_damage": round(safe_float(self.final_sensors.get("damage")), 3),
            "offtrack_steps": self.offtrack_steps,
            "avg_abs_steer": (
                round(self.abs_steer_sum / self.steps, 5)
                if self.steps
                else 0.0
            ),
            "max_abs_steer": round(self.max_abs_steer, 5),
            "steer_direction_changes": self.steer_direction_changes,
            "abs_steps": self.abs_steps,
            "tcs_steps": self.tcs_steps,
            "reason": reason,
        }
        file_exists = (
            os.path.exists(AUTO_RESULTS_PATH)
            and os.path.getsize(AUTO_RESULTS_PATH) > 0
        )
        with open(AUTO_RESULTS_PATH, "a", newline="", encoding="utf-8") as result_file:
            writer = csv.DictWriter(result_file, fieldnames=self.FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def main():
    profile = ExpertProfile(DATASET_PATH)
    knn_model = KNNImitationModel(
        profile.complete_laps,
        profile.track_length,
    )
    if not knn_model.available:
        raise SystemExit(
            "Impossibile avviare il pilota KNN: dataset valido assente."
        )

    policy = ExpertGuidedPolicy(profile, "knn", knn_model)
    adas = SharedADAS()
    summary = RunSummary(profile, DRIVER_VERSION)
    trace = TraceLogger()
    lap_detector = LapCompletionDetector()
    client = snakeoil3.Client(p=PORT)
    reason = "max_steps"

    print("[DRIVER] %s" % DRIVER_VERSION)
    if profile.available:
        print("[DATASET] Profilo umano dominante: %d giro/i, %d righe, %d settori." % (
            profile.laps,
            profile.rows,
            len(profile.bins),
        ))
        print(
            "[KNN] %d campioni, k=%d. "
            "Raccogliere almeno 5 giri per una validazione affidabile."
            % (knn_model.samples, knn_model.neighbours)
        )
    else:
        print("[DATASET] Nessun profilo valido: guida automatica base.")

    try:
        for step in range(MAX_STEPS):
            client.get_servers_input()
            if not client.so:
                reason = "server_closed"
                break

            sensors = client.S.d
            if safe_float(sensors.get("lastLapTime")) > 0.0:
                summary.final_sensors = sensors.copy()
                reason = "lap_complete"
                break
            if lap_detector.update(sensors):
                summary.final_sensors = sensors.copy()
                reason = "lap_complete_distance"
                break

            (
                intent,
                track_info,
                base_intent,
                expert_intent,
                used_profile,
            ) = policy.intention(sensors)
            action, diagnostics = adas.apply(sensors, intent)
            client.R.d.update(action)
            summary.record(sensors, action, diagnostics, used_profile)
            trace.write(
                step,
                sensors,
                track_info,
                base_intent,
                expert_intent,
                intent,
                action,
                diagnostics,
                used_profile,
            )
            client.respond_to_server()
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    finally:
        trace.close()
        summary.write(reason)
        client.shutdown()
        print("Risultato salvato in %s" % AUTO_RESULTS_PATH)
        print("Telemetria salvata in %s" % TRACE_PATH)


if __name__ == "__main__":
    main()
