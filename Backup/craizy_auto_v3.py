import bisect
import csv
import math
import os
import time

import snakeoil3_jm2 as snakeoil3


# ============================================================
# crAIzy pilots - automatic driver V3
# ============================================================

PORT = 3001
MAX_STEPS = 100000
DRIVER_VERSION = "craizy_auto_v3_guarded_replay"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "torcs_ps4_dataset.csv")
AUTO_RESULTS_PATH = os.path.join(BASE_DIR, "auto_v3_runs.csv")
TRACE_PATH = os.path.join(BASE_DIR, "auto_v3_trace.csv")
TRACE_EVERY = 5

MIN_COMPLETE_LAP_ROWS = 800
MIN_COMPLETE_LAP_DISTANCE = 3500.0
REPLAY_POSITION_GAIN = 0.55
REPLAY_ANGLE_GAIN = 0.65
REPLAY_LATERAL_DAMPING = 0.004
REPLAY_MAX_STEER_CORRECTION = 0.40
REPLAY_MAX_LINE_ERROR = 1.00
REPLAY_MAX_ANGLE_ERROR = 0.65
REPLAY_MAX_TRACK_POSITION = 0.98
REPLAY_MAX_LATERAL_SPEED_ERROR = 20.0
REPLAY_RISK_LINE_START = 0.35
REPLAY_RISK_ANGLE_START = 0.20
REPLAY_RISK_MAX_LINE_ERROR = 0.80
REPLAY_RISK_MAX_ANGLE_ERROR = 0.50
REPLAY_ANTI_STALL_SPEED = 15.0
REPLAY_BRAKE_RELEASE_SPEED_ERROR = 35.0

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
    def __init__(self, path=DATASET_PATH):
        self.path = path
        self.bins = {}
        self.samples = []
        self.distances = []
        self.rows = 0
        self.laps = 0
        self.track_length = 0.0
        self.selected_lap = 0
        self.load()

    @property
    def available(self):
        return bool(self.samples)

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

        fields = (
            "steer",
            "accel",
            "brake",
            "speedX",
            "speedY",
            "angle",
            "trackPos",
        )
        self.laps = len(complete_laps)
        selected_index, selected_rows = min(
            enumerate(complete_laps),
            key=lambda item: (
                max(safe_float(row.get("damage")) for row in item[1]) > 0.0,
                max(abs(safe_float(row.get("trackPos"))) for row in item[1])
                >= 1.0,
                len(item[1]),
            ),
        )
        self.selected_lap = selected_index + 1

        samples = []
        for row in selected_rows:
            if abs(safe_float(row.get("trackPos"))) >= 1.0:
                continue
            sample = {
                key: safe_float(row.get(key))
                for key in fields
            }
            sample["steer"] = clamp(sample["steer"], -1.0, 1.0)
            sample["accel"] = clamp(sample["accel"], 0.0, 1.0)
            sample["brake"] = clamp(sample["brake"], 0.0, 1.0)
            sample["distance"] = max(
                0.0,
                safe_float(row.get("distFromStart")),
            )
            samples.append(sample)

        samples.sort(key=lambda sample: sample["distance"])
        self.samples = samples
        self.distances = [sample["distance"] for sample in samples]
        self.rows = len(samples)
        self.bins = {
            index: sample
            for index, sample in enumerate(samples)
        }
        self.track_length = self.distances[-1] if self.distances else 0.0

    def reference_at(self, distance):
        if not self.available:
            return None

        distance = clamp(safe_float(distance), 0.0, self.track_length)
        upper_index = bisect.bisect_left(self.distances, distance)
        if upper_index <= 0:
            return self.samples[0].copy()
        if upper_index >= len(self.samples):
            return self.samples[-1].copy()

        lower = self.samples[upper_index - 1]
        upper = self.samples[upper_index]
        span = upper["distance"] - lower["distance"]
        fraction = (
            (distance - lower["distance"]) / span
            if span > 0.0001
            else 0.0
        )
        return {
            key: lower[key] + (upper[key] - lower[key]) * fraction
            for key in (
                "steer",
                "accel",
                "brake",
                "speedX",
                "speedY",
                "angle",
                "trackPos",
            )
        }

    def target_speed(self, distance, current_speed):
        reference = self.reference_at(distance)
        return reference["speedX"] if reference is not None else None


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
    def __init__(self, profile):
        self.profile = profile
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
                "expert_speed": 0.0,
                "expert_track_pos": 0.0,
                "expert_angle": 0.0,
                "line_error": 0.0,
                "angle_error": 0.0,
                "safety_blend": 1.0,
                "steer_profile_used": False,
                "speed_profile_used": False,
                "critical_steering": False,
            })
            return fallback_intent, track_info, fallback_intent, None, False

        track_pos = safe_float(sensors.get("trackPos"))
        angle = safe_float(sensors.get("angle"))
        speed_y = safe_float(sensors.get("speedY"))
        line_error = track_pos - reference["trackPos"]
        angle_error = angle - reference["angle"]
        lateral_speed_error = speed_y - reference["speedY"]

        steering_correction = clamp(
            angle_error * REPLAY_ANGLE_GAIN
            - line_error * REPLAY_POSITION_GAIN
            - lateral_speed_error * REPLAY_LATERAL_DAMPING,
            -REPLAY_MAX_STEER_CORRECTION,
            REPLAY_MAX_STEER_CORRECTION,
        )
        expert_steer = clamp(
            reference["steer"] + steering_correction,
            -1.0,
            1.0,
        )

        target_speed = reference["speedX"]
        speed_error = target_speed - speed
        accel = reference["accel"]
        brake = reference["brake"]

        if speed_error > 12.0 and brake < 0.10:
            accel = max(accel, clamp(speed_error / 30.0, 0.0, 1.0))
            brake = 0.0
        elif speed_error < -5.0:
            accel = 0.0
            brake = max(
                brake,
                clamp((-speed_error - 5.0) / 25.0, 0.0, 0.65),
            )

        line_risk = clamp(
            (abs(line_error) - REPLAY_RISK_LINE_START)
            / max(
                REPLAY_RISK_MAX_LINE_ERROR - REPLAY_RISK_LINE_START,
                0.001,
            ),
            0.0,
            1.0,
        )
        angle_risk = clamp(
            (abs(angle_error) - REPLAY_RISK_ANGLE_START)
            / max(
                REPLAY_RISK_MAX_ANGLE_ERROR - REPLAY_RISK_ANGLE_START,
                0.001,
            ),
            0.0,
            1.0,
        )
        trajectory_risk = max(line_risk, angle_risk)
        if trajectory_risk > 0.0:
            accel *= 1.0 - 0.85 * trajectory_risk
            if speed > 80.0:
                brake = max(brake, 0.18 * trajectory_risk)

        if (
            speed < REPLAY_ANTI_STALL_SPEED
            and target_speed > REPLAY_ANTI_STALL_SPEED + 20.0
        ) or speed_error > REPLAY_BRAKE_RELEASE_SPEED_ERROR:
            accel = 1.0
            brake = 0.0

        expert_intent = {
            "steer": clamp(expert_steer, -1.0, 1.0),
            "accel": clamp(accel, 0.0, 1.0),
            "brake": clamp(brake, 0.0, 1.0),
        }

        replay_usable = (
            abs(line_error) <= REPLAY_MAX_LINE_ERROR
            and abs(angle_error) <= REPLAY_MAX_ANGLE_ERROR
            and abs(track_pos) <= REPLAY_MAX_TRACK_POSITION
            and abs(lateral_speed_error) <= REPLAY_MAX_LATERAL_SPEED_ERROR
        )
        if replay_usable:
            intent = expert_intent.copy()
            mode = "replay"
            safety_blend = 0.0
        else:
            intent = fallback_intent.copy()
            mode = "recovery"
            safety_blend = 1.0

        track_info.update({
            "mode": mode,
            "target_speed": target_speed if replay_usable else track_info["target_speed"],
            "expert_speed": reference["speedX"],
            "expert_track_pos": reference["trackPos"],
            "expert_angle": reference["angle"],
            "line_error": line_error,
            "angle_error": angle_error,
            "safety_blend": safety_blend,
            "steer_profile_used": replay_usable,
            "speed_profile_used": replay_usable,
            "critical_steering": False,
        })
        return (
            intent,
            track_info,
            fallback_intent,
            expert_intent,
            replay_usable,
        )


class TraceLogger:
    FIELDS = [
        "step", "distFromStart", "speedX", "speedY", "angle", "trackPos",
        "mode", "front", "curve_strength", "target_speed", "expert_speed",
        "expert_track_pos", "expert_angle", "line_error", "angle_error",
        "safety_blend", "steer_profile_used", "speed_profile_used",
        "critical_steering",
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
            "steer_profile_used": int(track_info["steer_profile_used"]),
            "speed_profile_used": int(track_info["speed_profile_used"]),
            "critical_steering": int(track_info["critical_steering"]),
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

    def __init__(self, profile):
        self.profile = profile
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
            "driver_version": DRIVER_VERSION,
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
    profile = ExpertProfile()
    policy = ExpertGuidedPolicy(profile)
    adas = SharedADAS()
    summary = RunSummary(profile)
    trace = TraceLogger()
    client = snakeoil3.Client(p=PORT)
    reason = "max_steps"

    print("[DRIVER] %s" % DRIVER_VERSION)
    if profile.available:
        print(
            "[DATASET] %d giri puliti; replay del giro %d: "
            "%d campioni, %.1f metri."
            % (
                profile.laps,
                profile.selected_lap,
                profile.rows,
                profile.track_length,
            )
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
