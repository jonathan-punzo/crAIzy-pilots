import argparse
import bisect
import csv
import json
import math
import os
import statistics
import time

import snakeoil3_jm2 as snakeoil3


PORT = 3001
MAX_STEPS = 100000
DRIVER_VERSION = "craizy_auto_v4_stability_guard"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "torcs_ps4_dataset.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
AUTO_RESULTS_PATH = os.path.join(BASE_DIR, "auto_v4_runs.csv")
TRACE_EVERY = 5

PROFILE_STEP_METERS = 5.0
DANGER_SECTOR_METERS = 20.0
MIN_COMPLETE_LAP_ROWS = 800
MIN_COMPLETE_LAP_DISTANCE = 3500.0
MIN_CLEAN_LAPS = 3
MAX_TRACK_LENGTH_DEVIATION = 10.0
OFFTRACK_CONFIRM_TICKS = 3
OFFTRACK_TRACK_POS = 1.0

STEER_POSITION_GAIN = 0.16
STEER_ANGLE_GAIN = 0.22
STEER_LATERAL_GAIN = 0.002
MAX_STEER_CORRECTION = 0.06
MIN_STEER_CORRECTION = 0.025
MIN_CORRECTION_SPEED_SCALE = 0.25
MIN_RELIABILITY_CORRECTION_FACTOR = 0.70
TRACKPOS_DEADBAND = 0.03
ANGLE_DEADBAND = 0.03
SPEEDY_DEADBAND = 1.5
FEEDBACK_RAMP_METERS = 100.0
RECOVERY_GRACE_METERS = 100.0

TRACKPOS_MAD_FULL_CONFIDENCE = 0.15
ANGLE_MAD_FULL_CONFIDENCE = 0.10
SPEED_MAD_FULL_CONFIDENCE = 20.0
SPEEDY_MAD_FULL_CONFIDENCE = 8.0

BRAKE_START_THRESHOLD = 0.10
BRAKE_END_THRESHOLD = 0.05
BRAKE_CONFIRM_SAMPLES = 5
BRAKE_MERGE_GAP_METERS = 10.0

RECOVERY_ENTER_LINE_ERROR = 0.35
RECOVERY_EXIT_LINE_ERROR = 0.30
RECOVERY_ENTER_ANGLE_ERROR = 0.50
RECOVERY_EXIT_ANGLE_ERROR = 0.25
RECOVERY_ENTER_LATERAL_ERROR = 15.0
RECOVERY_EXIT_LATERAL_ERROR = 8.0
RECOVERY_ENTER_TRACK_POS = 1.0
RECOVERY_EXIT_TRACK_POS = 0.95
RECOVERY_BLEND_IN = 0.015
RECOVERY_BLEND_IN_DANGER = 0.010
RECOVERY_BLEND_OUT = 0.025
RECOVERY_TARGET_SPEED = 70.0
RECOVERY_CONFIRM_TICKS = 5
RECOVERY_EXIT_CONFIRM_TICKS = 10
RECOVERY_EXTREME_TRACK_POS = 1.12
RECOVERY_EXTREME_ANGLE = 0.85
RECOVERY_MAX_STEER = 0.65
RECOVERY_STEER_RATE = 0.04
STUCK_SPEED = 5.0
STUCK_TICKS = 75

STABILITY_LINE_START = 0.08
STABILITY_LINE_FULL = 0.33
STABILITY_ANGLE_START = 0.06
STABILITY_ANGLE_FULL = 0.24
STABILITY_LATERAL_START = 3.0
STABILITY_LATERAL_FULL = 12.0
STABILITY_EDGE_MARGIN = 0.10
STABILITY_MAX_THROTTLE_CUT = 0.55

PROFILE_EXPORT_PATH = os.path.join(
    RESULTS_DIR,
    "corkscrew_v4_profile_5m.csv",
)
DANGER_EXPORT_PATH = os.path.join(
    RESULTS_DIR,
    "corkscrew_v4_danger_20m.csv",
)
BRAKING_EXPORT_PATH = os.path.join(
    RESULTS_DIR,
    "corkscrew_v4_braking_zones.csv",
)
LAPS_EXPORT_PATH = os.path.join(
    RESULTS_DIR,
    "corkscrew_v4_lap_comparison.csv",
)

DATASET_COLUMNS = [
    "run_id", "step", "curLapTime",
    "steer_intent", "accel_intent", "brake_intent",
    "steer_action", "accel_action", "brake_action", "gear_action",
    "speedX", "speedY", "speedZ",
    "wheelSpinVel", "z", "track", "trackPos", "angle",
    "rpm", "damage", "distFromStart",
]

NUMERIC_DATASET_FIELDS = (
    "step",
    "curLapTime",
    "steer_intent",
    "accel_intent",
    "brake_intent",
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

PROFILE_INTERPOLATED_FIELDS = (
    "steer_action",
    "accel_action",
    "brake_action",
    "best_speedX",
    "best_speedY",
    "best_angle",
    "best_trackPos",
    "median_speedX",
    "median_speedY",
    "median_angle",
    "median_trackPos",
    "speedX_mad",
    "speedY_mad",
    "angle_mad",
    "trackPos_mad",
    "track_consensus",
    "angle_consensus",
    "reliability",
    "target_speedX",
    "target_speedY",
    "target_angle",
    "target_trackPos",
    "danger_score",
)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def safe_float(value, default=0.0):
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def parse_float(value):
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise ValueError("Valore numerico non finito.")
    return value


def safe_list(value, length, default_value=0.0):
    if not isinstance(value, (list, tuple)):
        return [default_value] * length
    values = [safe_float(item, default_value) for item in value[:length]]
    if len(values) < length:
        values += [default_value] * (length - len(values))
    return values


def blend(first, second, amount):
    return first + (second - first) * clamp(amount, 0.0, 1.0)


def apply_deadband(value, threshold):
    magnitude = abs(value)
    if magnitude <= threshold:
        return 0.0
    return math.copysign(magnitude - threshold, value)


def median_absolute_deviation(values):
    if not values:
        return 0.0
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = clamp(fraction, 0.0, 1.0) * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    return blend(
        ordered[lower_index],
        ordered[upper_index],
        position - lower_index,
    )


def danger_level(score):
    if score < 0.25:
        return "low"
    if score < 0.50:
        return "medium"
    if score < 0.75:
        return "high"
    return "critical"


def circular_distance_gap(start, end, track_length):
    return (start - end) % track_length


def distance_in_interval(distance, start, end, wraps):
    if wraps:
        return distance >= start or distance <= end
    return start <= distance <= end


class CorkscrewProfile:
    def __init__(self, path=DATASET_PATH):
        self.path = path
        self.error = ""
        self.rows = 0
        self.laps = 0
        self.track_length = 0.0
        self.selected_run_id = ""
        self.selected_lap_time = 0.0
        self.clean_laps = []
        self.lap_comparison = []
        self.launch = []
        self.launch_times = []
        self.launch_duration = 0.0
        self.profile = []
        self.profile_distances = []
        self.danger_map = []
        self.braking_zones = []
        self.load()

    @property
    def available(self):
        return bool(self.profile)

    @staticmethod
    def _group_runs(rows):
        groups = {}
        order = []
        for row in rows:
            run_id = str(row.get("run_id", "")).strip()
            if not run_id:
                continue
            if run_id not in groups:
                groups[run_id] = []
                order.append(run_id)
            groups[run_id].append(row)
        return [(run_id, groups[run_id]) for run_id in order]

    @staticmethod
    def _prepare_run(run_id, raw_rows):
        prepared = []
        reason = ""
        try:
            for raw in raw_rows:
                row = {"run_id": run_id}
                for field in NUMERIC_DATASET_FIELDS:
                    if field not in raw or raw[field] == "":
                        raise ValueError("Campo mancante: %s" % field)
                    row[field] = parse_float(raw[field])

                track = json.loads(raw.get("track", ""))
                wheel_spin = json.loads(raw.get("wheelSpinVel", ""))
                if not isinstance(track, list) or len(track) != 19:
                    raise ValueError("Sensori track non validi.")
                if not isinstance(wheel_spin, list) or len(wheel_spin) != 4:
                    raise ValueError("Sensori ruota non validi.")
                row["track"] = [parse_float(value) for value in track]
                row["wheelSpinVel"] = [
                    parse_float(value) for value in wheel_spin
                ]

                if not -1.0 <= row["steer_action"] <= 1.0:
                    raise ValueError("steer_action fuori range.")
                for field in (
                    "accel_action",
                    "brake_action",
                    "accel_intent",
                    "brake_intent",
                ):
                    if not 0.0 <= row[field] <= 1.0:
                        raise ValueError("%s fuori range." % field)
                if not -1.0 <= row["steer_intent"] <= 1.0:
                    raise ValueError("steer_intent fuori range.")
                gear = int(row["gear_action"])
                if row["gear_action"] != gear or gear < 1 or gear > 6:
                    raise ValueError("gear_action fuori range.")
                row["gear_action"] = gear
                prepared.append(row)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            reason = str(error)
            return None, reason

        prepared.sort(key=lambda row: row["step"])
        if len(prepared) < MIN_COMPLETE_LAP_ROWS:
            return None, "Meno di %d campioni." % MIN_COMPLETE_LAP_ROWS

        wrap_index = None
        for index in range(1, len(prepared)):
            previous_distance = prepared[index - 1]["distFromStart"]
            current_distance = prepared[index]["distFromStart"]
            if previous_distance > 3000.0 and current_distance < 500.0:
                wrap_index = index
                break

        if wrap_index is None:
            launch_rows = []
            racing_rows = prepared
        else:
            launch_rows = prepared[:wrap_index]
            racing_rows = prepared[wrap_index:]

        distances = [row["distFromStart"] for row in prepared]
        length = max(distances)
        if min(distances) >= 80.0 or length < MIN_COMPLETE_LAP_DISTANCE:
            return None, "Giro non completo."
        if len(racing_rows) < MIN_COMPLETE_LAP_ROWS:
            return None, "Porzione racing incompleta."
        if max(row["damage"] for row in prepared) > 0.0:
            return None, "Danno maggiore di zero."
        offtrack_ticks = 0
        max_offtrack_ticks = 0
        for row in prepared:
            offtrack = (
                abs(row["trackPos"]) >= OFFTRACK_TRACK_POS
                or min(row["track"]) < 0.0
            )
            offtrack_ticks = offtrack_ticks + 1 if offtrack else 0
            max_offtrack_ticks = max(max_offtrack_ticks, offtrack_ticks)
        if max_offtrack_ticks >= OFFTRACK_CONFIRM_TICKS:
            return None, (
                "Uscita pista confermata per %d tick consecutivi."
                % max_offtrack_ticks
            )

        previous = None
        for row in prepared:
            row["_deceleration"] = 0.0
            if previous is not None:
                delta_time = row["curLapTime"] - previous["curLapTime"]
                if 0.001 <= delta_time <= 1.0:
                    speed_delta = (
                        previous["speedX"] - row["speedX"]
                    ) / 3.6
                    row["_deceleration"] = max(
                        0.0,
                        speed_delta / delta_time,
                    )
            previous = row

        lap_times = [
            row["curLapTime"]
            for row in prepared
            if row["curLapTime"] > 0.0
        ]
        lap_time = max(lap_times) if lap_times else float("inf")
        return {
            "run_id": run_id,
            "rows": prepared,
            "launch_rows": launch_rows,
            "racing_rows": racing_rows,
            "wrap_index": wrap_index,
            "length": length,
            "lap_time": lap_time,
            "max_damage": max(row["damage"] for row in prepared),
            "max_abs_trackPos": max(
                abs(row["trackPos"]) for row in prepared
            ),
            "max_speed": max(row["speedX"] for row in prepared),
            "avg_speed": statistics.fmean(
                row["speedX"] for row in prepared
            ),
        }, reason

    @staticmethod
    def _deduplicate_by_distance(rows):
        ordered = sorted(rows, key=lambda row: row["distFromStart"])
        result = []
        for row in ordered:
            if (
                result
                and abs(
                    result[-1]["distFromStart"] - row["distFromStart"]
                ) < 0.000001
            ):
                result[-1] = row
            else:
                result.append(row)
        return result

    @staticmethod
    def _sample_lap(lap, distance, track_length):
        rows = lap["distance_rows"]
        distances = lap["distances"]
        distance = clamp(distance, 0.0, track_length)
        upper_index = bisect.bisect_left(distances, distance)
        if (
            upper_index < len(rows)
            and abs(distances[upper_index] - distance) < 0.000001
        ):
            exact = rows[upper_index]
            return {
                "steer_action": exact["steer_action"],
                "accel_action": exact["accel_action"],
                "brake_action": exact["brake_action"],
                "gear_action": exact["gear_action"],
                "speedX": exact["speedX"],
                "speedY": exact["speedY"],
                "angle": exact["angle"],
                "trackPos": exact["trackPos"],
            }

        if upper_index == 0:
            first = rows[0]
            return {
                "steer_action": first["steer_action"],
                "accel_action": first["accel_action"],
                "brake_action": first["brake_action"],
                "gear_action": first["gear_action"],
                "speedX": first["speedX"],
                "speedY": first["speedY"],
                "angle": first["angle"],
                "trackPos": first["trackPos"],
            }
        elif upper_index >= len(rows):
            last = rows[-1]
            return {
                "steer_action": last["steer_action"],
                "accel_action": last["accel_action"],
                "brake_action": last["brake_action"],
                "gear_action": last["gear_action"],
                "speedX": last["speedX"],
                "speedY": last["speedY"],
                "angle": last["angle"],
                "trackPos": last["trackPos"],
            }
        else:
            lower = rows[upper_index - 1]
            upper = rows[upper_index]
            lower_distance = distances[upper_index - 1]
            upper_distance = distances[upper_index]

        span = upper_distance - lower_distance
        fraction = (
            (distance - lower_distance) / span
            if span > 0.000001
            else 0.0
        )
        fields = (
            "steer_action",
            "accel_action",
            "brake_action",
            "speedX",
            "speedY",
            "angle",
            "trackPos",
        )
        result = {
            field: blend(lower[field], upper[field], fraction)
            for field in fields
        }
        result["gear_action"] = (
            lower["gear_action"]
            if fraction < 0.5
            else upper["gear_action"]
        )
        return result

    @staticmethod
    def _launch_sample(row, launch_index):
        return {
            "bin_index": -1,
            "launch_index": launch_index,
            "distance": row["distFromStart"],
            "steer_action": row["steer_action"],
            "accel_action": row["accel_action"],
            "brake_action": row["brake_action"],
            "gear_action": row["gear_action"],
            "best_speedX": row["speedX"],
            "best_speedY": row["speedY"],
            "best_angle": row["angle"],
            "best_trackPos": row["trackPos"],
            "median_speedX": row["speedX"],
            "median_speedY": row["speedY"],
            "median_angle": row["angle"],
            "median_trackPos": row["trackPos"],
            "speedX_mad": 0.0,
            "speedY_mad": 0.0,
            "angle_mad": 0.0,
            "trackPos_mad": 0.0,
            "track_consensus": 1.0,
            "angle_consensus": 1.0,
            "reliability": 1.0,
            "target_speedX": row["speedX"],
            "target_speedY": row["speedY"],
            "target_angle": row["angle"],
            "target_trackPos": row["trackPos"],
            "danger_sector": -1,
            "danger_score": 0.0,
            "danger_level": "low",
        }

    def _build_launch(self, best_lap):
        rows = best_lap["launch_rows"]
        if not rows:
            self.launch = []
            self.launch_times = []
            self.launch_duration = 0.0
            return

        start_time = rows[0]["curLapTime"]
        self.launch = []
        for index, row in enumerate(rows):
            sample = self._launch_sample(row, index)
            sample["launch_time"] = max(
                0.0,
                row["curLapTime"] - start_time,
            )
            self.launch.append(sample)
        self.launch_times = [
            sample["launch_time"] for sample in self.launch
        ]
        self.launch_duration = self.launch_times[-1]

    def _build_profile(self):
        count = int(math.ceil(self.track_length / PROFILE_STEP_METERS))
        grid = [
            index * PROFILE_STEP_METERS
            for index in range(count)
            if index * PROFILE_STEP_METERS < self.track_length
        ]
        best_lap = min(
            self.clean_laps,
            key=lambda lap: (lap["lap_time"], -len(lap["rows"])),
        )
        self.selected_run_id = best_lap["run_id"]
        self.selected_lap_time = best_lap["lap_time"]
        self._build_launch(best_lap)

        profile = []
        for index, distance in enumerate(grid):
            samples = [
                self._sample_lap(lap, distance, self.track_length)
                for lap in self.clean_laps
            ]
            best = self._sample_lap(
                best_lap,
                distance,
                self.track_length,
            )

            values = {
                field: [sample[field] for sample in samples]
                for field in ("speedX", "speedY", "angle", "trackPos")
            }
            medians = {
                field: statistics.median(field_values)
                for field, field_values in values.items()
            }
            mads = {
                field: median_absolute_deviation(field_values)
                for field, field_values in values.items()
            }

            track_consensus = clamp(
                1.0 - mads["trackPos"] / TRACKPOS_MAD_FULL_CONFIDENCE,
                0.0,
                1.0,
            )
            angle_consensus = clamp(
                1.0 - mads["angle"] / ANGLE_MAD_FULL_CONFIDENCE,
                0.0,
                1.0,
            )
            speed_consensus = clamp(
                1.0 - mads["speedX"] / SPEED_MAD_FULL_CONFIDENCE,
                0.0,
                1.0,
            )
            speedy_consensus = clamp(
                1.0 - mads["speedY"] / SPEEDY_MAD_FULL_CONFIDENCE,
                0.0,
                1.0,
            )
            reliability = statistics.fmean((
                track_consensus,
                angle_consensus,
                speed_consensus,
                speedy_consensus,
            ))

            profile.append({
                "bin_index": index,
                "distance": distance,
                "steer_action": best["steer_action"],
                "accel_action": best["accel_action"],
                "brake_action": best["brake_action"],
                "gear_action": best["gear_action"],
                "best_speedX": best["speedX"],
                "best_speedY": best["speedY"],
                "best_angle": best["angle"],
                "best_trackPos": best["trackPos"],
                "median_speedX": medians["speedX"],
                "median_speedY": medians["speedY"],
                "median_angle": medians["angle"],
                "median_trackPos": medians["trackPos"],
                "speedX_mad": mads["speedX"],
                "speedY_mad": mads["speedY"],
                "angle_mad": mads["angle"],
                "trackPos_mad": mads["trackPos"],
                "track_consensus": track_consensus,
                "angle_consensus": angle_consensus,
                "reliability": reliability,
                "target_speedX": blend(
                    best["speedX"],
                    medians["speedX"],
                    speed_consensus,
                ),
                "target_speedY": blend(
                    best["speedY"],
                    medians["speedY"],
                    speedy_consensus,
                ),
                "target_angle": blend(
                    best["angle"],
                    medians["angle"],
                    angle_consensus,
                ),
                "target_trackPos": blend(
                    best["trackPos"],
                    medians["trackPos"],
                    track_consensus,
                ),
                "danger_sector": 0,
                "danger_score": 0.0,
                "danger_level": "low",
            })

        self.profile = profile
        self.profile_distances = [
            sample["distance"] for sample in profile
        ]
        self.rows = len(profile)

    @staticmethod
    def _detected_braking_zones(lap, track_length):
        rows = lap.get("racing_rows", lap["rows"])
        zones = []
        active_start = None
        above_count = 0
        below_count = 0

        def profile_distance(row):
            return row.get("_profile_distance", row["distFromStart"])

        for index, row in enumerate(rows):
            brake = row["brake_action"]
            if active_start is None:
                above_count = (
                    above_count + 1
                    if brake > BRAKE_START_THRESHOLD
                    else 0
                )
                if above_count >= BRAKE_CONFIRM_SAMPLES:
                    active_start = index - BRAKE_CONFIRM_SAMPLES + 1
                    below_count = 0
            else:
                below_count = (
                    below_count + 1
                    if brake < BRAKE_END_THRESHOLD
                    else 0
                )
                if below_count >= BRAKE_CONFIRM_SAMPLES:
                    end_index = index - BRAKE_CONFIRM_SAMPLES
                    zone_rows = rows[active_start:end_index + 1]
                    if zone_rows:
                        zones.append({
                            "run_id": lap["run_id"],
                            "rows": zone_rows,
                            "start": profile_distance(zone_rows[0]),
                            "end": profile_distance(zone_rows[-1]),
                            "wraps": (
                                profile_distance(zone_rows[-1])
                                < profile_distance(zone_rows[0])
                            ),
                        })
                    active_start = None
                    above_count = 0
                    below_count = 0

        if active_start is not None:
            zone_rows = rows[active_start:]
            zones.append({
                "run_id": lap["run_id"],
                "rows": zone_rows,
                "start": profile_distance(zone_rows[0]),
                "end": profile_distance(zone_rows[-1]),
                "wraps": (
                    profile_distance(zone_rows[-1])
                    < profile_distance(zone_rows[0])
                ),
            })

        merged = []
        for zone in zones:
            if not merged:
                merged.append(zone)
                continue
            previous = merged[-1]
            gap = circular_distance_gap(
                zone["start"],
                previous["end"],
                track_length,
            )
            if gap < BRAKE_MERGE_GAP_METERS:
                previous["rows"].extend(zone["rows"])
                previous["end"] = zone["end"]
                previous["wraps"] = (
                    previous["wraps"]
                    or zone["wraps"]
                    or previous["end"] < previous["start"]
                )
            else:
                merged.append(zone)

        if len(merged) > 1:
            gap = circular_distance_gap(
                merged[0]["start"],
                merged[-1]["end"],
                track_length,
            )
            if gap < BRAKE_MERGE_GAP_METERS:
                last = merged.pop()
                first = merged.pop(0)
                combined = {
                    "run_id": lap["run_id"],
                    "rows": last["rows"] + first["rows"],
                    "start": last["start"],
                    "end": first["end"],
                    "wraps": True,
                }
                merged.insert(0, combined)
        return merged

    @staticmethod
    def _circular_groups(active):
        groups = []
        current = []
        for index, is_active in enumerate(active):
            if is_active:
                current.append(index)
            elif current:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        if (
            len(groups) > 1
            and groups[0][0] == 0
            and groups[-1][-1] == len(active) - 1
        ):
            groups[0] = groups[-1] + groups[0]
            groups.pop()
        return groups

    def _build_braking_map(self):
        detected = []
        for lap in self.clean_laps:
            detected.extend(
                self._detected_braking_zones(lap, self.track_length)
            )

        support = []
        for sample in self.profile:
            distance = sample["distance"]
            laps_supporting = {
                zone["run_id"]
                for zone in detected
                if distance_in_interval(
                    distance,
                    zone["start"],
                    zone["end"],
                    zone["wraps"],
                )
            }
            support.append(len(laps_supporting))

        zones = []
        for zone_index, indexes in enumerate(
            self._circular_groups([value > 0 for value in support])
        ):
            start = self.profile[indexes[0]]["distance"]
            end = (
                self.profile[indexes[-1]]["distance"]
                + PROFILE_STEP_METERS
            ) % self.track_length
            wraps = indexes[0] > indexes[-1] or end < start
            contributing = [
                zone
                for zone in detected
                if any(
                    distance_in_interval(
                        self.profile[index]["distance"],
                        zone["start"],
                        zone["end"],
                        zone["wraps"],
                    )
                    for index in indexes
                )
            ]
            samples = [
                row
                for zone in contributing
                for row in zone["rows"]
            ]
            if not samples:
                continue
            start_speeds = [
                zone["rows"][0]["speedX"] for zone in contributing
            ]
            minimum_speeds = [
                min(row["speedX"] for row in zone["rows"])
                for zone in contributing
            ]
            reductions = [
                max(0.0, start_speed - minimum_speed)
                for start_speed, minimum_speed in zip(
                    start_speeds,
                    minimum_speeds,
                )
            ]
            zones.append({
                "zone_id": zone_index,
                "start_distance": start,
                "end_distance": end,
                "wraps_finish": int(wraps),
                "support_laps": max(support[index] for index in indexes),
                "support_ratio": statistics.fmean(
                    support[index] / self.laps for index in indexes
                ),
                "initial_speed": statistics.median(start_speeds),
                "minimum_speed": statistics.median(minimum_speeds),
                "maximum_brake": max(
                    row["brake_action"] for row in samples
                ),
                "mean_brake": statistics.fmean(
                    row["brake_action"] for row in samples
                ),
                "speed_reduction": statistics.median(reductions),
            })
        self.braking_zones = zones

    def _build_danger_map(self):
        sector_count = int(
            math.ceil(self.track_length / DANGER_SECTOR_METERS)
        )
        sector_rows = [[] for _ in range(sector_count)]
        for lap in self.clean_laps:
            for row in lap.get("racing_rows", lap["rows"]):
                sector = min(
                    int(
                        (row["_profile_distance"] % self.track_length)
                        // DANGER_SECTOR_METERS
                    ),
                    sector_count - 1,
                )
                sector_rows[sector].append(row)

        danger_map = []
        for index, rows in enumerate(sector_rows):
            brake = clamp(
                percentile(
                    [row["brake_action"] for row in rows],
                    0.95,
                ),
                0.0,
                1.0,
            )
            steer = clamp(
                percentile(
                    [abs(row["steer_action"]) for row in rows],
                    0.95,
                ) / 0.8,
                0.0,
                1.0,
            )
            lateral = clamp(
                percentile(
                    [abs(row["speedY"]) for row in rows],
                    0.95,
                ) / 25.0,
                0.0,
                1.0,
            )
            angle = clamp(
                percentile(
                    [abs(row["angle"]) for row in rows],
                    0.95,
                ) / 0.50,
                0.0,
                1.0,
            )
            edge = clamp(
                percentile(
                    [
                        clamp(
                            (abs(row["trackPos"]) - 0.55) / 0.40,
                            0.0,
                            1.0,
                        )
                        for row in rows
                    ],
                    0.95,
                ),
                0.0,
                1.0,
            )
            deceleration = clamp(
                percentile(
                    [row["_deceleration"] for row in rows],
                    0.95,
                ) / 8.0,
                0.0,
                1.0,
            )
            score = clamp(
                0.25 * brake
                + 0.20 * steer
                + 0.15 * lateral
                + 0.15 * angle
                + 0.15 * edge
                + 0.10 * deceleration,
                0.0,
                1.0,
            )
            danger_map.append({
                "sector_index": index,
                "start_distance": index * DANGER_SECTOR_METERS,
                "end_distance": min(
                    (index + 1) * DANGER_SECTOR_METERS,
                    self.track_length,
                ),
                "sample_count": len(rows),
                "brake_component": brake,
                "steer_component": steer,
                "speedY_component": lateral,
                "angle_component": angle,
                "edge_component": edge,
                "deceleration_component": deceleration,
                "danger_score": score,
                "danger_level": danger_level(score),
            })

        self.danger_map = danger_map
        for sample in self.profile:
            sector = min(
                int(sample["distance"] // DANGER_SECTOR_METERS),
                len(danger_map) - 1,
            )
            danger = danger_map[sector]
            sample["danger_sector"] = sector
            sample["danger_score"] = danger["danger_score"]
            sample["danger_level"] = danger["danger_level"]

    def load(self):
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            self.error = "Dataset post-ADAS assente: registra almeno tre giri."
            return

        try:
            with open(self.path, newline="", encoding="utf-8-sig") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != DATASET_COLUMNS:
                    self.error = (
                        "Schema legacy rifiutato: V4 richiede intent e action "
                        "post-ADAS."
                    )
                    return
                raw_rows = list(reader)
        except (OSError, csv.Error) as error:
            self.error = "Impossibile leggere il dataset: %s" % error
            return

        candidates = []
        comparison = []
        for run_id, rows in self._group_runs(raw_rows):
            lap, reason = self._prepare_run(run_id, rows)
            if lap is None:
                comparison.append({
                    "run_id": run_id,
                    "accepted": 0,
                    "selected": 0,
                    "reason": reason,
                    "rows": len(rows),
                    "lap_time": "",
                    "track_length": "",
                    "max_damage": "",
                    "max_abs_trackPos": "",
                    "avg_speed": "",
                    "max_speed": "",
                })
                continue
            candidates.append(lap)
            comparison.append({
                "run_id": run_id,
                "accepted": 1,
                "selected": 0,
                "reason": "",
                "rows": len(lap["rows"]),
                "lap_time": lap["lap_time"],
                "track_length": lap["length"],
                "max_damage": lap["max_damage"],
                "max_abs_trackPos": lap["max_abs_trackPos"],
                "avg_speed": lap["avg_speed"],
                "max_speed": lap["max_speed"],
            })

        if len(candidates) < MIN_CLEAN_LAPS:
            self.lap_comparison = comparison
            self.error = (
                "Servono almeno %d giri post-ADAS puliti; trovati: %d."
                % (MIN_CLEAN_LAPS, len(candidates))
            )
            return

        median_length = statistics.median(
            lap["length"] for lap in candidates
        )
        clean_laps = []
        rejected_ids = set()
        for lap in candidates:
            if abs(lap["length"] - median_length) > MAX_TRACK_LENGTH_DEVIATION:
                rejected_ids.add(lap["run_id"])
            else:
                clean_laps.append(lap)

        for row in comparison:
            if row["run_id"] in rejected_ids:
                row["accepted"] = 0
                row["reason"] = "Lunghezza oltre 10 m dalla mediana."

        if len(clean_laps) < MIN_CLEAN_LAPS:
            self.lap_comparison = comparison
            self.error = (
                "Dopo il controllo lunghezza restano %d giri; ne servono %d."
                % (len(clean_laps), MIN_CLEAN_LAPS)
            )
            return

        self.clean_laps = clean_laps
        self.laps = len(clean_laps)
        self.track_length = statistics.median(
            lap["length"] for lap in clean_laps
        )
        for lap in clean_laps:
            distance_scale = self.track_length / lap["length"]
            for row in lap["racing_rows"]:
                row["_profile_distance"] = clamp(
                    row["distFromStart"] * distance_scale,
                    0.0,
                    self.track_length,
                )
            scaled_rows = [
                {
                    **row,
                    "distFromStart": row["_profile_distance"],
                }
                for row in lap["racing_rows"]
            ]
            lap["distance_rows"] = self._deduplicate_by_distance(
                scaled_rows
            )
            lap["distances"] = [
                row["distFromStart"] for row in lap["distance_rows"]
            ]
        self._build_profile()
        self._build_braking_map()
        self._build_danger_map()

        for row in comparison:
            row["selected"] = int(
                row["run_id"] == self.selected_run_id
            )
        self.lap_comparison = comparison

    def _interpolate_profile(self, lower, upper, fraction, distance):
        result = {
            field: blend(lower[field], upper[field], fraction)
            for field in PROFILE_INTERPOLATED_FIELDS
        }
        result["gear_action"] = (
            lower["gear_action"]
            if fraction < 0.5
            else upper["gear_action"]
        )
        nearest = lower if fraction < 0.5 else upper
        result["bin_index"] = nearest["bin_index"]
        result["distance"] = distance
        result["danger_sector"] = nearest["danger_sector"]
        result["danger_level"] = danger_level(result["danger_score"])
        return result

    def reference_at(self, distance):
        if not self.available:
            return None
        distance = clamp(
            safe_float(distance),
            self.profile_distances[0],
            self.profile_distances[-1],
        )
        upper_index = bisect.bisect_left(self.profile_distances, distance)

        if upper_index == 0:
            return self._interpolate_profile(
                self.profile[0],
                self.profile[0],
                0.0,
                distance,
            )
        elif upper_index >= len(self.profile):
            return self._interpolate_profile(
                self.profile[-1],
                self.profile[-1],
                0.0,
                distance,
            )
        else:
            lower = self.profile[upper_index - 1]
            upper = self.profile[upper_index]
            lower_distance = self.profile_distances[upper_index - 1]
            upper_distance = self.profile_distances[upper_index]

        span = upper_distance - lower_distance
        fraction = (
            (distance - lower_distance) / span
            if span > 0.000001
            else 0.0
        )
        return self._interpolate_profile(
            lower,
            upper,
            fraction,
            distance,
        )

    def launch_reference_at(self, elapsed):
        if not self.launch:
            return None
        elapsed = clamp(
            safe_float(elapsed),
            self.launch_times[0],
            self.launch_times[-1],
        )
        upper_index = bisect.bisect_left(self.launch_times, elapsed)
        if upper_index == 0:
            return self.launch[0].copy()
        if upper_index >= len(self.launch):
            return self.launch[-1].copy()

        lower = self.launch[upper_index - 1]
        upper = self.launch[upper_index]
        lower_time = self.launch_times[upper_index - 1]
        upper_time = self.launch_times[upper_index]
        span = upper_time - lower_time
        fraction = (
            (elapsed - lower_time) / span
            if span > 0.000001
            else 0.0
        )
        result = {
            field: blend(lower[field], upper[field], fraction)
            for field in PROFILE_INTERPOLATED_FIELDS
        }
        result["gear_action"] = (
            lower["gear_action"]
            if fraction < 0.5
            else upper["gear_action"]
        )
        nearest = lower if fraction < 0.5 else upper
        result.update({
            "bin_index": -1,
            "launch_index": nearest["launch_index"],
            "launch_time": elapsed,
            "distance": blend(
                lower["distance"],
                upper["distance"],
                fraction,
            ),
            "danger_sector": -1,
            "danger_level": "low",
        })
        return result

    def export_reports(self, results_dir=RESULTS_DIR):
        os.makedirs(results_dir, exist_ok=True)
        paths = {
            "profile": os.path.join(
                results_dir,
                os.path.basename(PROFILE_EXPORT_PATH),
            ),
            "danger": os.path.join(
                results_dir,
                os.path.basename(DANGER_EXPORT_PATH),
            ),
            "braking": os.path.join(
                results_dir,
                os.path.basename(BRAKING_EXPORT_PATH),
            ),
            "laps": os.path.join(
                results_dir,
                os.path.basename(LAPS_EXPORT_PATH),
            ),
        }

        profile_rows = []
        for sample in self.profile:
            profile_rows.append({
                **sample,
                "distance_norm": sample["distance"] / self.track_length,
                "trackPos_norm": sample["target_trackPos"],
                "angle_norm": sample["target_angle"] / math.pi,
                "speedX_norm": sample["target_speedX"] / 300.0,
                "speedY_norm": sample["target_speedY"] / 100.0,
                "steer_norm": sample["steer_action"],
                "accel_norm": sample["accel_action"],
                "brake_norm": sample["brake_action"],
            })
        self._write_rows(paths["profile"], profile_rows)
        self._write_rows(paths["danger"], self.danger_map)
        self._write_rows(paths["braking"], self.braking_zones)
        self._write_rows(paths["laps"], self.lap_comparison)
        return paths

    @staticmethod
    def _write_rows(path, rows):
        if not rows:
            with open(path, "w", encoding="utf-8"):
                return
        with open(path, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


class PostAdasReplayPolicy:
    def __init__(self, profile):
        self.profile = profile
        self.phase = "launch" if profile.launch else "racing"
        self.previous_distance = None
        self.crossings = 0
        self.launch_start_time = None
        self.recovery_active = False
        self.recovery_blend = 0.0
        self.recovery_confirmation_ticks = 0
        self.recovery_exit_ticks = 0
        self.offtrack_confirmation_ticks = 0
        self.previous_recovery_steer = None
        self.stuck_ticks = 0
        self.stuck_active = False

    @staticmethod
    def _gear_for_speed(speed):
        if speed < 40.0:
            return 1
        if speed < 80.0:
            return 2
        if speed < 125.0:
            return 3
        if speed < 170.0:
            return 4
        if speed < 215.0:
            return 5
        return 6

    @staticmethod
    def _recovery_steer_target(angle, track_pos):
        return clamp(
            angle * 1.25 - track_pos * 0.70,
            -RECOVERY_MAX_STEER,
            RECOVERY_MAX_STEER,
        )

    def _recovery_action(self, sensors, replay_steer, stuck):
        speed = safe_float(sensors.get("speedX"))
        angle = safe_float(sensors.get("angle"))
        track_pos = safe_float(sensors.get("trackPos"))
        speed_y = safe_float(sensors.get("speedY"))
        desired_steer = self._recovery_steer_target(angle, track_pos)
        if self.previous_recovery_steer is None:
            self.previous_recovery_steer = replay_steer
        steer_delta = clamp(
            desired_steer - self.previous_recovery_steer,
            -RECOVERY_STEER_RATE,
            RECOVERY_STEER_RATE,
        )
        steer = clamp(
            self.previous_recovery_steer + steer_delta,
            -RECOVERY_MAX_STEER,
            RECOVERY_MAX_STEER,
        )
        self.previous_recovery_steer = steer

        if stuck:
            accel = 0.40
            brake = 0.0
            gear = -1
        elif speed > 140.0:
            accel = 0.0
            brake = 0.0
            gear = self._gear_for_speed(speed)
        elif speed > 90.0:
            accel = 0.0
            brake = (
                0.08
                if abs(angle) < 0.35 and abs(speed_y) < 12.0
                else 0.0
            )
            gear = self._gear_for_speed(speed)
        elif speed > RECOVERY_TARGET_SPEED + 10.0:
            accel = 0.0
            brake = 0.10
            gear = self._gear_for_speed(speed)
        elif speed < RECOVERY_TARGET_SPEED - 10.0:
            accel = 0.35
            brake = 0.0
            gear = self._gear_for_speed(speed)
        else:
            accel = 0.15
            brake = 0.0
            gear = self._gear_for_speed(speed)

        action = {
            "steer": steer,
            "accel": accel,
            "brake": brake,
            "gear": gear,
            "clutch": 0.0,
            "meta": 0,
        }
        return action, desired_steer

    def _select_reference(self, sensors):
        distance = safe_float(sensors.get("distFromStart"))
        transitioned = False
        if (
            self.phase == "launch"
            and self.previous_distance is not None
            and self.previous_distance > 3000.0
            and distance < 500.0
        ):
            self.phase = "racing"
            self.crossings += 1
            self.recovery_active = False
            self.recovery_blend = 0.0
            self.recovery_confirmation_ticks = 0
            self.recovery_exit_ticks = 0
            self.offtrack_confirmation_ticks = 0
            self.previous_recovery_steer = None
            transitioned = True

        self.previous_distance = distance
        if self.phase == "launch":
            current_time = safe_float(sensors.get("curLapTime"))
            if self.launch_start_time is None:
                self.launch_start_time = current_time
            elapsed = max(0.0, current_time - self.launch_start_time)
            return self.profile.launch_reference_at(elapsed), transitioned
        return self.profile.reference_at(distance), transitioned

    @staticmethod
    def _speed_correction_limit(speed):
        speed = abs(speed)
        if speed <= 100.0:
            return MAX_STEER_CORRECTION
        if speed >= 200.0:
            return MIN_STEER_CORRECTION
        return blend(
            MAX_STEER_CORRECTION,
            MIN_STEER_CORRECTION,
            (speed - 100.0) / 100.0,
        )

    @staticmethod
    def _stability_score(
        line_error,
        angle_error,
        lateral_error,
        track_pos,
        demonstrated_track_pos,
    ):
        line_score = clamp(
            (abs(line_error) - STABILITY_LINE_START)
            / (STABILITY_LINE_FULL - STABILITY_LINE_START),
            0.0,
            1.0,
        )
        angle_score = clamp(
            (abs(angle_error) - STABILITY_ANGLE_START)
            / (STABILITY_ANGLE_FULL - STABILITY_ANGLE_START),
            0.0,
            1.0,
        )
        lateral_score = clamp(
            (abs(lateral_error) - STABILITY_LATERAL_START)
            / (STABILITY_LATERAL_FULL - STABILITY_LATERAL_START),
            0.0,
            1.0,
        )
        demonstrated_edge = min(
            0.98,
            abs(demonstrated_track_pos) + STABILITY_EDGE_MARGIN,
        )
        edge_score = clamp(
            (abs(track_pos) - demonstrated_edge)
            / max(1.0 - demonstrated_edge, 0.02),
            0.0,
            1.0,
        )
        return max(line_score, angle_score, lateral_score, edge_score)

    def action(self, sensors):
        reference, transitioned = self._select_reference(sensors)
        if reference is None:
            raise RuntimeError(self.profile.error or "Dataset non disponibile.")

        speed = safe_float(sensors.get("speedX"))
        track_pos = safe_float(sensors.get("trackPos"))
        angle = safe_float(sensors.get("angle"))
        speed_y = safe_float(sensors.get("speedY"))
        track = safe_list(sensors.get("track"), 19, 200.0)

        # Actions and feedback must describe the same demonstrated trajectory.
        # Multilap targets remain useful for analysis, but do not steer the car.
        control_track_pos = reference["best_trackPos"]
        control_angle = reference["best_angle"]
        control_speed_y = reference["best_speedY"]
        control_speed_x = reference["best_speedX"]
        line_error = track_pos - control_track_pos
        angle_error = (
            angle - control_angle
        )
        lateral_error = (
            speed_y - control_speed_y
        )
        danger = clamp(reference["danger_score"], 0.0, 1.0)
        reliability = clamp(reference["reliability"], 0.0, 1.0)

        if self.phase == "racing":
            feedback_ramp = clamp(
                safe_float(sensors.get("distFromStart"))
                / FEEDBACK_RAMP_METERS,
                0.0,
                1.0,
            )
        else:
            feedback_ramp = 0.0
        correction_limit = self._speed_correction_limit(speed) * blend(
            MIN_RELIABILITY_CORRECTION_FACTOR,
            1.0,
            reliability,
        ) * blend(1.0, 0.85, danger) * feedback_ramp
        speed_scale = clamp(
            abs(speed) / max(abs(control_speed_x), 1.0),
            MIN_CORRECTION_SPEED_SCALE,
            1.0,
        )
        corrected_line_error = apply_deadband(
            line_error,
            TRACKPOS_DEADBAND,
        )
        corrected_angle_error = apply_deadband(
            angle_error,
            ANGLE_DEADBAND,
        )
        corrected_lateral_error = apply_deadband(
            lateral_error,
            SPEEDY_DEADBAND,
        )
        raw_correction = (
            corrected_angle_error * STEER_ANGLE_GAIN
            - corrected_line_error * STEER_POSITION_GAIN
            - corrected_lateral_error * STEER_LATERAL_GAIN
        ) * speed_scale
        steering_correction = clamp(
            raw_correction,
            -correction_limit,
            correction_limit,
        )

        replay_action = {
            "steer": clamp(
                reference["steer_action"] + steering_correction,
                -1.0,
                1.0,
            ),
            "accel": clamp(reference["accel_action"], 0.0, 1.0),
            "brake": clamp(reference["brake_action"], 0.0, 1.0),
            "gear": reference["gear_action"],
            "clutch": 0.0,
            "meta": 0,
        }

        stability_score = 0.0
        throttle_cut = 0.0
        if self.phase == "racing":
            stability_score = self._stability_score(
                line_error,
                angle_error,
                lateral_error,
                track_pos,
                control_track_pos,
            )
            speed_guard = clamp((abs(speed) - 120.0) / 80.0, 0.0, 1.0)
            throttle_cut = (
                STABILITY_MAX_THROTTLE_CUT
                * stability_score
                * speed_guard
            )
            replay_action["accel"] *= 1.0 - throttle_cut

        enter_line = RECOVERY_ENTER_LINE_ERROR
        exit_line = RECOVERY_EXIT_LINE_ERROR
        enter_angle = RECOVERY_ENTER_ANGLE_ERROR
        exit_angle = RECOVERY_EXIT_ANGLE_ERROR
        enter_lateral = RECOVERY_ENTER_LATERAL_ERROR
        exit_lateral = RECOVERY_EXIT_LATERAL_ERROR
        enter_track = RECOVERY_ENTER_TRACK_POS
        exit_track = RECOVERY_EXIT_TRACK_POS
        recovery_blend_in = (
            RECOVERY_BLEND_IN
            + RECOVERY_BLEND_IN_DANGER * danger
        )

        offtrack_now = (
            abs(track_pos) > OFFTRACK_TRACK_POS
            or min(track) < 0.0
        )
        if self.phase == "racing" and offtrack_now:
            self.offtrack_confirmation_ticks += 1
        else:
            self.offtrack_confirmation_ticks = 0

        severe_dynamics = (
            abs(angle_error) > enter_angle
            and abs(lateral_error) > enter_lateral
        )
        if self.phase == "racing" and severe_dynamics:
            self.recovery_confirmation_ticks += 1
        else:
            self.recovery_confirmation_ticks = 0

        if (
            self.phase == "racing"
            and not self.stuck_active
            and abs(speed) < STUCK_SPEED
        ):
            self.stuck_ticks += 1
        elif not self.stuck_active:
            self.stuck_ticks = 0
        if self.stuck_ticks >= STUCK_TICKS:
            self.stuck_active = True
        if self.stuck_active and speed < -8.0:
            self.stuck_active = False
            self.stuck_ticks = 0
        stuck = self.stuck_active

        extreme_track = abs(track_pos) > RECOVERY_EXTREME_TRACK_POS
        extreme_angle = abs(angle) > RECOVERY_EXTREME_ANGLE
        normal_recovery_allowed = (
            self.phase == "racing"
            and safe_float(sensors.get("distFromStart"))
            >= RECOVERY_GRACE_METERS
        )

        recovery_cause = ""
        if stuck:
            recovery_cause = "stuck"
        elif extreme_track:
            recovery_cause = "extreme_track"
        elif extreme_angle:
            recovery_cause = "extreme_angle"
        elif (
            normal_recovery_allowed
            and self.offtrack_confirmation_ticks >= OFFTRACK_CONFIRM_TICKS
        ):
            recovery_cause = "offtrack"
        elif (
            normal_recovery_allowed
            and self.recovery_confirmation_ticks >= RECOVERY_CONFIRM_TICKS
        ):
            recovery_cause = "severe_dynamics"

        enter_recovery = bool(recovery_cause)
        exit_recovery = (
            abs(line_error) < exit_line
            and abs(angle_error) < exit_angle
            and abs(lateral_error) < exit_lateral
            and abs(track_pos) < exit_track
            and not offtrack_now
            and not stuck
        )
        if enter_recovery:
            self.recovery_active = True
            self.recovery_exit_ticks = 0
        elif self.recovery_active:
            if exit_recovery:
                self.recovery_exit_ticks += 1
            else:
                self.recovery_exit_ticks = 0
            if self.recovery_exit_ticks >= RECOVERY_EXIT_CONFIRM_TICKS:
                self.recovery_active = False
                self.recovery_exit_ticks = 0

        target_blend = 1.0 if self.recovery_active else 0.0
        if target_blend > self.recovery_blend:
            if stuck or extreme_track or extreme_angle:
                recovery_blend_in = max(recovery_blend_in, 0.04)
            self.recovery_blend = min(
                target_blend,
                self.recovery_blend + recovery_blend_in,
            )
        else:
            self.recovery_blend = max(
                target_blend,
                self.recovery_blend - RECOVERY_BLEND_OUT,
            )

        if not self.recovery_active and self.recovery_blend <= 0.0:
            self.previous_recovery_steer = replay_action["steer"]
        recovery_action, recovery_steer_target = self._recovery_action(
            sensors,
            replay_action["steer"],
            stuck,
        )
        action = replay_action.copy()
        for key in ("steer", "accel", "brake"):
            action[key] = blend(
                replay_action[key],
                recovery_action[key],
                self.recovery_blend,
            )
        if self.recovery_blend > 0.0:
            action["gear"] = recovery_action["gear"]

        diagnostics = {
            "mode": (
                "recovery"
                if self.recovery_blend > 0.0
                else "stability"
                if stability_score > 0.0
                else "replay"
            ),
            "phase": self.phase,
            "phase_transition": int(transitioned),
            "launch_index": reference.get("launch_index", -1),
            "crossings": self.crossings,
            "recovery_blend": self.recovery_blend,
            "recovery_confirmation_ticks": self.recovery_confirmation_ticks,
            "recovery_exit_ticks": self.recovery_exit_ticks,
            "offtrack_confirmation_ticks": (
                self.offtrack_confirmation_ticks
            ),
            "recovery_cause": recovery_cause,
            "recovery_steer_target": recovery_steer_target,
            "recovery_steer_limited": recovery_action["steer"],
            "recovery_gear": recovery_action["gear"],
            "control_speedX": control_speed_x,
            "control_speedY": control_speed_y,
            "control_trackPos": control_track_pos,
            "control_angle": control_angle,
            "line_error": line_error,
            "angle_error": angle_error,
            "lateral_error": lateral_error,
            "corrected_line_error": corrected_line_error,
            "corrected_angle_error": corrected_angle_error,
            "corrected_lateral_error": corrected_lateral_error,
            "feedback_ramp": feedback_ramp,
            "steering_correction": steering_correction,
            "correction_limit": correction_limit,
            "stability_score": stability_score,
            "throttle_cut": throttle_cut,
            "enter_line": enter_line,
            "enter_angle": enter_angle,
            "enter_lateral": enter_lateral,
            "enter_track": enter_track,
            "recovery_blend_in": recovery_blend_in,
            "reference": reference,
            "replay_action": replay_action,
        }
        return action, diagnostics


class TraceLogger:
    FIELDS = [
        "step", "phase", "phase_transition", "launch_index", "crossings",
        "distFromStart", "profile_bin", "danger_sector",
        "speedX", "speedY", "trackPos", "angle", "target_speedX",
        "target_speedY", "target_trackPos", "target_angle", "trackPos_mad",
        "angle_mad", "control_speedX", "control_speedY",
        "control_trackPos", "control_angle", "reliability", "danger_score",
        "danger_level", "mode", "stability_score", "throttle_cut",
        "recovery_blend", "recovery_confirmation_ticks",
        "offtrack_confirmation_ticks", "recovery_exit_ticks",
        "recovery_cause", "recovery_gear",
        "recovery_steer_target", "recovery_steer_limited",
        "line_error", "angle_error", "lateral_error",
        "corrected_line_error", "corrected_angle_error",
        "corrected_lateral_error", "feedback_ramp",
        "correction_limit", "steering_correction", "enter_line",
        "enter_angle", "enter_lateral", "enter_track", "recovery_blend_in",
        "recorded_steer", "final_steer", "recorded_accel", "final_accel",
        "recorded_brake", "final_brake", "recorded_gear", "final_gear",
        "damage", "offtrack", "first_damage_event", "first_offtrack_event",
    ]

    def __init__(self, path=None):
        if path is None:
            os.makedirs(LOGS_DIR, exist_ok=True)
            timestamp = "%s_%06d" % (
                time.strftime("%Y%m%d_%H%M%S"),
                (time.time_ns() // 1000) % 1000000,
            )
            path = os.path.join(
                LOGS_DIR,
                "auto_v4_trace_%s.csv" % timestamp,
            )
        self.path = path
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.damage_seen = False
        self.offtrack_seen = False

    def write(self, step, sensors, action, diagnostics):
        if step % TRACE_EVERY != 0:
            return
        reference = diagnostics["reference"]
        damage = safe_float(sensors.get("damage"))
        offtrack = int(abs(safe_float(sensors.get("trackPos"))) > 1.0)
        first_damage_event = int(damage > 0.0 and not self.damage_seen)
        first_offtrack_event = int(bool(offtrack) and not self.offtrack_seen)
        self.damage_seen = self.damage_seen or damage > 0.0
        self.offtrack_seen = self.offtrack_seen or bool(offtrack)
        self.writer.writerow({
            "step": step,
            "phase": diagnostics["phase"],
            "phase_transition": diagnostics["phase_transition"],
            "launch_index": diagnostics["launch_index"],
            "crossings": diagnostics["crossings"],
            "distFromStart": safe_float(sensors.get("distFromStart")),
            "profile_bin": reference["bin_index"],
            "danger_sector": reference["danger_sector"],
            "speedX": safe_float(sensors.get("speedX")),
            "speedY": safe_float(sensors.get("speedY")),
            "trackPos": safe_float(sensors.get("trackPos")),
            "angle": safe_float(sensors.get("angle")),
            "target_speedX": reference["target_speedX"],
            "target_speedY": reference["target_speedY"],
            "target_trackPos": reference["target_trackPos"],
            "target_angle": reference["target_angle"],
            "trackPos_mad": reference["trackPos_mad"],
            "angle_mad": reference["angle_mad"],
            "control_speedX": diagnostics["control_speedX"],
            "control_speedY": diagnostics["control_speedY"],
            "control_trackPos": diagnostics["control_trackPos"],
            "control_angle": diagnostics["control_angle"],
            "reliability": reference["reliability"],
            "danger_score": reference["danger_score"],
            "danger_level": reference["danger_level"],
            "mode": diagnostics["mode"],
            "stability_score": diagnostics["stability_score"],
            "throttle_cut": diagnostics["throttle_cut"],
            "recovery_blend": diagnostics["recovery_blend"],
            "recovery_confirmation_ticks": diagnostics[
                "recovery_confirmation_ticks"
            ],
            "offtrack_confirmation_ticks": diagnostics[
                "offtrack_confirmation_ticks"
            ],
            "recovery_exit_ticks": diagnostics["recovery_exit_ticks"],
            "recovery_cause": diagnostics["recovery_cause"],
            "recovery_gear": diagnostics["recovery_gear"],
            "recovery_steer_target": diagnostics["recovery_steer_target"],
            "recovery_steer_limited": diagnostics[
                "recovery_steer_limited"
            ],
            "line_error": diagnostics["line_error"],
            "angle_error": diagnostics["angle_error"],
            "lateral_error": diagnostics["lateral_error"],
            "corrected_line_error": diagnostics["corrected_line_error"],
            "corrected_angle_error": diagnostics["corrected_angle_error"],
            "corrected_lateral_error": diagnostics[
                "corrected_lateral_error"
            ],
            "feedback_ramp": diagnostics["feedback_ramp"],
            "correction_limit": diagnostics["correction_limit"],
            "steering_correction": diagnostics["steering_correction"],
            "enter_line": diagnostics["enter_line"],
            "enter_angle": diagnostics["enter_angle"],
            "enter_lateral": diagnostics["enter_lateral"],
            "enter_track": diagnostics["enter_track"],
            "recovery_blend_in": diagnostics["recovery_blend_in"],
            "recorded_steer": reference["steer_action"],
            "final_steer": action["steer"],
            "recorded_accel": reference["accel_action"],
            "final_accel": action["accel"],
            "recorded_brake": reference["brake_action"],
            "final_brake": action["brake"],
            "recorded_gear": reference["gear_action"],
            "final_gear": action["gear"],
            "damage": damage,
            "offtrack": offtrack,
            "first_damage_event": first_damage_event,
            "first_offtrack_event": first_offtrack_event,
        })

    def close(self):
        self.file.close()


class RunSummary:
    FIELDS = [
        "timestamp", "driver_version", "selected_run_id",
        "selected_lap_time", "dataset_laps", "profile_points", "steps",
        "lap_time", "damage", "offtrack_steps", "stability_steps",
        "recovery_steps",
        "avg_speed", "max_speed", "avg_danger", "max_danger",
        "critical_steps", "critical_time_s", "recovery_low",
        "recovery_medium", "recovery_high", "recovery_critical",
        "recovery_low_time_s", "recovery_medium_time_s",
        "recovery_high_time_s", "recovery_critical_time_s",
        "avg_line_error", "max_line_error", "reason",
    ]

    def __init__(self, profile):
        self.profile = profile
        self.steps = 0
        self.offtrack_steps = 0
        self.stability_steps = 0
        self.recovery_steps = 0
        self.speed_sum = 0.0
        self.max_speed = 0.0
        self.danger_sum = 0.0
        self.max_danger = 0.0
        self.critical_steps = 0
        self.recovery_by_level = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0,
        }
        self.recovery_time_by_level = {
            "low": 0.0,
            "medium": 0.0,
            "high": 0.0,
            "critical": 0.0,
        }
        self.critical_time = 0.0
        self.previous_lap_time = None
        self.line_error_sum = 0.0
        self.max_line_error = 0.0
        self.final_sensors = {}

    def record(self, sensors, diagnostics):
        speed = safe_float(sensors.get("speedX"))
        track = safe_list(sensors.get("track"), 19, 200.0)
        danger = diagnostics["reference"]["danger_score"]
        level = diagnostics["reference"]["danger_level"]
        line_error = abs(diagnostics["line_error"])
        recovering = diagnostics["recovery_blend"] > 0.0
        stabilizing = diagnostics["mode"] == "stability"
        current_lap_time = safe_float(sensors.get("curLapTime"))
        delta_time = 0.0
        if self.previous_lap_time is not None:
            measured_delta = current_lap_time - self.previous_lap_time
            if 0.0 < measured_delta <= 1.0:
                delta_time = measured_delta
        self.previous_lap_time = current_lap_time

        self.steps += 1
        self.speed_sum += speed
        self.max_speed = max(self.max_speed, speed)
        self.danger_sum += danger
        self.max_danger = max(self.max_danger, danger)
        self.critical_steps += int(level == "critical")
        if level == "critical":
            self.critical_time += delta_time
        self.line_error_sum += line_error
        self.max_line_error = max(self.max_line_error, line_error)
        self.offtrack_steps += int(
            abs(safe_float(sensors.get("trackPos"))) > 1.0
            or min(track) < 0.0
        )
        self.stability_steps += int(stabilizing)
        self.recovery_steps += int(recovering)
        if recovering:
            self.recovery_by_level[level] += 1
            self.recovery_time_by_level[level] += delta_time
        self.final_sensors = sensors.copy()

    def write(self, reason):
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "driver_version": DRIVER_VERSION,
            "selected_run_id": self.profile.selected_run_id,
            "selected_lap_time": self.profile.selected_lap_time,
            "dataset_laps": self.profile.laps,
            "profile_points": len(self.profile.profile),
            "steps": self.steps,
            "lap_time": safe_float(self.final_sensors.get("lastLapTime")),
            "damage": safe_float(self.final_sensors.get("damage")),
            "offtrack_steps": self.offtrack_steps,
            "stability_steps": self.stability_steps,
            "recovery_steps": self.recovery_steps,
            "avg_speed": self.speed_sum / self.steps if self.steps else 0.0,
            "max_speed": self.max_speed,
            "avg_danger": (
                self.danger_sum / self.steps if self.steps else 0.0
            ),
            "max_danger": self.max_danger,
            "critical_steps": self.critical_steps,
            "critical_time_s": self.critical_time,
            "recovery_low": self.recovery_by_level["low"],
            "recovery_medium": self.recovery_by_level["medium"],
            "recovery_high": self.recovery_by_level["high"],
            "recovery_critical": self.recovery_by_level["critical"],
            "recovery_low_time_s": self.recovery_time_by_level["low"],
            "recovery_medium_time_s": (
                self.recovery_time_by_level["medium"]
            ),
            "recovery_high_time_s": self.recovery_time_by_level["high"],
            "recovery_critical_time_s": (
                self.recovery_time_by_level["critical"]
            ),
            "avg_line_error": (
                self.line_error_sum / self.steps if self.steps else 0.0
            ),
            "max_line_error": self.max_line_error,
            "reason": reason,
        }
        self._ensure_compatible_results_file()
        exists = (
            os.path.exists(AUTO_RESULTS_PATH)
            and os.path.getsize(AUTO_RESULTS_PATH) > 0
        )
        with open(AUTO_RESULTS_PATH, "a", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=self.FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    @classmethod
    def _ensure_compatible_results_file(cls):
        if not os.path.exists(AUTO_RESULTS_PATH):
            return
        try:
            with open(
                AUTO_RESULTS_PATH,
                newline="",
                encoding="utf-8",
            ) as source:
                fields = csv.DictReader(source).fieldnames
        except OSError:
            return
        if fields == cls.FIELDS:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archived = os.path.join(
            BASE_DIR,
            "auto_v4_runs_legacy_%s.csv" % timestamp,
        )
        os.replace(AUTO_RESULTS_PATH, archived)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="V4 post-ADAS con stability guard per TORCS Corkscrew.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Genera i quattro report senza collegarsi a TORCS.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    profile = CorkscrewProfile()
    if not profile.available:
        for lap in profile.lap_comparison:
            if not lap["accepted"]:
                print(
                    "[DATASET] Run %s rifiutata: %s"
                    % (lap["run_id"], lap["reason"])
                )
        raise SystemExit("[DATASET] %s" % profile.error)

    paths = profile.export_reports()
    print("[DRIVER] %s" % DRIVER_VERSION)
    print(
        "[DATASET] %d giri; migliore %s (%.3f s); "
        "%d punti su %.1f m."
        % (
            profile.laps,
            profile.selected_run_id,
            profile.selected_lap_time,
            len(profile.profile),
            profile.track_length,
        )
    )
    for label, path in paths.items():
        print("[REPORT:%s] %s" % (label.upper(), path))

    if arguments.analyze_only:
        print("[STOP] analyze_only")
        return

    policy = PostAdasReplayPolicy(profile)
    summary = RunSummary(profile)
    trace = TraceLogger()
    client = snakeoil3.Client(p=PORT)
    reason = "max_steps"

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

            action, diagnostics = policy.action(sensors)
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
