import csv
import json
import os
import tempfile
import time
import unittest

import craizy_auto_v3 as v3
import craizy_manual as manual


def dataset_row(
    run_id,
    index,
    count,
    lap_time,
    track_length=3600.0,
    line_offset=0.0,
    dangerous=False,
):
    distance = track_length * index / (count - 1)
    progress = distance / track_length
    in_danger = dangerous and 900.0 <= distance <= 1020.0
    speed = 120.0
    if in_danger:
        speed = 170.0 - (distance - 900.0) * 0.45
    return {
        "run_id": run_id,
        "step": index,
        "curLapTime": lap_time * progress,
        "steer_intent": 0.8 if in_danger else progress * 0.2,
        "accel_intent": 0.0 if in_danger else 0.8,
        "brake_intent": 0.8 if in_danger else 0.0,
        "steer_action": 0.7 if in_danger else progress * 0.2,
        "accel_action": 0.0 if in_danger else 0.65,
        "brake_action": 0.8 if in_danger else 0.0,
        "gear_action": 4,
        "speedX": speed,
        "speedY": 22.0 if in_danger else 1.5,
        "speedZ": 0.0,
        "wheelSpinVel": json.dumps([1.0, 1.0, 1.0, 1.0]),
        "z": 0.3,
        "track": json.dumps([200.0] * 19),
        "trackPos": (
            (0.88 if in_danger else 0.1) + line_offset
        ),
        "angle": 0.45 if in_danger else 0.02 + line_offset * 0.2,
        "rpm": 6000.0,
        "damage": 0.0,
        "distFromStart": distance,
    }


def write_dataset(path, runs):
    rows = []
    for run in runs:
        rows.extend(
            dataset_row(
                run["run_id"],
                index,
                run.get("count", 900),
                run["lap_time"],
                run.get("track_length", 3600.0),
                run.get("line_offset", 0.0),
                run.get("dangerous", False),
            )
            for index in range(run.get("count", 900))
        )
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=v3.DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class MultilapProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = os.path.join(self.temp_dir.name, "dataset.csv")
        write_dataset(
            self.dataset_path,
            [
                {
                    "run_id": "slow",
                    "lap_time": 100.0,
                    "line_offset": -0.05,
                    "dangerous": True,
                },
                {
                    "run_id": "fast",
                    "lap_time": 90.0,
                    "line_offset": 0.0,
                    "dangerous": True,
                },
                {
                    "run_id": "medium",
                    "lap_time": 95.0,
                    "line_offset": 0.05,
                    "dangerous": True,
                },
            ],
        )
        self.profile = v3.CorkscrewProfile(self.dataset_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_profile_uses_all_laps_and_selects_fastest(self):
        self.assertTrue(self.profile.available, self.profile.error)
        self.assertEqual(self.profile.laps, 3)
        self.assertEqual(self.profile.selected_run_id, "fast")
        self.assertEqual(len(self.profile.profile), 720)
        self.assertEqual(len(self.profile.danger_map), 180)

    def test_legacy_schema_is_rejected(self):
        path = os.path.join(self.temp_dir.name, "legacy.csv")
        with open(path, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["steer", "accel", "brake"])
            writer.writerow([0.0, 1.0, 0.0])
        profile = v3.CorkscrewProfile(path)
        self.assertFalse(profile.available)
        self.assertIn("legacy", profile.error.lower())

    def test_invalid_laps_are_reported_and_excluded(self):
        path = os.path.join(self.temp_dir.name, "invalid.csv")
        runs = [
            {"run_id": "one", "lap_time": 90.0},
            {"run_id": "two", "lap_time": 91.0},
            {"run_id": "three", "lap_time": 92.0},
            {"run_id": "long", "lap_time": 89.0, "track_length": 3620.0},
        ]
        write_dataset(path, runs)
        profile = v3.CorkscrewProfile(path)
        self.assertTrue(profile.available, profile.error)
        rejected = {
            row["run_id"]: row["reason"]
            for row in profile.lap_comparison
            if not row["accepted"]
        }
        self.assertIn("long", rejected)
        self.assertIn("10 m", rejected["long"])

    def test_incomplete_damaged_and_offtrack_runs_are_rejected(self):
        incomplete = [
            dataset_row("short", index, 100, 10.0)
            for index in range(100)
        ]
        _, reason = v3.CorkscrewProfile._prepare_run(
            "short",
            incomplete,
        )
        self.assertIn("campioni", reason)

        damaged = [
            dataset_row("damaged", index, 900, 90.0)
            for index in range(900)
        ]
        damaged[400]["damage"] = 1.0
        _, reason = v3.CorkscrewProfile._prepare_run(
            "damaged",
            damaged,
        )
        self.assertIn("Danno", reason)

        offtrack = [
            dataset_row("offtrack", index, 900, 90.0)
            for index in range(900)
        ]
        offtrack[400]["trackPos"] = 1.0
        _, reason = v3.CorkscrewProfile._prepare_run(
            "offtrack",
            offtrack,
        )
        self.assertIn("trackPos", reason)

        negative_track = [
            dataset_row("negative", index, 900, 90.0)
            for index in range(900)
        ]
        negative_track[400]["track"] = json.dumps(
            [-1.0] + [200.0] * 18
        )
        _, reason = v3.CorkscrewProfile._prepare_run(
            "negative",
            negative_track,
        )
        self.assertIn("sensori track", reason)

    def test_finish_line_wrap_is_circular(self):
        before = self.profile.reference_at(10.0)
        wrapped = self.profile.reference_at(
            self.profile.track_length + 10.0
        )
        for key in v3.PROFILE_INTERPOLATED_FIELDS:
            self.assertAlmostEqual(before[key], wrapped[key], places=9)

    def test_grid_preserves_best_lap_actions(self):
        sample = self.profile.profile[100]
        distance = sample["distance"]
        best_lap = next(
            lap
            for lap in self.profile.clean_laps
            if lap["run_id"] == "fast"
        )
        best = self.profile._sample_lap(
            best_lap,
            distance,
            self.profile.track_length,
        )
        self.assertAlmostEqual(
            sample["steer_action"],
            best["steer_action"],
            delta=1e-12,
        )
        self.assertAlmostEqual(
            sample["accel_action"],
            best["accel_action"],
            delta=1e-12,
        )
        self.assertAlmostEqual(
            sample["brake_action"],
            best["brake_action"],
            delta=1e-12,
        )

    def test_multilap_median_mad_and_racing_line(self):
        sample = self.profile.reference_at(500.0)
        self.assertAlmostEqual(sample["median_trackPos"], 0.1, places=3)
        self.assertAlmostEqual(sample["trackPos_mad"], 0.05, places=3)
        self.assertGreater(sample["track_consensus"], 0.6)
        self.assertLessEqual(
            abs(sample["target_trackPos"] - sample["median_trackPos"]),
            abs(sample["best_trackPos"] - sample["median_trackPos"]),
        )

    def test_zero_error_reproduces_recorded_actions(self):
        reference = self.profile.reference_at(1800.0)
        sensors = {
            "distFromStart": 1800.0,
            "speedX": reference["target_speedX"],
            "speedY": reference["target_speedY"],
            "trackPos": reference["target_trackPos"],
            "angle": reference["target_angle"],
        }
        action, diagnostics = v3.PostAdasReplayPolicy(
            self.profile
        ).action(sensors)
        self.assertAlmostEqual(
            action["steer"],
            reference["steer_action"],
            delta=1e-6,
        )
        self.assertAlmostEqual(
            action["accel"],
            reference["accel_action"],
            delta=1e-6,
        )
        self.assertAlmostEqual(
            action["brake"],
            reference["brake_action"],
            delta=1e-6,
        )
        self.assertEqual(action["gear"], reference["gear_action"])
        self.assertEqual(diagnostics["steering_correction"], 0.0)

    def test_dynamic_correction_limit_is_respected(self):
        sensors = {
            "distFromStart": 950.0,
            "speedX": 120.0,
            "speedY": -100.0,
            "trackPos": -2.0,
            "angle": 2.0,
        }
        _, diagnostics = v3.PostAdasReplayPolicy(
            self.profile
        ).action(sensors)
        self.assertLessEqual(
            abs(diagnostics["steering_correction"]),
            diagnostics["correction_limit"],
        )
        self.assertLessEqual(
            diagnostics["correction_limit"],
            v3.MAX_STEER_CORRECTION,
        )

    def test_danger_does_not_change_pedals_outside_recovery(self):
        reference = self.profile.reference_at(950.0)
        sensors = {
            "distFromStart": 950.0,
            "speedX": reference["target_speedX"],
            "speedY": reference["target_speedY"],
            "trackPos": reference["target_trackPos"],
            "angle": reference["target_angle"],
        }
        action, diagnostics = v3.PostAdasReplayPolicy(
            self.profile
        ).action(sensors)
        self.assertEqual(diagnostics["mode"], "replay")
        self.assertAlmostEqual(
            action["accel"],
            reference["accel_action"],
        )
        self.assertAlmostEqual(
            action["brake"],
            reference["brake_action"],
        )

    def test_danger_score_and_level_are_deterministic(self):
        safe = self.profile.reference_at(500.0)
        dangerous = self.profile.reference_at(950.0)
        self.assertGreater(
            dangerous["danger_score"],
            safe["danger_score"],
        )
        self.assertIn(
            dangerous["danger_level"],
            ("high", "critical"),
        )
        sector = self.profile.danger_map[dangerous["danger_sector"]]
        expected = (
            0.25 * sector["brake_component"]
            + 0.20 * sector["steer_component"]
            + 0.15 * sector["speedY_component"]
            + 0.15 * sector["angle_component"]
            + 0.15 * sector["edge_component"]
            + 0.10 * sector["deceleration_component"]
        )
        self.assertAlmostEqual(sector["danger_score"], expected)

    def test_recovery_thresholds_and_blend_follow_danger(self):
        reference = self.profile.reference_at(950.0)
        sensors = {
            "distFromStart": 950.0,
            "speedX": reference["target_speedX"],
            "speedY": reference["target_speedY"],
            "trackPos": 1.1,
            "angle": reference["target_angle"],
        }
        policy = v3.PostAdasReplayPolicy(self.profile)
        _, first = policy.action(sensors)
        _, second = policy.action(sensors)
        self.assertGreater(first["recovery_blend_in"], v3.RECOVERY_BLEND_IN)
        self.assertAlmostEqual(
            first["recovery_blend"],
            first["recovery_blend_in"],
        )
        self.assertAlmostEqual(
            second["recovery_blend"],
            min(1.0, first["recovery_blend_in"] * 2.0),
        )
        self.assertLess(
            first["enter_track"],
            v3.RECOVERY_ENTER_TRACK_POS,
        )

        safe_sensors = {
            "distFromStart": 950.0,
            "speedX": reference["target_speedX"],
            "speedY": reference["target_speedY"],
            "trackPos": reference["target_trackPos"],
            "angle": reference["target_angle"],
        }
        previous_blend = second["recovery_blend"]
        _, leaving = policy.action(safe_sensors)
        self.assertLess(leaving["recovery_blend"], previous_blend)
        self.assertAlmostEqual(
            previous_blend - leaving["recovery_blend"],
            v3.RECOVERY_BLEND_OUT,
        )

    def test_braking_detection_merges_short_spatial_gap(self):
        rows = []
        for index in range(100):
            brake = (
                0.5
                if 10 <= index <= 25 or 33 <= index <= 48
                else 0.0
            )
            rows.append({
                "step": index,
                "brake_action": brake,
                "distFromStart": float(index),
                "speedX": 150.0 - index,
            })
        lap = {"run_id": "brake-test", "rows": rows}
        zones = v3.CorkscrewProfile._detected_braking_zones(
            lap,
            100.0,
        )
        self.assertEqual(len(zones), 1)
        self.assertEqual(zones[0]["start"], 10.0)
        self.assertGreaterEqual(zones[0]["end"], 48.0)

    def test_four_reports_are_generated(self):
        export_dir = os.path.join(self.temp_dir.name, "reports")
        paths = self.profile.export_reports(export_dir)
        self.assertEqual(len(paths), 4)
        for path in paths.values():
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_profile_lookup_p95_is_under_one_millisecond(self):
        durations = []
        for index in range(3000):
            start = time.perf_counter()
            self.profile.reference_at(index * 1.17)
            durations.append((time.perf_counter() - start) * 1000.0)
        self.assertLess(v3.percentile(durations, 0.95), 1.0)

    def test_manual_csv_preserves_exact_action(self):
        sensors = {
            "curLapTime": 12.5,
            "speedX": 100.0,
            "distFromStart": 500.0,
        }
        intention = {"steer": 1.0, "accel": 0.8, "brake": 0.0}
        action = {
            "steer": 0.312345678,
            "accel": 0.456789,
            "brake": 0.0,
            "gear": 4,
        }
        path = os.path.join(self.temp_dir.name, "manual.csv")
        dataset = manual.TransactionalDataset(path)
        dataset.run_id = "run-test"
        dataset.append(sensors, intention, action, 42)
        self.assertEqual(dataset.commit(), 1)
        with open(path, newline="", encoding="utf-8") as source:
            row = next(csv.DictReader(source))

        self.assertEqual(float(row["steer_action"]), action["steer"])
        self.assertEqual(float(row["accel_action"]), action["accel"])
        self.assertEqual(float(row["brake_action"]), action["brake"])
        self.assertEqual(int(row["gear_action"]), action["gear"])


if __name__ == "__main__":
    unittest.main()
