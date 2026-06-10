import ast
import csv
from collections import OrderedDict
from pathlib import Path
import unittest

import numpy as np

import craizy_auto_v8 as v8


def sensors(**overrides):
    values = {
        "angle": 0.0,
        "trackPos": 0.0,
        "speedX": 80.0,
        "speedY": 0.0,
        "rpm": 5000.0,
        "gear": 3,
        "wheelSpinVel": [67.0, 67.0, 67.0, 67.0],
        "track": [100.0] * 19,
    }
    values.update(overrides)
    return values


class BasePolicyTests(unittest.TestCase):
    def test_track_position_sign_centers_the_car(self):
        policy = v8.BaseSensorPolicy()
        left = policy.action_intent(sensors(trackPos=0.5))
        right = policy.action_intent(sensors(trackPos=-0.5))
        self.assertLess(left["steer"], 0.0)
        self.assertGreater(right["steer"], 0.0)

    def test_slow_mode_caps_target_speed(self):
        intent = v8.BaseSensorPolicy(slow=True).action_intent(
            sensors(track=[200.0] * 19)
        )
        self.assertLessEqual(intent["target_speed"], v8.SLOW_SPEED_CAP)

    def test_closed_track_reduces_target_speed(self):
        policy = v8.BaseSensorPolicy()
        open_road = policy.action_intent(sensors(track=[200.0] * 19))
        closed_road = policy.action_intent(sensors(track=[35.0] * 19))
        self.assertLess(
            closed_road["target_speed"], open_road["target_speed"]
        )

    def test_open_sensor_anticipates_tight_curve(self):
        left_curve = [15.0] * 19
        left_curve[2] = 80.0
        right_curve = [15.0] * 19
        right_curve[16] = 80.0
        policy = v8.BaseSensorPolicy()
        self.assertGreater(
            policy.action_intent(sensors(track=left_curve))["steer"],
            0.0,
        )
        self.assertLess(
            policy.action_intent(sensors(track=right_curve))["steer"],
            0.0,
        )


class AutomaticGearboxTests(unittest.TestCase):
    @staticmethod
    def state(speed, rpm, gear=None):
        values = {"speedX": speed, "rpm": rpm}
        if gear is not None:
            values["gear"] = gear
        return values

    def test_rpm_upshifts_use_hysteresis_thresholds(self):
        transitions = (
            (3, 4, 98.0, 85.0),
            (4, 5, 130.0, 120.0),
            (5, 6, 163.0, 153.0),
        )
        for current, target, upshift_speed, post_shift_speed in transitions:
            with self.subTest(current=current):
                gearbox = v8.AutomaticGearbox()
                gearbox.gear = current
                shifted = gearbox.update(
                    self.state(upshift_speed, v8.UPSHIFT_RPM, current),
                    accel=1.0,
                    now=0.0,
                )
                self.assertEqual(shifted, target)
                self.assertEqual(
                    gearbox.update(
                        self.state(post_shift_speed, 6000.0, target),
                        accel=1.0,
                        now=v8.SHIFT_COOLDOWN + 0.01,
                    ),
                    target,
                )

    def test_pending_shift_ignores_stale_sensor_during_cooldown(self):
        gearbox = v8.AutomaticGearbox()
        gearbox.gear = 3
        self.assertEqual(
            gearbox.update(
                self.state(98.0, v8.UPSHIFT_RPM, 3),
                accel=1.0,
                now=0.0,
            ),
            4,
        )
        self.assertEqual(
            gearbox.update(
                self.state(90.0, 6500.0, 3),
                accel=1.0,
                now=v8.SHIFT_COOLDOWN / 2.0,
            ),
            4,
        )

    def test_strong_acceleration_blocks_ordinary_downshift(self):
        gearbox = v8.AutomaticGearbox()
        gearbox.gear = 4
        self.assertEqual(
            gearbox.update(
                self.state(85.0, 5000.0, 4),
                accel=1.0,
                now=0.0,
            ),
            4,
        )

    def test_braking_and_panic_rpm_still_allow_downshift(self):
        braking = v8.AutomaticGearbox()
        braking.gear = 4
        self.assertEqual(
            braking.update(
                self.state(85.0, 5000.0, 4),
                accel=0.8,
                brake=0.2,
                now=0.0,
            ),
            3,
        )

        panic = v8.AutomaticGearbox()
        panic.gear = 4
        self.assertEqual(
            panic.update(
                self.state(100.0, 1900.0, 4),
                accel=1.0,
                now=0.0,
            ),
            3,
        )

    def test_manual_and_v8_gearboxes_remain_identical(self):
        root = Path(__file__).resolve().parent

        def gearbox_ast(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            gearbox = next(
                node for node in tree.body
                if isinstance(node, ast.ClassDef)
                and node.name == "AutomaticGearbox"
            )
            return ast.dump(gearbox, include_attributes=False)

        self.assertEqual(
            gearbox_ast(root / "craizy_manual.py"),
            gearbox_ast(root / "craizy_auto_v8.py"),
        )

    def test_dataset_replay_has_no_shift_oscillation(self):
        path = Path(__file__).resolve().parent / "torcs_ps4_dataset.csv"
        runs = OrderedDict()
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                runs.setdefault(row["run_id"], []).append(row)

        self.assertGreaterEqual(len(runs), 4)
        for run_id, rows in runs.items():
            gearbox = v8.AutomaticGearbox()
            shifts = []
            for row in rows:
                previous = gearbox.gear
                current = gearbox.update(
                    {
                        "speedX": row["speedX"],
                        "rpm": row["rpm"],
                    },
                    accel=float(row["accel_action"]),
                    brake=float(row["brake_action"]),
                    now=float(row["curLapTime"]),
                )
                if current != previous:
                    shifts.append((
                        float(row["curLapTime"]),
                        previous,
                        current,
                    ))

            reversals = sum(
                current[1] == previous[2]
                and current[2] == previous[1]
                and current[0] - previous[0] < 2.0
                for previous, current in zip(shifts, shifts[1:])
            )
            with self.subTest(run_id=run_id):
                self.assertGreaterEqual(len(shifts), 23)
                self.assertLessEqual(len(shifts), 30)
                self.assertLessEqual(reversals, 3)


class ValidationTests(unittest.TestCase):
    def test_track_blocks_cover_named_course_sections(self):
        expected = (
            "S01",
            "S02_FIRST_CORNER",
            "S03",
            "S04",
            "S05",
            "S06",
            "S07_CORKSCREW",
            "S08",
            "S09_LAST_CORNER",
            "S10",
        )
        self.assertEqual(
            tuple(block.name for block in v8.TRACK_BLOCKS),
            expected,
        )
        self.assertEqual(
            v8.track_block_at(400.0).name,
            "S02_FIRST_CORNER",
        )
        self.assertEqual(
            v8.track_block_at(2400.0).name,
            "S07_CORKSCREW",
        )
        self.assertEqual(
            v8.track_block_at(3200.0).name,
            "S09_LAST_CORNER",
        )

    def test_track_blocks_are_contiguous(self):
        for left, right in zip(
            v8.TRACK_BLOCKS, v8.TRACK_BLOCKS[1:]
        ):
            self.assertEqual(left.end, right.start)

    def test_validation_metrics_collect_sector_values(self):
        metrics = v8.ValidationMetrics()
        metrics.observe(sensors(
            curLapTime=1.0,
            distFromStart=100.0,
            speedX=120.0,
            trackPos=0.2,
        ))
        metrics.observe(sensors(
            curLapTime=1.1,
            distFromStart=120.0,
            speedX=140.0,
            trackPos=-0.4,
        ))
        values = metrics.values()
        self.assertAlmostEqual(values["S01_time"], 0.1)
        self.assertAlmostEqual(values["S01_avg_speed"], 130.0)
        self.assertAlmostEqual(values["S01_max_track_pos"], 0.4)

    def test_validation_summary_uses_only_clean_completed_laps(self):
        rows = [
            {
                "reason": "lap_complete",
                "lap_time": "90.0",
                "clean": "1",
                "S01_avg_speed": "150.0",
                "S01_max_track_pos": "0.4",
            },
            {
                "reason": "lap_complete",
                "lap_time": "92.0",
                "clean": "1",
                "S01_avg_speed": "140.0",
                "S01_max_track_pos": "0.5",
            },
            {
                "reason": "server_closed",
                "lap_time": "0.0",
                "clean": "0",
            },
        ]
        report = v8.validation_summary(rows)
        self.assertEqual(report["attempts"], 3)
        self.assertEqual(report["clean"], 2)
        self.assertEqual(report["best"], 90.0)
        self.assertEqual(report["mean"], 91.0)
        self.assertEqual(report["median"], 91.0)
        self.assertEqual(report["sectors"][0]["avg_speed"], 145.0)
        self.assertEqual(report["sectors"][0]["max_track_pos"], 0.5)

    def test_validation_accepts_clean_server_finish(self):
        row = {
            "reason": "server_closed",
            "lap_time": "0.0",
            "clean": "0",
            "offtrack_steps": "0",
            "recovery_steps": "0",
        }
        for name, _, _ in v8.VALIDATION_SECTORS:
            row["%s_time" % name] = "9.0"
            row["%s_avg_speed" % name] = "150.0"
            row["%s_max_track_pos" % name] = "0.5"
        report = v8.validation_summary([row])
        self.assertEqual(report["completed"], 1)
        self.assertEqual(report["clean"], 1)
        self.assertEqual(report["median"], 90.0)


class AdvisorAndSafetyTests(unittest.TestCase):
    class FixedSpeedProfile:
        def __init__(self, speed):
            self.speed = speed

        def speed_at(self, _distance):
            return self.speed

    def test_advisor_respects_authority_bounds(self):
        features = np.asarray([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
        ])
        targets = np.asarray([
            [1.0, 100.0],
            [1.0, 100.0],
            [1.0, 100.0],
            [1.0, 100.0],
            [1.0, 100.0],
            [1.0, 100.0],
            [1.0, 100.0],
        ])
        prediction = v8.ResidualAdvisor(features, targets).predict(
            features[0]
        )
        self.assertLessEqual(
            abs(prediction["delta_steer"]), v8.MAX_STEER_ADVICE
        )
        self.assertLessEqual(
            abs(prediction["delta_speed"]), v8.MAX_SPEED_ADVICE
        )

    def test_runtime_diagnostics_include_track_block(self):
        _, diagnostics = v8.RuntimePolicy().action(
            sensors(distFromStart=1200.0)
        )
        self.assertEqual(diagnostics["track_block"], "S04")
        self.assertEqual(diagnostics["track_block_role"], "fast")

    def test_runtime_diagnostics_include_vehicle_dynamics(self):
        action, diagnostics = v8.RuntimePolicy().action(
            sensors(
                distFromStart=1800.0,
                trackPos=-0.4,
                wheelSpinVel=[60.0, 61.0, 62.0, 63.0],
            )
        )
        self.assertIn("track_pos_rate", diagnostics)
        self.assertIn("filtered_steer_target", diagnostics)
        self.assertIn("front_wheel_speed", diagnostics)
        self.assertIn("rear_wheel_speed", diagnostics)
        self.assertIn("abs_slip", diagnostics)
        self.assertIn("traction_slip", diagnostics)
        self.assertEqual(
            diagnostics["actuator_steer"], action["steer"]
        )

    def test_edge_brake_overrides_positive_advice(self):
        state = sensors(trackPos=0.96, speedX=180.0)
        base = v8.BaseSensorPolicy().action_intent(state)
        governed = v8.SafetyGovernor().apply(
            state,
            base,
            {"delta_steer": 0.12, "delta_speed": 25.0},
        )
        self.assertEqual(governed["accel"], 0.0)
        self.assertGreater(governed["brake"], 0.0)
        self.assertIn("edge_brake", governed["interventions"])

    def test_base_braking_has_priority(self):
        state = sensors(speedX=260.0, track=[35.0] * 19)
        base = v8.BaseSensorPolicy().action_intent(state)
        self.assertLess(base["pedal"], 0.0)
        governed = v8.SafetyGovernor().apply(
            state,
            base,
            {"delta_steer": 0.0, "delta_speed": 25.0},
        )
        self.assertEqual(governed["accel"], 0.0)
        self.assertGreater(governed["brake"], 0.0)

    def test_projected_edge_adds_inward_steering(self):
        governor = v8.SafetyGovernor()
        governor.previous_track_pos = -0.50
        governor.track_pos_rate = -0.01
        state = sensors(trackPos=-0.54, speedX=120.0)
        base = v8.BaseSensorPolicy().action_intent(state)
        governed = governor.apply(
            state,
            base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        self.assertGreater(governed["steer"], base["steer"])
        self.assertIn("projected_edge", governed["interventions"])

    def test_tight_curve_cap_survives_brief_sensor_reopening(self):
        governor = v8.SafetyGovernor()
        tight_state = sensors(
            speedX=120.0,
            distFromStart=2450.0,
            track=[22.0] * 19,
        )
        tight_base = v8.BaseSensorPolicy().action_intent(tight_state)
        first = governor.apply(
            tight_state,
            tight_base,
            {"delta_steer": 0.0, "delta_speed": 25.0},
        )
        open_state = sensors(
            speedX=90.0,
            distFromStart=2470.0,
            track=[100.0] * 19,
        )
        open_base = v8.BaseSensorPolicy().action_intent(open_state)
        second = governor.apply(
            open_state,
            open_base,
            {"delta_steer": 0.0, "delta_speed": 25.0},
        )
        self.assertLess(first["target_speed"], 90.0)
        self.assertLess(second["target_speed"], 90.0)
        self.assertIn("tight_curve_hold", second["interventions"])

    def test_normal_curve_uses_faster_cap(self):
        governor = v8.SafetyGovernor()
        state = sensors(
            speedX=120.0,
            distFromStart=1500.0,
            track=[22.0] * 19,
        )
        base = v8.BaseSensorPolicy().action_intent(state)
        governed = governor.apply(
            state,
            base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        self.assertGreaterEqual(governed["target_speed"], 105.0)
        self.assertEqual(
            governed["tight_curve_ticks"],
            v8.FAST_CURVE_HOLD_TICKS,
        )
        self.assertIn(
            "performance_bonus", governed["interventions"]
        )

    def test_performance_bonus_skips_protected_sectors(self):
        base_policy = v8.BaseSensorPolicy()
        normal = sensors(
            speedX=120.0,
            distFromStart=2000.0,
            track=[80.0] * 19,
        )
        protected = sensors(
            speedX=120.0,
            distFromStart=2400.0,
            track=[80.0] * 19,
        )
        normal_governed = v8.SafetyGovernor().apply(
            normal,
            base_policy.action_intent(normal),
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        protected_governed = v8.SafetyGovernor().apply(
            protected,
            base_policy.action_intent(protected),
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        self.assertGreater(
            normal_governed["target_speed"],
            protected_governed["target_speed"],
        )
        self.assertNotIn(
            "performance_bonus",
            protected_governed["interventions"],
        )

    def test_first_corner_keeps_safe_profile(self):
        state = sensors(
            speedX=160.0,
            distFromStart=420.0,
            track=[22.0] * 19,
        )
        base = v8.BaseSensorPolicy().action_intent(state)
        governed = v8.SafetyGovernor().apply(
            state,
            base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        self.assertLessEqual(governed["target_speed"], 90.0)
        self.assertNotIn(
            "performance_bonus", governed["interventions"]
        )

    def test_first_corner_brakes_before_turn_in(self):
        state = sensors(
            speedX=205.0,
            distFromStart=360.0,
            track=[100.0] * 19,
        )
        base = v8.BaseSensorPolicy().action_intent(state)
        governed = v8.SafetyGovernor().apply(
            state,
            base,
            {"delta_steer": 0.0, "delta_speed": 25.0},
        )
        self.assertEqual(governed["target_speed"], 155.0)
        self.assertGreater(governed["brake"], 0.0)
        self.assertIn(
            "first_corner_speed", governed["interventions"]
        )

    def test_protected_edge_adds_extra_inward_steering(self):
        governor = v8.SafetyGovernor()
        governor.previous_track_pos = 0.62
        governor.track_pos_rate = 0.01
        state = sensors(
            speedX=105.0,
            trackPos=0.70,
            distFromStart=2400.0,
        )
        base = v8.BaseSensorPolicy().action_intent(state)
        governed = governor.apply(
            state,
            base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        self.assertLess(governed["steer"], base["steer"])
        self.assertIn(
            "protected_edge", governed["interventions"]
        )

    def test_corkscrew_rescue_activates_only_outside_clean_corridor(self):
        clean_governor = v8.SafetyGovernor()
        clean_governor.previous_track_pos = -0.12
        clean_state = sensors(
            speedX=225.0,
            trackPos=-0.12,
            distFromStart=2275.0,
        )
        clean_base = v8.BaseSensorPolicy().action_intent(clean_state)
        clean = clean_governor.apply(
            clean_state,
            clean_base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )

        risk_governor = v8.SafetyGovernor()
        risk_governor.previous_track_pos = -0.35
        risk_state = sensors(
            speedX=225.0,
            trackPos=-0.40,
            distFromStart=2275.0,
        )
        risk_base = v8.BaseSensorPolicy().action_intent(risk_state)
        risk = risk_governor.apply(
            risk_state,
            risk_base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )

        self.assertNotIn(
            "corkscrew_rescue", clean["interventions"]
        )
        self.assertIn(
            "corkscrew_rescue", risk["interventions"]
        )
        self.assertGreater(risk["steer"], risk_base["steer"])
        self.assertLessEqual(risk["target_speed"], 225.0)

    def test_s03_entry_guard_corrects_only_risky_approach(self):
        state = sensors(
            speedX=182.0,
            speedY=0.2,
            angle=0.02,
            trackPos=0.05,
            distFromStart=685.0,
        )
        base = v8.BaseSensorPolicy().action_intent(state)
        result = v8.SafetyGovernor().apply(
            state,
            base,
            {"delta_steer": 0.0, "delta_speed": 25.0},
        )

        self.assertIn(
            "s03_entry_guard",
            result["interventions"],
        )
        self.assertGreater(result["steer"], base["steer"])
        self.assertLessEqual(result["target_speed"], 176.0)

    def test_s03_entry_guard_preserves_safe_approaches(self):
        advisor = {"delta_steer": 0.02, "delta_speed": 0.0}
        outside_distance = sensors(
            speedX=182.0,
            trackPos=0.05,
            distFromStart=640.0,
        )
        safe_line = sensors(
            speedX=182.0,
            trackPos=0.38,
            distFromStart=685.0,
        )
        safe_speed = sensors(
            speedX=168.0,
            trackPos=0.05,
            distFromStart=685.0,
        )
        distance_base = v8.BaseSensorPolicy().action_intent(
            outside_distance
        )
        line_base = v8.BaseSensorPolicy().action_intent(safe_line)
        speed_base = v8.BaseSensorPolicy().action_intent(safe_speed)
        distance_result = v8.SafetyGovernor().apply(
            outside_distance, distance_base, advisor
        )
        line_result = v8.SafetyGovernor().apply(
            safe_line, line_base, advisor
        )
        speed_result = v8.SafetyGovernor().apply(
            safe_speed, speed_base, advisor
        )

        self.assertNotIn(
            "s03_entry_guard",
            distance_result["interventions"],
        )
        self.assertNotIn(
            "s03_entry_guard",
            line_result["interventions"],
        )
        self.assertNotIn(
            "s03_entry_guard",
            speed_result["interventions"],
        )

    def test_expert_speed_floor_boosts_only_safe_sector(self):
        profile = self.FixedSpeedProfile(260.0)
        normal_state = sensors(
            speedX=120.0,
            distFromStart=1200.0,
            track=[70.0] * 19,
        )
        normal_base = v8.BaseSensorPolicy().action_intent(normal_state)
        normal = v8.SafetyGovernor(speed_profile=profile).apply(
            normal_state,
            normal_base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        protected_state = sensors(
            speedX=120.0,
            distFromStart=2400.0,
            track=[70.0] * 19,
        )
        protected_base = v8.BaseSensorPolicy().action_intent(
            protected_state
        )
        protected = v8.SafetyGovernor(speed_profile=profile).apply(
            protected_state,
            protected_base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        self.assertGreaterEqual(
            normal["target_speed"],
            260.0 * v8.PERFORMANCE_PROFILE_FACTOR,
        )
        self.assertIn(
            "expert_speed_floor", normal["interventions"]
        )
        self.assertNotIn(
            "expert_speed_floor", protected["interventions"]
        )

    def test_expert_speed_floor_requires_stable_lateral_speed(self):
        profile = self.FixedSpeedProfile(260.0)
        state = sensors(
            speedX=120.0,
            speedY=12.0,
            distFromStart=1200.0,
            track=[70.0] * 19,
        )
        base = v8.BaseSensorPolicy().action_intent(state)
        governed = v8.SafetyGovernor(speed_profile=profile).apply(
            state,
            base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        self.assertNotIn(
            "expert_speed_floor", governed["interventions"]
        )

    def test_expert_curve_cap_relaxes_only_normal_sector(self):
        profile = self.FixedSpeedProfile(200.0)
        normal_state = sensors(
            speedX=120.0,
            distFromStart=1200.0,
            track=[25.0] * 19,
        )
        normal_base = v8.BaseSensorPolicy().action_intent(normal_state)
        normal = v8.SafetyGovernor(speed_profile=profile).apply(
            normal_state,
            normal_base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        protected_state = sensors(
            speedX=120.0,
            distFromStart=2400.0,
            track=[25.0] * 19,
        )
        protected_base = v8.BaseSensorPolicy().action_intent(
            protected_state
        )
        protected = v8.SafetyGovernor(speed_profile=profile).apply(
            protected_state,
            protected_base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        self.assertGreaterEqual(
            normal["target_speed"],
            200.0 * v8.CURVE_PROFILE_FACTOR,
        )
        self.assertIn("expert_curve_cap", normal["interventions"])
        self.assertNotIn(
            "expert_curve_cap", protected["interventions"]
        )
        self.assertLess(protected["target_speed"], normal["target_speed"])

    def test_last_corner_has_local_speed_cap(self):
        governor = v8.SafetyGovernor()
        state = sensors(
            speedX=210.0,
            distFromStart=3200.0,
            track=[100.0] * 19,
        )
        base = v8.BaseSensorPolicy().action_intent(state)
        governed = governor.apply(
            state,
            base,
            {"delta_steer": 0.0, "delta_speed": 25.0},
        )
        self.assertEqual(governed["target_speed"], 180.0)
        self.assertIn(
            "last_corner_speed", governed["interventions"]
        )

    def test_last_corner_prepares_outside_line(self):
        governor = v8.SafetyGovernor()
        state = sensors(
            speedX=170.0,
            trackPos=0.15,
            distFromStart=3100.0,
            track=[150.0] * 19,
        )
        base = v8.BaseSensorPolicy().action_intent(state)
        governed = governor.apply(
            state,
            base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        self.assertLess(governed["steer"], base["steer"])
        self.assertIn(
            "last_corner_line", governed["interventions"]
        )

    def test_last_corner_does_not_affect_other_sectors(self):
        governor = v8.SafetyGovernor()
        state = sensors(
            speedX=170.0,
            trackPos=0.15,
            distFromStart=2000.0,
            track=[150.0] * 19,
        )
        base = v8.BaseSensorPolicy().action_intent(state)
        governed = governor.apply(
            state,
            base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        self.assertNotIn(
            "last_corner_line", governed["interventions"]
        )
        self.assertNotIn(
            "last_corner_speed", governed["interventions"]
        )

    def test_last_corner_line_releases_before_apex(self):
        governor = v8.SafetyGovernor()
        state = sensors(
            speedX=110.0,
            trackPos=0.10,
            distFromStart=3250.0,
            track=[40.0] * 19,
        )
        base = v8.BaseSensorPolicy().action_intent(state)
        governed = governor.apply(
            state,
            base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        self.assertNotIn(
            "last_corner_line", governed["interventions"]
        )
        self.assertIn(
            "last_corner_speed", governed["interventions"]
        )

    def test_lateral_drift_adds_countersteer(self):
        governor = v8.SafetyGovernor()
        state = sensors(speedX=100.0, speedY=-10.0)
        base = v8.BaseSensorPolicy().action_intent(state)
        governed = governor.apply(
            state,
            base,
            {"delta_steer": 0.0, "delta_speed": 0.0},
        )
        self.assertGreater(governed["steer"], base["steer"])
        self.assertIn("lateral_steer", governed["interventions"])


if __name__ == "__main__":
    unittest.main()
