import csv
import json
import math
import os
import socket
import sys
import time

import pygame
import snakeoil3_jm2 as snakeoil3

# ============================================================
# crAIzy pilots - DualShock 4 controller and dataset recorder
# ============================================================

PORT = 3001
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "torcs_ps4_dataset.csv")

DATASET_COLUMNS = [
    "run_id", "step", "curLapTime",
    "steer_intent", "accel_intent", "brake_intent",
    "steer_action", "accel_action", "brake_action", "gear_action",
    "speedX", "speedY", "speedZ",
    "wheelSpinVel", "z", "track", "trackPos", "angle",
    "rpm", "damage", "distFromStart",
]

OFFTRACK_CONFIRM_TICKS = 10
OFFTRACK_TRACK_POS = 1.05
PRINT_EVERY = 50
SERVER_SILENCE_TIMEOUTS = 3
SERVER_FINISH_MIN_SECONDS = 45.0
SERVER_FINISH_MIN_DISTANCE = 3500.0
SERVER_FINISH_MIN_ROWS = 1000


# DualShock 4 mapping and input calibration.
STEER_AXIS = 0
L2_AXIS = 4
R2_AXIS = 5
SHARE_BUTTON = 4
OPTIONS_BUTTON = 6

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

SHIFT_COOLDOWN = 0.35
UPSHIFT_RPM = 7600.0
DOWNSHIFT_RPM = 3300.0
PANIC_DOWNSHIFT_RPM = 2300.0
UPSHIFT_SPEED = {1: 45.0, 2: 78.0, 3: 112.0, 4: 148.0, 5: 184.0}
DOWNSHIFT_SPEED = {2: 30.0, 3: 58.0, 4: 90.0, 5: 122.0, 6: 155.0}
MIN_SPEED_FOR_UPSHIFT = {1: 22.0, 2: 48.0, 3: 78.0, 4: 110.0, 5: 145.0}


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


def apply_trigger_curve(value):
    value = max(0.0, min(1.0, safe_float(value)))
    if value <= TRIGGER_DEADZONE:
        return 0.0

    normalized = (value - TRIGGER_DEADZONE) / max(1.0 - TRIGGER_DEADZONE, 0.0001)
    return max(0.0, min(1.0, normalized)) ** TRIGGER_PROGRESSION


def axis_value(joystick, axis_index, default=0.0):
    try:
        if axis_index is None:
            return default
        if axis_index < 0 or axis_index >= joystick.get_numaxes():
            return default
        return safe_float(joystick.get_axis(axis_index), default)
    except Exception:
        return default


def normalize_trigger(raw):
    raw = safe_float(raw)
    value = (raw + 1.0) / 2.0 if raw < -0.05 else raw
    return max(0.0, min(1.0, value))


def apply_deadzone_and_curve(value, deadzone, progression):
    value = safe_float(value)
    if abs(value) <= deadzone:
        return 0.0

    sign = 1.0 if value > 0.0 else -1.0
    normalized = (abs(value) - deadzone) / max(1.0 - deadzone, 0.0001)
    normalized = max(0.0, min(1.0, normalized))
    return sign * (normalized ** progression)


def build_dataset_row(sensors, intention, action, run_id, step):
    return {
        "run_id": run_id,
        "step": int(step),
        "curLapTime": sensors.get("curLapTime", 0.0),
        "steer_intent": intention.get("steer", 0.0),
        "accel_intent": intention.get("accel", 0.0),
        "brake_intent": intention.get("brake", 0.0),
        "steer_action": action.get("steer", 0.0),
        "accel_action": action.get("accel", 0.0),
        "brake_action": action.get("brake", 0.0),
        "gear_action": int(action.get("gear", 1)),
        "speedX": sensors.get("speedX", 0.0),
        "speedY": sensors.get("speedY", 0.0),
        "speedZ": sensors.get("speedZ", 0.0),
        "wheelSpinVel": json.dumps(safe_list(
            sensors.get(
                "wheelSpinVel",
                sensors.get("wheelSpedVel", [0.0] * 4),
            ),
            4,
            0.0,
        )),
        "z": sensors.get("z", 0.0),
        "track": json.dumps(safe_list(
            sensors.get("track", [200.0] * 19),
            19,
            200.0,
        )),
        "trackPos": sensors.get("trackPos", 0.0),
        "angle": sensors.get("angle", 0.0),
        "rpm": sensors.get("rpm", 0.0),
        "damage": sensors.get("damage", 0.0),
        "distFromStart": sensors.get("distFromStart", 0.0),
    }


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
            if (
                (rpm >= UPSHIFT_RPM and enough_speed)
                or speed >= UPSHIFT_SPEED.get(current, 999.0)
            ):
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

        target_accel *= linear_limit(
            target_steer,
            THROTTLE_STEER_START,
            THROTTLE_STEER_FULL,
            THROTTLE_STEER_MIN_ACCEL,
        )

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


class LapCompletionDetector:
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


class SupremePS4Controller:
    def __init__(self):
        self.running = True
        self.restart_requested = False
        self.exit_requested = False
        self.stop_reason = ""

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= 0:
            raise RuntimeError("Nessun controller rilevato.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print("[CONTROLLER] %s" % self.joystick.get_name())
        print(
            "[CONTROLLER] Assi: %d | Pulsanti: %d"
            % (self.joystick.get_numaxes(), self.joystick.get_numbuttons())
        )

    def reset_requests(self):
        self.restart_requested = False
        self.exit_requested = False

    def process_events(self):
        self.restart_requested = False

        removed_event = getattr(pygame, "JOYDEVICEREMOVED", None)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.exit_requested = True
                self.stop_reason = "window_closed"
                continue

            if removed_event is not None and event.type == removed_event:
                self.running = False
                self.exit_requested = True
                self.stop_reason = "controller_disconnected"
                print("\n[STOP] Controller scollegato.")
                continue

            if event.type != pygame.JOYBUTTONDOWN:
                continue

            if event.button == SHARE_BUTTON:
                self.restart_requested = True
                print("\n[SHARE] Tentativo scartato. Riavvio gara.")
                continue

            if event.button == OPTIONS_BUTTON:
                self.restart_requested = True
                print(
                    "\n[START/OPTIONS] Tentativo scartato. Riavvio gara."
                )

    def intention(self):
        self.process_events()
        if not self.running:
            return {"steer": 0.0, "accel": 0.0, "brake": 0.0}

        raw_steer = axis_value(self.joystick, STEER_AXIS, 0.0)
        if INVERT_STEERING:
            raw_steer *= -1.0

        steer = apply_deadzone_and_curve(
            raw_steer,
            STEER_DEADZONE,
            STEER_PROGRESSION,
        )
        raw_l2 = normalize_trigger(
            axis_value(self.joystick, L2_AXIS, 0.0)
        )
        raw_r2 = normalize_trigger(
            axis_value(self.joystick, R2_AXIS, 0.0)
        )

        return {
            "steer": steer,
            "accel": apply_trigger_curve(raw_r2),
            "brake": apply_trigger_curve(raw_l2),
        }

    def stop(self):
        try:
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass


class TransactionalDataset:
    def __init__(self, path=DATASET_PATH):
        self.path = path
        self.rows = []
        self.valid = True
        self.invalid_reason = ""
        self.offtrack_ticks = 0
        self.run_id = str(time.time_ns())

    def reset(self):
        self.rows = []
        self.valid = True
        self.invalid_reason = ""
        self.offtrack_ticks = 0
        self.run_id = str(time.time_ns())

    def discard(self, reason):
        discarded = len(self.rows)
        self.rows = []
        self.valid = False
        self.invalid_reason = reason
        return discarded

    def observe_track(self, sensors):
        track = safe_list(sensors.get("track", [200.0] * 19), 19, 200.0)
        track_pos = abs(safe_float(sensors.get("trackPos", 0.0)))
        offtrack = track_pos > OFFTRACK_TRACK_POS or min(track) < 0.0

        if offtrack:
            self.offtrack_ticks += 1
        else:
            self.offtrack_ticks = 0

        if self.valid and self.offtrack_ticks >= OFFTRACK_CONFIRM_TICKS:
            self.valid = False
            self.invalid_reason = "offtrack"
            print(
                "\n[DATASET] Tentativo invalidato: fuori pista. "
                "Premi SELECT/SHARE o START/OPTIONS per scartare "
                "e ripartire."
            )

    def append(self, sensors, intention, action, step):
        if not self.valid:
            return
        self.rows.append(build_dataset_row(
            sensors,
            intention,
            action,
            self.run_id,
            step,
        ))

    def commit(self):
        if not self.valid or not self.rows:
            return 0

        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        temporary_path = self.path + ".tmp"
        existing = os.path.exists(self.path) and os.path.getsize(self.path) > 0

        with open(temporary_path, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=DATASET_COLUMNS)
            writer.writeheader()

            if existing:
                with open(self.path, newline="", encoding="utf-8") as input_file:
                    reader = csv.DictReader(input_file)
                    if reader.fieldnames != DATASET_COLUMNS:
                        raise ValueError(
                            "Il dataset esistente ha colonne incompatibili."
                        )
                    writer.writerows(reader)

            writer.writerows(self.rows)

        committed = len(self.rows)
        os.replace(temporary_path, self.path)
        self.rows = []
        return committed


class RaceProgress:
    def __init__(self):
        self.max_dist_raced = 0.0
        self.max_dist_from_start = 0.0

    def observe(self, sensors):
        self.max_dist_raced = max(
            self.max_dist_raced,
            safe_float(sensors.get("distRaced", 0.0)),
        )
        self.max_dist_from_start = max(
            self.max_dist_from_start,
            safe_float(sensors.get("distFromStart", 0.0)),
        )

    def confirms_corkscrew_finish(self, elapsed, dataset):
        distance = max(self.max_dist_raced, self.max_dist_from_start)
        return (
            dataset.valid
            and len(dataset.rows) >= SERVER_FINISH_MIN_ROWS
            and elapsed >= SERVER_FINISH_MIN_SECONDS
            and distance >= SERVER_FINISH_MIN_DISTANCE
        )


def create_client():
    client = snakeoil3.Client(p=PORT, vision=False)
    client.get_servers_input()
    return client


def receive_server_input(client):
    """Receive one telemetry packet without waiting forever after the race."""
    if client is None or getattr(client, "so", None) is None:
        return "closed"

    timeouts = 0
    while timeouts < SERVER_SILENCE_TIMEOUTS:
        try:
            sockdata, _ = client.so.recvfrom(snakeoil3.data_size)
            sockdata = sockdata.decode("utf-8")
        except socket.timeout:
            timeouts += 1
            print(".", end=" ", flush=True)
            continue
        except OSError:
            client.shutdown()
            return "socket_error"

        if "***identified***" in sockdata:
            continue
        if "***shutdown***" in sockdata:
            print(
                "\nServer TORCS ha terminato la gara sulla porta %d."
                % client.port
            )
            client.shutdown()
            return "shutdown"
        if "***restart***" in sockdata:
            print("\nServer TORCS ha riavviato la gara sulla porta %d." % client.port)
            client.shutdown()
            return "restart"
        if not sockdata:
            continue

        client.S.parse_server_str(sockdata)
        return "data"

    return "timeout"


def restart_client(client):
    try:
        if client is not None and getattr(client, "so", None) is not None:
            client.R.d["steer"] = 0.0
            client.R.d["accel"] = 0.0
            client.R.d["brake"] = 0.0
            client.R.d["gear"] = 1
            client.R.d["clutch"] = 0.0
            client.R.d["meta"] = 1
            client.respond_to_server()
            client.get_servers_input()
    finally:
        if client is not None:
            client.shutdown()

    return create_client()


def send_safe_stop(client):
    try:
        if client is None or getattr(client, "so", None) is None:
            return
        client.R.d.update(
            {
                "steer": 0.0,
                "accel": 0.0,
                "brake": 1.0,
                "gear": 1,
                "clutch": 0.0,
                "meta": 0,
            }
        )
        client.respond_to_server()
    except Exception:
        pass


def main():
    controller = None
    client = None
    dataset = TransactionalDataset()
    adas = SharedADAS()
    finish_detector = LapCompletionDetector()
    race_progress = RaceProgress()
    attempt_started_at = time.time()
    step = 0
    stop_reason = "unknown"

    try:
        controller = SupremePS4Controller()
        client = create_client()

        print("=" * 72)
        print(" crAIzy pilots - DUALSHOCK 4 DATASET CONTROLLER")
        print("=" * 72)
        print(" Stick sinistro = sterzo | R2 = gas | L2 = freno")
        print(" SELECT/SHARE = scarta e riparti")
        print(" START/OPTIONS = scarta e riparti")
        print(" Ctrl+C = scarta il tentativo ed esce")
        print(" Dataset: %s" % DATASET_PATH)
        print("=" * 72)

        while controller.running:
            if client is None or getattr(client, "so", None) is None:
                stop_reason = "server_closed"
                break

            sensors = client.S.d
            elapsed = time.time() - attempt_started_at
            race_progress.observe(sensors)
            finished = finish_detector.update(sensors)

            if finished:
                if dataset.valid:
                    committed = dataset.commit()
                    print(
                        "\n[GIRO COMPLETO] %d righe aggiunte a %s."
                        % (committed, DATASET_PATH)
                    )
                    stop_reason = "lap_complete"
                else:
                    print(
                        "\n[GIRO NON SALVATO] Tentativo invalido: %s."
                        % dataset.invalid_reason
                    )
                    stop_reason = "invalid_lap_complete"
                break

            intention = controller.intention()
            if controller.exit_requested or not controller.running:
                stop_reason = controller.stop_reason or "controller_exit"
                break

            if controller.restart_requested:
                discarded = dataset.discard("manual_restart")
                print(
                    "[RESTART] %d righe temporanee eliminate."
                    % discarded
                )

                adas.reset()
                finish_detector = LapCompletionDetector()
                race_progress = RaceProgress()
                attempt_started_at = time.time()
                step = 0
                client = restart_client(client)
                dataset.reset()
                controller.reset_requests()
                print("[RESTART] Pista e registrazione riavviate.")
                continue

            dataset.observe_track(sensors)
            action, diagnostics = adas.apply(sensors, intention)
            dataset.append(sensors, intention, action, step)
            client.R.d.update(action)
            client.respond_to_server()
            receive_status = receive_server_input(client)
            step += 1

            if receive_status != "data":
                elapsed = time.time() - attempt_started_at
                if race_progress.confirms_corkscrew_finish(elapsed, dataset):
                    committed = dataset.commit()
                    print(
                        "\n[GIRO COMPLETO] %d righe aggiunte a %s."
                        % (committed, DATASET_PATH)
                    )
                    print(
                        "[GIRO COMPLETO] Confermato dalla fine del server "
                        "dopo %.0f m." % max(
                            race_progress.max_dist_raced,
                            race_progress.max_dist_from_start,
                        )
                    )
                    stop_reason = "lap_complete_server_" + receive_status
                else:
                    stop_reason = "server_" + receive_status
                break

            if step % PRINT_EVERY == 0:
                sys.stdout.write(
                    "\rstep=%6d speed=%6.1f gear=%d steer=%6.3f "
                    "accel=%5.2f brake=%5.2f ABS=%4.2f TCS=%4.2f rows=%6d   "
                    % (
                        step,
                        safe_float(sensors.get("speedX")),
                        action["gear"],
                        action["steer"],
                        action["accel"],
                        action["brake"],
                        diagnostics["abs_release"],
                        diagnostics["traction_cut"],
                        len(dataset.rows),
                    )
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("\n[CTRL+C] Tentativo scartato.")
    except RuntimeError as error:
        stop_reason = "controller_error"
        print("\n[ERRORE] %s" % error)
    finally:
        discarded = len(dataset.rows)
        dataset.rows = []
        send_safe_stop(client)
        if client is not None:
            client.shutdown()
        if controller is not None:
            controller.stop()
        print()
        print("Sessione terminata: %s" % stop_reason)
        if not stop_reason.startswith("lap_complete"):
            print("Righe temporanee scartate: %d" % discarded)


if __name__ == "__main__":
    main()
