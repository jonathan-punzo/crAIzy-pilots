from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass

from driver.math_utils import clamp, lerp
from driver.telemetry import TelemetryLogger
from torcs_client import Client


VK = {
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "space": 0x20,
    "a": 0x41,
    "d": 0x44,
    "e": 0x45,
    "g": 0x47,
    "q": 0x51,
    "r": 0x52,
    "s": 0x53,
    "w": 0x57,
    "x": 0x58,
    "esc": 0x1B,
}


def key_down(name: str) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK[name]) & 0x8000)


def pressed_edge(name: str, memory: dict[str, bool]) -> bool:
    current = key_down(name)
    previous = memory.get(name, False)
    memory[name] = current
    return current and not previous


@dataclass
class ManualState:
    steer: float = 0.0
    accel: float = 0.0
    brake: float = 0.0
    gear: int = 1
    automatic_gears: bool = True
    stabilization: bool = True


class ManualController:
    def __init__(self) -> None:
        self.state = ManualState()
        self.key_memory: dict[str, bool] = {}
        self.last_info: dict[str, object] = {
            "target_speed": "manual",
            "corner_pressure": "",
            "slip": "",
            "mode": "manual",
            "opponent_guard": False,
        }

    def update(self, sensors: dict[str, object]) -> dict[str, float | int]:
        if pressed_edge("g", self.key_memory):
            self.state.automatic_gears = not self.state.automatic_gears
        if pressed_edge("x", self.key_memory):
            self.state.stabilization = not self.state.stabilization
        if pressed_edge("e", self.key_memory):
            self.state.gear = min(6, self.state.gear + 1)
            self.state.automatic_gears = False
        if pressed_edge("q", self.key_memory):
            self.state.gear = max(-1, self.state.gear - 1)
            self.state.automatic_gears = False
        if pressed_edge("r", self.key_memory):
            self.state.gear = -1
            self.state.automatic_gears = False

        steer_target = 0.0
        if key_down("left") or key_down("a"):
            steer_target += 1.0
        if key_down("right") or key_down("d"):
            steer_target -= 1.0

        speed = float(sensors.get("speedX", 0.0) or 0.0)
        angle = float(sensors.get("angle", 0.0) or 0.0)
        track_pos = float(sensors.get("trackPos", 0.0) or 0.0)

        speed_steer_limit = clamp(1.12 - speed / 330.0, 0.35, 1.0)
        steer_target *= speed_steer_limit
        if self.state.stabilization and abs(steer_target) < 0.05:
            steer_target += clamp(angle * 0.9 - track_pos * 0.18, -0.28, 0.28)

        accel_target = 1.0 if key_down("up") or key_down("w") else 0.0
        brake_target = 1.0 if key_down("down") or key_down("s") or key_down("space") else 0.0

        self.state.steer = clamp(lerp(self.state.steer, steer_target, 0.36), -1.0, 1.0)
        self.state.accel = clamp(lerp(self.state.accel, accel_target, 0.42), 0.0, 1.0)
        self.state.brake = clamp(lerp(self.state.brake, brake_target, 0.55), 0.0, 1.0)

        if self.state.automatic_gears:
            self.state.gear = self._automatic_gear(speed, float(sensors.get("rpm", 0.0) or 0.0))

        self.last_info = {
            "target_speed": "manual",
            "corner_pressure": "",
            "slip": "",
            "mode": f"manual_auto_gear={self.state.automatic_gears}_stab={self.state.stabilization}",
            "opponent_guard": False,
        }
        return {
            "steer": self.state.steer,
            "accel": self.state.accel,
            "brake": self.state.brake,
            "gear": self.state.gear,
            "clutch": 0.0,
            "meta": 0,
        }

    @staticmethod
    def _automatic_gear(speed: float, rpm: float) -> int:
        gear = 1
        for index, threshold in enumerate([0, 55, 100, 145, 190, 235], start=1):
            if speed >= threshold:
                gear = index
        if rpm > 8800:
            gear += 1
        if rpm < 2500:
            gear -= 1
        return min(max(gear, 1), 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual TORCS controller")
    parser.add_argument("--host", default="localhost", help="TORCS server host")
    parser.add_argument("--port", type=int, default=3001, help="TORCS server port")
    parser.add_argument("--id", default="SCR", help="TORCS client id")
    parser.add_argument("--steps", type=int, default=100000, help="Maximum simulation steps")
    parser.add_argument("--log-dir", default="results/manual_runs", help="Telemetry output directory")
    parser.add_argument("--no-log", action="store_true", help="Disable telemetry CSV logging")
    parser.add_argument("--debug", action="store_true", help="Print raw server state")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = ManualController()
    client = Client(host=args.host, port=args.port, client_id=args.id, max_steps=args.steps, debug=args.debug)
    logger = None
    if not args.no_log:
        logger = TelemetryLogger(args.log_dir, "manual", 1)
        print(f"Telemetry log: {logger.path}")

    print("Manual control: W/Up accel, S/Down/Space brake, A/Left and D/Right steer.")
    print("Q/E shift, G toggle automatic gears, X toggle stabilization, R reverse, Esc quit.")

    try:
        for step in range(args.steps):
            if key_down("esc"):
                break
            if not client.get_servers_input():
                break
            actions = controller.update(client.S.d)
            client.R.d.update(actions)
            if logger:
                logger.write(step, client.S.d, client.R.d, controller.last_info)
            client.respond_to_server()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        client.shutdown()
        if logger:
            logger.close()


if __name__ == "__main__":
    main()

