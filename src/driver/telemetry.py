from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any


SENSOR_COLUMNS = [
    "angle",
    "trackPos",
    "speedX",
    "speedY",
    "speedZ",
    "rpm",
    "gear",
    "damage",
    "curLapTime",
    "lastLapTime",
    "distFromStart",
    "distRaced",
    "racePos",
    "z",
]

ACTION_COLUMNS = ["cmd_steer", "cmd_accel", "cmd_brake", "cmd_gear", "cmd_clutch", "cmd_meta"]
INFO_COLUMNS = [
    "target_speed",
    "corner_pressure",
    "edge_pressure",
    "radar_front",
    "radar_bias",
    "wall_bias",
    "straight_enough",
    "slip",
    "mode",
    "opponent_guard",
]


class TelemetryLogger:
    def __init__(self, log_dir: str | Path, run_name: str, log_every: int = 1) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(log_dir) / f"{timestamp}_{run_name}.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log_every = max(1, log_every)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=["step"] + SENSOR_COLUMNS + ACTION_COLUMNS + INFO_COLUMNS,
        )
        self.writer.writeheader()

    def write(
        self,
        step: int,
        sensors: dict[str, Any],
        actions: dict[str, Any],
        info: dict[str, Any],
    ) -> None:
        if step % self.log_every != 0:
            return
        row: dict[str, Any] = {"step": step}
        for column in SENSOR_COLUMNS:
            row[column] = sensors.get(column, "")
        for column in ACTION_COLUMNS:
            action_name = column.removeprefix("cmd_")
            row[column] = actions.get(action_name, "")
        for column in INFO_COLUMNS:
            row[column] = info.get(column, "")
        self.writer.writerow(row)

    def close(self) -> None:
        self.handle.close()
