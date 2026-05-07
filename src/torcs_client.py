import socket
import time
from typing import Any


DATA_SIZE = 2**17
DEFAULT_SENSOR_ANGLES = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
WINDOWS_UDP_RESET = 10054


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def destringify(value: Any) -> Any:
    if not value:
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if isinstance(value, list):
        if len(value) == 1:
            return destringify(value[0])
        return [destringify(item) for item in value]
    return value


class ServerState:
    def __init__(self) -> None:
        self.servstr = ""
        self.d: dict[str, Any] = {}

    def parse_server_str(self, server_string: str) -> None:
        self.servstr = server_string.strip()[:-1]
        parts = self.servstr.strip().lstrip("(").rstrip(")").split(")(")
        for part in parts:
            tokens = part.split(" ")
            if not tokens or not tokens[0]:
                continue
            self.d[tokens[0]] = destringify(tokens[1:])

    def __repr__(self) -> str:
        lines = []
        for key in sorted(self.d):
            value = self.d[key]
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            lines.append(f"{key}: {value}")
        return "\n".join(lines)


class DriverAction:
    def __init__(self) -> None:
        self.d: dict[str, Any] = {
            "accel": 0.2,
            "brake": 0.0,
            "clutch": 0.0,
            "gear": 1,
            "steer": 0.0,
            "focus": [-90, -45, 0, 45, 90],
            "meta": 0,
        }

    def clip_to_limits(self) -> None:
        self.d["steer"] = clip(float(self.d.get("steer", 0.0)), -1.0, 1.0)
        self.d["brake"] = clip(float(self.d.get("brake", 0.0)), 0.0, 1.0)
        self.d["accel"] = clip(float(self.d.get("accel", 0.0)), 0.0, 1.0)
        self.d["clutch"] = clip(float(self.d.get("clutch", 0.0)), 0.0, 1.0)
        if self.d.get("gear") not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d["gear"] = 0
        if self.d.get("meta") not in [0, 1]:
            self.d["meta"] = 0
        focus = self.d.get("focus", 0)
        if isinstance(focus, list):
            if min(focus) < -180 or max(focus) > 180:
                self.d["focus"] = 0
        elif focus != 0:
            self.d["focus"] = 0

    def __repr__(self) -> str:
        self.clip_to_limits()
        chunks = []
        for key, value in self.d.items():
            if isinstance(value, list):
                formatted = " ".join(str(item) for item in value)
            else:
                formatted = f"{float(value):.3f}"
            chunks.append(f"({key} {formatted})")
        return "".join(chunks)


class Client:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3001,
        client_id: str = "SCR",
        max_steps: int = 100000,
        sensor_angles: str = DEFAULT_SENSOR_ANGLES,
        debug: bool = False,
        connect_timeout: float | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.sid = client_id
        self.maxSteps = max_steps
        self.sensor_angles = sensor_angles
        self.debug = debug
        self.S = ServerState()
        self.R = DriverAction()
        self.so: socket.socket | None = None
        self.active = True
        self.setup_connection(connect_timeout)

    def setup_connection(self, connect_timeout: float | None) -> None:
        self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.so.settimeout(1)
        initmsg = f"{self.sid}(init {self.sensor_angles})"
        started = time.monotonic()

        while True:
            if connect_timeout is not None and time.monotonic() - started > connect_timeout:
                raise TimeoutError(f"TORCS server not identified on {self.host}:{self.port}")

            self.so.sendto(initmsg.encode("utf-8"), (self.host, self.port))
            try:
                sockdata, _ = self.so.recvfrom(DATA_SIZE)
            except (socket.timeout, ConnectionResetError, OSError) as exc:
                if not self._is_waiting_error(exc):
                    raise
                print(f"Waiting for TORCS server on {self.host}:{self.port} ...")
                continue

            message = sockdata.decode("utf-8", errors="replace")
            if "***identified***" in message:
                print(f"Client connected on {self.host}:{self.port}")
                return

    def get_servers_input(self) -> bool:
        if not self.so:
            return False

        while True:
            try:
                sockdata, _ = self.so.recvfrom(DATA_SIZE)
            except (socket.timeout, ConnectionResetError, OSError) as exc:
                if not self._is_waiting_error(exc):
                    raise
                print(".", end=" ")
                continue

            message = sockdata.decode("utf-8", errors="replace")
            if "***identified***" in message:
                continue
            if "***shutdown***" in message or "***restart***" in message:
                self.active = False
                return False
            if not message:
                continue

            self.S.parse_server_str(message)
            if self.debug:
                print(self.S)
            return True

    @staticmethod
    def _is_waiting_error(exc: BaseException) -> bool:
        if isinstance(exc, socket.timeout):
            return True
        if isinstance(exc, ConnectionResetError):
            return True
        return getattr(exc, "winerror", None) == WINDOWS_UDP_RESET

    def respond_to_server(self) -> None:
        if not self.so:
            return
        self.so.sendto(repr(self.R).encode("utf-8"), (self.host, self.port))

    def shutdown(self) -> None:
        if self.so:
            print(f"Race terminated or {self.maxSteps} steps elapsed. Shutting down {self.port}.")
            self.so.close()
            self.so = None
        self.active = False
