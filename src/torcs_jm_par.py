from __future__ import annotations

import argparse

from driver import DriverConfig, TorcsAIDriver
from driver.telemetry import TelemetryLogger
from torcs_client import Client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TORCS AI Racing modular driver")
    parser.add_argument("--host", default="localhost", help="TORCS server host")
    parser.add_argument("--port", type=int, default=3001, help="TORCS server port")
    parser.add_argument("--id", default="SCR", help="TORCS client id")
    parser.add_argument("--steps", type=int, default=100000, help="Maximum simulation steps")
    parser.add_argument("--log-dir", default="results/runs", help="Telemetry output directory")
    parser.add_argument("--no-log", action="store_true", help="Disable telemetry CSV logging")
    parser.add_argument("--debug", action="store_true", help="Print raw server state")
    parser.add_argument("--connect-timeout", type=float, default=None, help="Seconds before connection failure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DriverConfig()
    driver = TorcsAIDriver(config)
    client = Client(
        host=args.host,
        port=args.port,
        client_id=args.id,
        max_steps=args.steps,
        debug=args.debug,
        connect_timeout=args.connect_timeout,
    )
    logger = None

    if not args.no_log:
        logger = TelemetryLogger(args.log_dir, config.name, config.log_every)
        print(f"Telemetry log: {logger.path}")

    try:
        for step in range(args.steps):
            if not client.get_servers_input():
                break
            actions = driver.update(client.S.d)
            client.R.d.update(actions)
            if logger:
                logger.write(step, client.S.d, client.R.d, driver.last_info)
            client.respond_to_server()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        client.shutdown()
        if logger:
            logger.close()


if __name__ == "__main__":
    main()
