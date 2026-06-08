# Project: crAIzy-pilots Autonomous Driver v8

Development of a simple, complete, and functional autonomous driving bot (v8) for TORCS on the Corkscrew track.

## Architecture
The system consists of three main blocks in a feedforward-override layout:
1. **Base Sensory Bot**: Deterministic controller reacting to physical sensors (angle, trackPos, track distances) to determine basic steering and throttle/brake intentions.
2. **KNN Advisor**: Machine learning advisor trained on expert demonstration data (`torcs_ps4_dataset.csv`) to compute residual steering/speed deviations (`delta_steer_knn`, `delta_speed_knn`).
3. **Safety Governor & ADAS**: Override/filter layer implementing priority safety rules (track position boundaries, speed capping, lateral drift limits) and vehicle control systems (ABS, TCS, automatic gearbox).

```
[Sensory State] ──┬──> [Base Sensory Bot] ──> Base Action (steer, speed)
                  │                                  │
                  └──> [KNN Advisor]      ──> Residual Delta (steer, speed)
                                                     │
                                                     ▼
                                              [Clamped Blend]
                                                     │
                                                     ▼
                                              Combined Action
                                                     │
                                                     ▼
                                            [Safety Governor]
                                                     │
                                                     ▼
                                                  [ADAS] ──> Final Action to TORCS
```

## Interface Contracts
The interface between the modules is inside the `RuntimePolicy` / `Driver` execution loop.
- **Sensors Input**: Dictionary of TORCS sensor data as parsed by `snakeoil3_jm2.py` (contains `trackPos`, `angle`, `speedX`, `speedY`, `track`, `rpm`, `wheelSpinVel`, `damage`, etc.).
- **Base Bot Outputs**: `steer_base` (float [-1.0, 1.0]), `target_speed_base` (float [km/h]).
- **KNN Advisor Outputs**: `delta_steer_knn` (float, clamped to [-0.12, 0.12]), `delta_speed_knn` (float, clamped to [-25.0, 25.0]).
- **Safety Governor Inputs**: `steer` (base + delta), `target_speed` (base + delta), telemetry.
- **Safety Governor Outputs**: `steer_safe`, `pedal_safe` (clamped intention).
- **ADAS Outputs**: `steer_final`, `accel_final`, `brake_final`, `gear_final` sent directly to the simulator.

## Code Layout
- `craizy_auto_v8.py` — Main entry point for the v8 bot.
- `snakeoil3_jm2.py` — TORCS client socket wrapper.
- `torcs_ps4_dataset.csv` — CSV file containing professional demonstrations for KNN training.
- `logs/` — Directory containing traces (`auto_v8_latest.csv`) and run summaries.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Phase 1: Base Sensory Bot | Implement deterministic sensory steer and target speed control. Complete corkscrew lap at 60-90 km/h. | None | PLANNED |
| 2 | Phase 2: Target Speed Sensoriale | Optimize target speed based on 19 track sensors to decelerate before curves. | M1 | PLANNED |
| 3 | Phase 3: KNN Advisor | Train KNN on torcs_ps4_dataset.csv and integrate as residual advisor with rigid clamp. | M2 | PLANNED |
| 4 | Phase 4: Safety Governor & ADAS | Implement Safety Governor rules and integrate ADAS. Pass E2E test (3 consecutive clean laps). | M3 | PLANNED |
