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

## Track Blocks
The Corkscrew track is represented by one ordered `TRACK_BLOCKS` catalog.
Each block has a name, distance interval, driving role and protection flag:

| Block | Distance | Role |
|---|---:|---|
| S01 | 0-330 m | Start straight |
| S02_FIRST_CORNER | 330-550 m | Protected first corner |
| S03 | 550-1000 m | Technical |
| S04 | 1000-1500 m | Fast |
| S05 | 1500-2000 m | Technical |
| S06 | 2000-2330 m | Corkscrew approach |
| S07_CORKSCREW | 2330-2530 m | Protected Corkscrew |
| S08 | 2530-3080 m | Technical |
| S09_LAST_CORNER | 3080-3310 m | Protected last corner |
| S10 | 3310-3610 m | Finish straight |

Telemetry, validation and future local policies use this shared catalog.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Phase 1: Base Sensory Bot | Implement deterministic sensory steer and target speed control. Complete corkscrew lap at 60-90 km/h. | None | COMPLETE |
| 2 | Phase 2: Target Speed Sensoriale | Optimize target speed based on 19 track sensors to decelerate before curves. | M1 | COMPLETE |
| 3 | Phase 3: KNN Advisor | Train KNN on torcs_ps4_dataset.csv and integrate as residual advisor with rigid clamp. | M2 | IMPLEMENTED, OFFLINE VERIFIED |
| 4 | Phase 4: Safety Governor & ADAS | Implement Safety Governor rules and integrate ADAS. Pass E2E test (3 consecutive clean laps). | M3 | COMPLETE: V9R4, 5 CLEAN LAPS |
| 5 | Phase 5: Statistical Validation | Measure reliability, lap-time variance and sector performance before further tuning. | M4 | V9R4 INITIAL: 5/5 CLEAN, EXTENDED RUN PENDING |
