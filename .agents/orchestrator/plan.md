# Plan — Autonomous Driving Bot (v8) for TORCS on Corkscrew Track

This implementation plan details the 4-phase development of `craizy_auto_v8.py`, verifying milestones at each stage.

## 4-Phase Plan Decomposition

### Phase 1: Base Sensory Bot (No KNN)
- **Objective**: Develop a purely deterministic sensor-based controller.
- **Tasks**:
  1. Set up the basic structure of the controller following the course slide sequence (`steer` -> `accel` -> `brake` -> `TCS` -> `gear`).
  2. Implement steering center logic: `steer_center = (angle * K_ANGLE / math.pi) - (trackPos * K_POS)`. Verify correctness of signs (deviations to the left must steer right).
  3. Estimate curves using `track` sensors to compute `curve_hint`, adding to steering.
  4. Implement basic speed control targeting a low speed of 60-90 km/h to verify stability.
  5. Test basic recovery state machine.
- **Verification**: Complete a slow lap (60-90 km/h) on the Corkscrew track without spinning or oscillating.

### Phase 2: Target Speed Sensoriale
- **Objective**: Optimize speed control to handle curves safely using sensor readings.
- **Tasks**:
  1. Calculate curve risk using the 19 track sensors (front sensor `track[9]`, near sensors `min(track[7:12])`, and left/right differences).
  2. Map this risk to reduce the target speed dynamically.
  3. Reduce target speed if high lateral drift (`speedY`), high orientation deviation (`angle`), or large distance from center (`trackPos`).
  4. Ensure proportional braking when the current speed exceeds the safe target speed.
- **Verification**: Perform a lap at higher speeds safely, braking before curves.

### Phase 3: KNN as Advisor (Rifinitore)
- **Objective**: Add a KNN model trained on `torcs_ps4_dataset.csv` as a residual advisor.
- **Tasks**:
  1. Train or load `KNeighborsRegressor` on the professional dataset.
  2. Predict steering deviation (`delta_steer_knn`) and speed deviation (`delta_speed_knn`) relative to the base bot.
  3. Apply rigid clamps to the advisor's outputs: `delta_steer_knn` within `[-0.12, 0.12]`, and `delta_speed_knn` within `[-25.0, 25.0]`.
  4. Combine baseline actions with clamped KNN residuals:
     - `steer = steer_base + clamp(delta_steer_knn, -0.12, 0.12)`
     - `target_speed = target_speed_base + clamp(delta_speed_knn, -25.0, 25.0)`
  5. Verify real-time inference time is under 1.5 ms per tick.
- **Verification**: Verify that the KNN advisor behaves as a residual advisor without causing crashes.

### Phase 4: Safety Governor & ADAS Integration
- **Objective**: Implement the Safety Governor (Supervisore) and integrate ADAS.
- **Tasks**:
  1. If `abs(trackPos) > 0.85`, cut throttle (`accel = 0`).
  2. If `abs(trackPos) > 0.95`, force emergency braking.
  3. If `abs(speedY) > threshold`, cut throttle to stabilize.
  4. If `speedX > target_speed + 40.0`, force braking.
  5. If KNN suggests accelerating while the base bot suggests braking, enforce base braking (base brake priority).
  6. Integrate ADAS (ABS, TCS, automatic gears) exactly as tested in `craizy_manual.py`.
- **Verification**: Complete 3 consecutive clean laps on Corkscrew track with zero damage, zero offtracks, and no prolonged recovery activation.

---

## Verification Plan

### Phase 1 Verification
- Run TORCS with the base sensory bot.
- Confirm it completes a full lap between 60 km/h and 90 km/h.
- Verify that signs of steering and track position are physically correct.

### Phase 3/4 Verification
- Run TORCS with the complete system (Base + KNN Advisor + Safety Governor + ADAS).
- Verify 3 consecutive clean laps.
- Confirm inference time per tick is < 1.5 ms.
