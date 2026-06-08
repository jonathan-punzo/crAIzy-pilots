# Original User Request

## Initial Request — 2026-06-08T23:45:26+02:00

Sviluppare un bot di guida autonoma (v8) completo, semplice e funzionale per TORCS sul circuito Corkscrew. Lo sviluppo deve seguire rigorosamente un approccio in 4 fasi, partendo da un controllore deterministico basato su sensori per verificare la stabilità e la correttezza dei segni dello sterzo e della velocità prima di integrare un modello KNN come rifinitore ("advisor").

Working directory: `c:\Universita\Intelligenza Artificiale\torcs\crAIzy-pilots`
Integrity mode: development

## Requirements

### R1. Verifica dei Segni e Formule di Base
Verificare la correttezza fisica dei comandi inviati a TORCS:
- `trackPos`: 0 al centro, -1 a destra, +1 a sinistra.
- `steer`: -1 sterzata completa a destra, +1 sterzata completa a sinistra.
- Formula di centratura delle slide: `steer_center = (S['angle'] * K_ANGLE / math.pi) - (S['trackPos'] * K_POS)`. Se l'auto devia a sinistra (`trackPos > 0`), il termine di centratura deve sottrarre per sterzare a destra (valore negativo).

### R2. Fase 1: Bot Sensoriale Base (Senza KNN)
Implementare un bot deterministico basato su sensori con l'obiettivo di completare il circuito Corkscrew a velocità ridotta (60-90 km/h) per testare i segni, il freno e il recovery.
- Calcolo dello sterzo:
  `steer_center = K_ANGLE * angle / math.pi - K_POS * trackPos`
  `curve_hint = estimate_curve_from_track(track)`
  `steer = steer_center + K_CURVE * curve_hint`
- Controllo velocità:
  `target_speed = compute_target_speed(track, steer, trackPos, angle, speedY)`
  `if speedX < target_speed: accel = gradual, brake = 0`
  `else: accel = 0, brake = proportional to error`
- Structure: Seguire la sequenza modulare delle slide (`steer` -> `accel` -> `brake` -> `TCS` -> `gear`).

### R3. Fase 2: Target Speed Sensoriale
Ottimizzare la velocità target utilizzando i 19 sensori `track` per anticipare le frenate in curva:
- Calcolare il rischio curva combinando i sensori frontali (`track[9]`), quelli vicini (`min(track[7:12])`) e la differenza sinistra/destra.
- Ridurre la velocità target se l'auto rileva curve strette, sbandamento trasversale (`speedY` elevato), orientamento scorretto (`angle` elevato) o distanza dal centro (`trackPos` elevato). Il bot deve frenare per raggiungere la velocità target di sicurezza.

### R4. Fase 3: KNN come Advisor (Rifinitore)
Una volta ottenuto un giro completo e pulito con il bot deterministico base, introdurre il modello KNN (`KNeighborsRegressor` di `scikit-learn` addestrato sui dati di `torcs_ps4_dataset.csv`) esclusivamente come rifinitore residuo:
- Il KNN stima una deviazione dello sterzo (`delta_steer_knn`) e della velocità (`delta_speed_knn`) rispetto al bot base.
- L'azione finale viene calcolata applicando dei limiti rigidi all'influenza del KNN:
  `steer = steer_base + clamp(delta_steer_knn, -0.12, 0.12)`
  `target_speed = target_speed_base + clamp(delta_speed_knn, -25.0, 25.0)`

### R5. Fase 4: Safety Governor (Supervisore)
Aggiungere un supervisore di sicurezza con regole prioritarie e deterministiche a protezione del bot per prevenire sbandate ed uscite:
- Se `abs(trackPos) > 0.85` -> azzera acceleratore (`accel = 0`).
- Se `abs(trackPos) > 0.95` -> forza frenata di emergenza.
- Se `abs(speedY) > soglia` -> taglia accelerazione per stabilizzare.
- Se `speedX > target_speed + 40.0` -> forza frenata.
- Se il KNN suggerisce di accelerare ma la base consiglia di frenare, prevale la frenata della base.
- Integrare gli ADAS (ABS, TCS, marce automatiche) testati in `craizy_manual.py` (senza modificarli).

## Acceptance Criteria

### Fase 1 (Slow Lap Check)
- [ ] Il bot deterministico completa Corkscrew a velocità ridotta (60-90 km/h) senza testacoda o oscillazioni.

### Fase 3 e 4 (Full Run Check)
- [ ] Il bot finale (Base + KNN Advisor + Safety Governor) completa 3 giri consecutivi senza incidenti, danni, uscite o attivazione prolungata del recovery.
- [ ] Il KNN non può provocare incidenti o deviazioni oltre i limiti protetti dai clamp e dal Safety Governor.
- [ ] L'inferenza del KNN in tempo reale richiede meno di 1.5 ms per tick.
