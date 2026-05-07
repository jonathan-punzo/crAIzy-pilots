# Test report

Compilare una riga per ogni sessione.

| Data | Config | Track | Driver | Giri completati | Best lap | Media lap | Damage max | Offtrack | Note |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-05-07 | best_lap | Corkscrew | scr_server 1 | TBD | TBD | TBD | TBD | TBD | Unico preset automatico da gara |
| 2026-05-07 | manual_control | Corkscrew | scr_server 1 | TBD | TBD | TBD | TBD | TBD | Guida manuale per debug e dataset |
| 2026-05-07 | best_lap | Corkscrew | scr_server 1 | No | N/A | N/A | 0 | 0 | Log `20260507_092734_best_lap.csv`: blocco a ~56 km/h, accel medio 0.87, RPM medi ~7879, picchi 17800, cambio oscillante 1/2. Fix: isteresi cambio, launch control, traction cut piu' forte, log comandi con prefisso `cmd_`. |
| 2026-05-07 | best_lap | Corkscrew | scr_server 1 | No | N/A | N/A | 0 | Si | Log `20260507_093300_best_lap.csv`: oscillazione su rettilineo a ~155 km/h, `cmd_steer` fino a +0.46/-0.61, `speedY` oltre 13 km/h e recovery penalizzato dalla sesta a bassa velocita'. Fix: gain sterzo ridotto, damping su `speedY`, limite sterzo ad alta velocita', rate limit sterzo, scalata forzata in recovery. |
| 2026-05-07 | best_lap | Corkscrew | scr_server 1 | No | N/A | N/A | 1793 | Si | Log `20260507_093845_best_lap.csv`: 445s di run, 14.657 campioni offtrack su 21.925, primo offtrack a 17.376s / 434m. Causa principale: segno dello sterzo invertito rispetto al comportamento reale del client; `cmd_steer` positivo riduce `trackPos`, quindi il controller spingeva verso l'esterno. Fix: inversione segno steering/manual/recovery, edge guard prima di `abs(trackPos)>1`, logging `edge_pressure`. |
| 2026-05-07 | best_lap | Corkscrew | scr_server 1 | TBD | TBD | TBD | TBD | TBD | Fix radar: sensori `track` larghi `-90..+90`, steering basato anche sul corridoio libero, wall avoidance, deadband su rettilineo (`straight_enough`) e log di `radar_front`, `radar_bias`, `wall_bias`. |
| 2026-05-07 | best_lap | Corkscrew | scr_server 1 | No | N/A | N/A | 98 | Si | Log `20260507_101238_best_lap.csv`: max 104.5 km/h ma p95 84.7 km/h perche' a ~434m entra fuori pista (`trackPos=-1.003`), poi 5.167 campioni in offtrack recovery e 2.166 in reverse. Analisi empirica: `cmd_steer` positivo fa diminuire `trackPos`, quindi il segno automatico era ancora opposto. Fix: inversione target steering, offtrack recovery e stabilizzazione manuale. |

## Criteri minimi

- 5 giri consecutivi: baseline accettabile.
- 10 giri consecutivi: stabilita' accettabile.
- Test con altri driver: must-have.
- Miglior giro registrato da partenza ferma: materiale video ufficiale.
