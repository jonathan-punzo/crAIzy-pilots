# crAIzy pilots - V3 multigiro post-ADAS

AI driver Python per TORCS specializzato sul circuito Corkscrew.

## File ufficiali

- `craizy_manual.py`: guida con controller PS4 e registra intenzione e azione
  post-ADAS realmente inviata a TORCS.
- `craizy_auto_v3.py`: costruisce il profilo multigiro e guida.
- `snakeoil3_jm2.py`: comunicazione con `scr_server 1`.
- `torcs_ps4_dataset.csv`: dataset originale, creato dal manuale.

V2 e V4 sono esperimenti storici indipendenti. V3 non importa altri piloti.

## Raccolta Dataset

```powershell
conda activate torcs-env
python craizy_manual.py
```

- `SELECT/SHARE`: scarta il tentativo e riavvia.
- `START/OPTIONS`: scarta il tentativo e riavvia.
- Un giro completo pulito viene aggiunto al dataset principale.

Servono almeno tre giri post-ADAS completi, senza danni e senza uscite.
I segmenti parziali non vengono salvati né usati dal profilo multigiro.

## Profilo Corkscrew

Ad ogni avvio V3:

1. valida tutti i giri;
2. stima la lunghezza pista dalla mediana;
3. sceglie il giro più veloce per gas, freno, sterzo e marcia;
4. ricampiona ogni giro su una griglia circolare da `5 m`;
5. calcola racing line, velocità mediana, MAD e affidabilità;
6. rileva le zone di frenata sui campioni temporali originali;
7. costruisce una danger map ogni `20 m`.

La danger map modifica soltanto limite della correzione sterzo e soglie del
recovery. Non cambia gas, freno o marcia durante il replay normale.

## Analisi Senza TORCS

```powershell
python craizy_auto_v3.py --analyze-only
```

Genera:

```text
results/corkscrew_profile_5m.csv
results/corkscrew_danger_20m.csv
results/corkscrew_braking_zones.csv
results/corkscrew_lap_comparison.csv
```

Il profilo contiene sia valori TORCS reali sia colonne normalizzate per
analisi e futuro machine learning. Il dataset originale non viene modificato.

## Avvio Pilota

```powershell
python craizy_auto_v3.py
```

Nel replay normale:

- le azioni post-ADAS provengono dal giro migliore;
- la racing line sfrutta il consenso tra tutti i giri;
- la correzione sterzo non supera `+-0,12`;
- bassa affidabilità e danger elevato riducono la correzione;
- recovery usa isteresi e blend graduale.

## Verifica

```powershell
python -m unittest -v test_post_adas_v3.py
```

Gate 1: un giro completo, danno zero e zero uscite.

Gate 2: tre giri consecutivi, danno zero, zero uscite e nessun recovery
grave. KNN verrà valutato soltanto dopo il Gate 2.
