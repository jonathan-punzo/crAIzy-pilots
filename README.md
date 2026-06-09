# crAIzy pilots - V8 Sensor Base + KNN Advisor

La versione in sviluppo e' `craizy_auto_v8.py`. Combina:

- controllore deterministico basato sui 19 sensori pista;
- target speed predittivo con frenata proporzionale;
- KNN residuale limitato a `+/-0,12` sterzo e `+/-25 km/h`;
- Safety Governor con priorita' alla frenata della base;
- ABS, TCS e cambio automatico derivati da `craizy_manual.py`.

V6 e V7 restano baseline e non sono importate dalla V8.

## Comandi V8

```powershell
conda activate torcs-env

python craizy_auto_v8.py --analyze-only
python craizy_auto_v8.py --base-only --slow
python craizy_auto_v8.py --base-only
python craizy_auto_v8.py
python craizy_auto_v8.py --validation-report --validation-runs 10
python -m unittest test_craizy_auto_v8.py
```

La sequenza operativa e': giro lento, giro base, giro completo, quindi tre
giri completi consecutivi. I trace sono salvati in
`logs/auto_v8_latest.csv`; i riepiloghi in `results/auto_v8_runs.csv`.
Ogni tentativo salva inoltre tempo ufficiale e statistiche di dieci
settori in `results/auto_v8_validation.csv`. Dopo dieci tentativi, il
comando `--validation-report` mostra affidabilita', best lap, media,
deviazione standard, velocita' media e massimo `trackPos` per settore.
Il trace completo di ogni tentativo viene conservato in
`logs/auto_v8_runs/`, con timestamp e indicazione `clean` oppure `error`;
`logs/auto_v8_latest.csv` continua a contenere l'ultimo giro.
Il circuito e' suddiviso in dieci blocchi `TRACK_BLOCKS`; ogni riga dei
nuovi trace include `track_block` e `track_block_role`. Questa struttura
e' condivisa da telemetria e validazione e consente future policy locali
senza duplicare intervalli di distanza nel codice.
I trace V9r4 includono inoltre diagnostica behavior-neutral per
`track_pos_rate`, sterzo richiesto e filtrato, velocita' ruote, slip ABS
e traction slip. Questi campi servono a isolare le biforcazioni S05
senza introdurre nuove correzioni nel controller.

La variante sperimentale `v9r5_s03_entry` usa tale diagnostica per
proteggere l'ingresso tecnico di S03. Tra 660 e 710 metri interviene
solo quando l'auto e' ancora sulla linea interna e supera 172 km/h:
prepara la linea esterna con una correzione limitata e applica un cap
locale di 176 km/h. I passaggi gia' corretti non vengono modificati.

La V8 v9r4 usa il 96% della velocita' mediana dei giri esperti come
riferimento minimo nei settori ordinari. Prima curva, Corkscrew e ultima
curva mantengono i limiti protetti; il riferimento veloce viene
disattivato quando l'auto ha troppo moto laterale. Nelle curve ordinarie
il limite sensoriale puo' salire fino al 90% della velocita' esperta
locale, evitando che il mantenimento curva rallenti oltre meta' giro.
La revisione `r1` anticipa la frenata della prima curva e rinforza il
rientro soltanto vicino al bordo nei settori protetti. La revisione `r4`
mantiene invariata la V9r1 nei passaggi normali e attiva un recupero
locale soltanto quando, tra 2240 e 2370 metri, la posizione proiettata
esce dal corridoio osservato nei giri puliti.

## Validazione TORCS V9r4

Prima sessione di conferma:

- `5/5` giri completati senza uscita o recovery;
- affidabilita' `100%`;
- best lap `88,422 s`;
- mediana `88,946 s`;
- deviazione standard `0,627 s`.

La prossima fase e' una validazione estesa da almeno 10-20 giri prima di
ulteriori incrementi prestazionali.

## Benchmark Offline V8

Leave-one-lap-out sui quattro giri esperti:

| Comando | Base | Base + advisor |
|---|---:|---:|
| MAE sterzo | 0,1605 | 0,1095 |
| MAE pedale firmato | 0,5914 | 0,4389 |

Inferenza KNN al 95 percentile: circa `1,18 ms`, sotto il gate di `1,5 ms`.

## Archivio V7

Driver autonomo Python per TORCS basato sul metodo di imitation learning
presentato nelle slide del corso.

## File principali

- `craizy_auto_v7.py`: training, guida, analisi e raccolta DAgger.
- `models/craizy_v7_bc.pt`: rete PyTorch addestrata.
- `torcs_ps4_dataset.csv`: quattro giri esperti post-ADAS.
- `torcs_v7_dagger.csv`: correzioni raccolte durante i takeover.
- `snakeoil3_jm2.py`: comunicazione con `scr_server 1`.

La V6 resta congelata come baseline KNN e non viene importata dalla V7.

## Architettura

La policy normale è una rete:

```text
25 sensori -> Dense 128 ReLU -> Dense 64 ReLU -> 2 tanh
```

Le feature sono i 19 sensori `track`, `trackPos`, `angle`, `speedX`,
`speedY`, `rpm` e slittamento delle ruote. `distFromStart` non viene usato.

Gli output sono sterzo e pedale firmato:

```text
pedal > 0  -> acceleratore
pedal < 0  -> freno
```

Non sono presenti edge guard, brake guard, speed cap o correzioni ordinarie
della carreggiata. Durante il training si aggiunge un piccolo rumore ai
sensori per evitare reazioni estreme a stati quasi identici. Al runtime la
variazione massima per tick replica i limiti fisici già presenti nelle azioni
post-ADAS del dataset: sterzo dipendente dalla velocità e pedale `0,20`.
Cambio, clipping e recovery restano deterministici.

## Comandi

```powershell
conda activate torcs-env

python craizy_auto_v7.py --train
python craizy_auto_v7.py --analyze-only
python craizy_auto_v7.py
python craizy_auto_v7.py --dagger
```

In modalità DAgger, tenere premuto `L1` per guidare con il joystick. Vengono
salvati soltanto gli stati e le azioni post-ADAS durante il takeover. Dopo
la sessione, eseguire nuovamente `--train`.

## Risultati offline iniziali

Valutazione leave-one-lap-out sui quattro giri:

| Modello | MAE sterzo | MAE pedale | MAE frenate |
|---|---:|---:|---:|
| V7 MLP robusta | 0,0583 | 0,0660 | 0,0982 |
| V6 KNN | 0,0677 | 0,1654 | 0,4272 |

Inferenza V7 al 95 percentile: circa `0,04 ms`.

## Validazione TORCS

1. 500 metri senza uscita;
2. un giro completo;
3. takeover DAgger nei punti problematici;
4. riaddestramento;
5. tre giri consecutivi senza danni, uscite o recovery.
