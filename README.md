# crAIzy pilots - V7 Behavioral Cloning

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
