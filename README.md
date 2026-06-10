# crAIzy Auto

`crAIzy Auto` e' un pilota autonomo per TORCS progettato e validato sul
circuito Corkscrew con la monoposto `car1-ow1`.

Il sistema combina un controllore sensoriale deterministico, un advisor KNN
addestrato su dimostrazioni umane e un livello finale di sicurezza. Il modello
non riproduce rigidamente un giro registrato: usa esempi simili per correggere
in modo limitato la decisione della base, mentre il Safety Governor conserva
l'autorita' finale.

## Componenti

- `craizy_auto_v8.py`: pilota autonomo, analisi offline e validazione.
- `craizy_manual.py`: guida con DualShock 4 e raccolta transazionale dei giri.
- `snakeoil3_jm2.py`: client UDP per `scr_server 1`.
- `torcs_ps4_dataset.csv`: 10 giri completi, 37.726 campioni post-ADAS.
- `test_craizy_auto_v8.py`: test di policy, KNN, sicurezza, settori e cambio.
- `DOCUMENTAZIONE_PROGETTO.md`: relazione tecnica completa.

## Installazione

```powershell
pip install -r requirements.txt
```

TORCS deve essere configurato con `scr_server 1`, porta UDP `3001`, pista
Corkscrew e sensori pista abilitati.

## Utilizzo

Avviare prima la gara in TORCS, quindi:

```powershell
python craizy_auto_v8.py
```

Comandi disponibili:

```powershell
python craizy_auto_v8.py --base-only
python craizy_auto_v8.py --base-only --slow
python craizy_auto_v8.py --analyze-only
python craizy_auto_v8.py --validation-report --validation-runs 10
python -m unittest test_craizy_auto_v8.py
```

La modalita' normale usa il KNN. `--base-only` lo disattiva e lascia attivi
controllore sensoriale, Safety Governor e ADAS.

## Raccolta dati

```powershell
python craizy_manual.py
```

- stick sinistro: sterzo;
- `R2`: acceleratore;
- `L2`: freno;
- `SHARE` o `OPTIONS`: scarta il tentativo e riparte;
- `Ctrl+C`: termina senza salvare il giro incompleto.

Il dataset viene aggiornato solo alla conclusione di un giro valido. Uscite
persistenti, aumento del danno, restart e interruzioni eliminano tutte le
righe temporanee. Il salvataggio usa un file temporaneo e sostituzione
atomica, quindi un errore non lascia il CSV parzialmente scritto.

## Risultati

Il dataset consegnato contiene:

- 10 giri puliti;
- 37.726 campioni;
- 3.691-3.907 campioni per giro;
- tempo registrato tra `75,932 s` e `80,684 s`;
- velocita' massima osservata tra `265,731` e `273,462 km/h`.

La variante `v9r7_s05_projection` mantiene il dataset da dieci giri e
aggiunge una frenata preventiva locale tra 1935 e 1975 metri. Interviene
solo quando la posizione proiettata indica che l'auto sta gia'
convergendo oltre il bordo interno di S05; i passaggi stabili non
ricevono un limite di velocita' aggiuntivo.

Nella sessione di validazione autonoma registrata:

- 10 tentativi;
- 8 giri riconosciuti come completati;
- 7 giri completati senza uscita o recovery;
- affidabilita' complessiva `70%`;
- best lap `88,406 s`;
- mediana dei giri puliti `90,202 s`;
- deviazione standard `1,021 s`.

I risultati descrivono la configurazione TORCS usata durante il progetto e
non costituiscono una garanzia su piste, vetture o setup fisici differenti.
