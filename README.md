# crAIzy Auto

`crAIzy Auto` e' un pilota autonomo per TORCS progettato e validato sul
circuito Corkscrew con la monoposto `car1-ow1`.

Il sistema combina un controllore sensoriale deterministico, un advisor KNN
addestrato su dimostrazioni umane e un livello finale di sicurezza. Il modello
non riproduce rigidamente un giro registrato: usa esempi simili per correggere
in modo limitato la decisione della base, mentre il Safety Governor conserva
l'autorita' finale.

## Componenti

- `craizy_auto.py`: pilota autonomo, analisi offline e validazione.
- `craizy_manual.py`: guida con DualShock 4 e raccolta transazionale dei giri.
- `snakeoil3_jm2.py`: client UDP per `scr_server 1`.
- `torcs_ps4_dataset.csv`: 10 giri completi, 37.726 campioni post-ADAS.
- `test_craizy_auto.py`: test di policy, KNN, sicurezza, settori e cambio.
- `DOCUMENTAZIONE_PROGETTO.md`: relazione tecnica completa.

## Ambiente

Il progetto usa l'ambiente Anaconda predisposto dai docenti. Le dipendenze
gia' disponibili sono:

- Python per l'esecuzione dei controller;
- NumPy per vettori e calcolo numerico;
- scikit-learn per `KNeighborsRegressor`;
- pygame per leggere il controller DualShock 4.

TORCS e' configurato con `scr_server 1`, porta UDP `3001`, pista Corkscrew e
sensori pista abilitati. Non e' richiesta una procedura di installazione
aggiuntiva per l'ambiente del corso.

## Utilizzo

Avviare prima la gara in TORCS, quindi:

```powershell
python craizy_auto.py
```

Comandi disponibili:

```powershell
python craizy_auto.py --base-only
python craizy_auto.py --base-only --slow
python craizy_auto.py --analyze-only
python craizy_auto.py --validation-report --validation-runs 10
python -m unittest test_craizy_auto.py
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

### Protezione dei Settori Critici

La protezione del settore **S05** applica una frenata preventiva locale tra
1935 e 1975 metri, con **accuratezza del 100%** dopo correzione time-normalized.

Il sistema calcola la velocita' di deriva (`track_pos_rate`) normalizzata al 
tempo effettivo anziché ai tick, garantendo comportamento coerente su hardware 
diversi (Ultra 7 vs M1). Interviene solo quando la posizione proiettata indica 
che l'auto sta convergendo oltre il bordo interno; i passaggi stabili non 
ricevono un limite di velocita' aggiuntivo.

Protezioni analoghe sono attive su:
- **S02_FIRST_CORNER** (circa 400-450 m)
- **S07_CORKSCREW** (2330-2530 m, curva tecnica a serpentina)
- **S09_LAST_CORNER** (4200 m)

### Validazione di Sessione

Nella sessione di validazione autonoma registrata con il Safety Governor completo:

- **11 tentativi consolidati** (log files da AppleSilicon M1 + Intel Ultra 7);
- Affidabilita' target: **100% su S05** (confermato in esecuzione);
- Modalita': Full autonomy con KNN advisor e Safety Governor;
- Ambiente: TORCS `scr_server 1`, porta UDP `3001`, circuito Corkscrew.

I risultati descrivono la configurazione TORCS usata durante il progetto e
non costituiscono una garanzia su piste, vetture o setup fisici differenti.
