# crAIzy pilots - TORCS AI Racing League

Repository ufficiale del progetto **crAIzy pilots**, sviluppato per la IBM AI Racing League / UNISA nell'ambito del percorso IBM SkillsBuild.

Il progetto consiste in un AI driver Python per TORCS, pensato per ottenere il miglior giro possibile sul circuito Corkscrew rispettando le linee guida della competizione: codice IA modificabile, partenza da fermo, uso documentato di IBM Granite / SkillsBuild, repository accessibile e materiali di submission.

## Team

**Nome gruppo:** crAIzy pilots

| Nome | Ruolo |
| --- | --- |
| Jonathan Punzo | Portavoce del gruppo |
| Felice Iandoli | Membro del team |
| Andrea Botta | Membro del team |
| Mariagiusy Cientanni | Membro del team |
| Simona Ravotti | Membro del team |

## Requisiti di gara

- TORCS versione Python.
- Circuito ufficiale: Corkscrew.
- Giro valido da partenza ferma.
- Driver TORCS: `scr_server 1`.
- Modifiche limitate al codice Python dell'IA.
- Video giro veloce, video team in inglese max 3 minuti, repository, certificati SkillsBuild e submit finale.

## Struttura

```text
src/
  torcs_jm_par.py        # entrypoint per TORCS
  manual_control.py      # controller manuale
  torcs_client.py        # client UDP compatibile SnakeOil/SCR
  driver/                # logica modulare del pilota
    config.py            # parametri gara hardcoded in Python
  training/              # behavioral cloning e valutazione
  utils/                 # report, plot e riepiloghi
docs/                    # strategia, Granite log, checklist, video/blog
data/                    # dataset manuali o telemetry esportata
results/                 # log CSV e report locali
media/                   # video, screenshot, asset team
tests/                   # test unitari del controller
```

## Avvio automatico

1. Apri TORCS.
2. Vai su `Race -> Practice` o `Race -> Quick Race -> Configure Race`.
3. Seleziona Corkscrew.
4. Seleziona `scr_server 1` come driver.
5. Avvia `New Race`, lasciando TORCS in attesa del client.
6. Da PowerShell:

```powershell
py .\src\torcs_jm_par.py --port 3001
```

Oppure:

```powershell
.\scripts\run_race.ps1
```

## Guida manuale

Avvia TORCS nello stesso modo, poi:

```powershell
py .\src\manual_control.py --port 3001
```

Oppure:

```powershell
.\scripts\run_manual.ps1
```

Comandi:

- `W` o freccia su: accelera.
- `S`, freccia giu' o spazio: frena.
- `A`/freccia sinistra e `D`/freccia destra: sterza.
- `Q`/`E`: scala/aumenta marcia e disattiva cambio automatico.
- `G`: attiva/disattiva cambio automatico.
- `X`: attiva/disattiva stabilizzazione leggera.
- `R`: retromarcia.
- `Esc`: esci.

I log automatici vengono scritti in `results/runs/`; i log manuali in `results/manual_runs/`, salvo uso di `--no-log`.

## Workflow

1. Usa `src/driver/config.py` come unico punto di tuning del pilota da gara.
2. Usa `src/manual_control.py` per provare il tracciato e raccogliere dati.
3. Registra ogni prova in `docs/test_report.md`.
4. Documenta l'uso di IBM Granite in `docs/granite_usage_log.md`.
5. Usa `src/training/behavioral_cloning.py` solo dopo aver raccolto dataset puliti.

## Verifica locale

```powershell
py -m unittest discover -s tests
```

Oppure:

```powershell
.\scripts\run_tests.ps1
```
