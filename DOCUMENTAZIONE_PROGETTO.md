# V3 multigiro post-ADAS per TORCS

## 1. Obiettivo

Il progetto realizza un driver autonomo Python per TORCS, specializzato sul
circuito Corkscrew. La priorità è riprodurre una guida professionale stabile
prima di introdurre KNN o reti neurali.

La versione ufficiale è `craizy_auto_v3.py`, definibile come:

> controller autonomo basato su replay post-ADAS, profilo spaziale multigiro,
> racing line robusta e sistemi dinamici di sicurezza.

## 2. Dataset

`craizy_manual.py` registra:

- intenzione del joystick;
- azione prodotta dagli ADAS e realmente inviata a TORCS;
- sensori e posizione;
- `run_id`, step e tempo giro.

V3 usa le azioni post-ADAS. Non applica nuovamente smoothing, ABS, traction
control o limiti sterzo, evitando il doppio filtro.

Il dataset operativo contiene esclusivamente giri completi. SHARE e OPTIONS
scartano l'intero tentativo e riavviano la gara. I prefissi antecedenti a una
sbandata non vengono conservati: un rollback temporale fisso non garantisce
che siano assenti le condizioni dinamiche che hanno causato l'errore.

Sono accettati soltanto giri con almeno 800 campioni, distanza di almeno
3500 m, danno zero, `abs(trackPos) < 1` e sensori pista non negativi. Servono
almeno tre giri. La lunghezza pista è la mediana delle lunghezze accettate;
un giro distante oltre 10 m dalla mediana viene escluso.

## 3. Profilo Operativo Da 5 Metri

Ogni giro viene interpolato sulla stessa griglia circolare da 5 m. Con una
pista di circa 3600 m il profilo contiene circa 720 punti.

Per ciascun punto:

- gas, freno, sterzo e marcia provengono dal giro pulito più veloce;
- `speedX`, `speedY`, `angle` e `trackPos` sono analizzati su tutti i giri;
- mediana e MAD descrivono consenso e dispersione;
- la racing line combina mediana multigiro e stato del giro migliore.

Il peso del consenso usa:

```text
track_consensus = clamp(1 - trackPos_MAD / 0,15)
angle_consensus = clamp(1 - angle_MAD / 0,10)
```

L'affidabilità include anche MAD della velocità longitudinale e laterale.
Quando i giri discordano, il target si avvicina al giro migliore e la
correzione automatica diventa più debole.

## 4. Zone Di Frenata

Le frenate vengono rilevate prima del ricampionamento:

- inizio: `brake_action > 0,10` per almeno 5 campioni;
- fine: `brake_action < 0,05` per almeno 5 campioni;
- intervalli separati da meno di 10 m vengono uniti.

Per ogni zona vengono esportati supporto tra i giri, velocità iniziale e
minima, freno massimo e medio e riduzione della velocità. La mappa è
descrittiva e non aggiunge frenate al pilota.

## 5. Danger Map Da 20 Metri

Ogni settore usa il 95° percentile dei segnali:

```text
danger =
    0,25 * brake
  + 0,20 * abs(steer)
  + 0,15 * abs(speedY)
  + 0,15 * abs(angle)
  + 0,15 * vicinanza_bordo
  + 0,10 * decelerazione
```

Classificazione:

- sotto 0,25: bassa;
- sotto 0,50: media;
- sotto 0,75: alta;
- da 0,75: critica.

La danger map non modifica i pedali nel replay. Riduce il limite della
correzione sterzo da 0,12 verso 0,08 e anticipa fino al 20% le soglie di
recovery. L'ingresso recovery cresce gradualmente da 0,08 a massimo 0,12 per
tick; l'uscita resta graduale e usa isteresi. Le soglie assolute di
`trackPos` non possono diventare più interne della racing line dimostrata:
una traiettoria professionale vicina al bordo non viene quindi scambiata per
un'uscita pista.

## 6. Output Analitici

`python craizy_auto_v3.py --analyze-only` genera:

- `corkscrew_profile_5m.csv`;
- `corkscrew_danger_20m.csv`;
- `corkscrew_braking_zones.csv`;
- `corkscrew_lap_comparison.csv`.

Il profilo esporta anche distanza, angolo, velocità e comandi normalizzati.
Queste colonne sono derivate: il dataset originale conserva sempre i valori
TORCS reali.

Durante la guida `auto_v3_trace.csv` registra bin, settore danger, racing
line, MAD, affidabilità, soglie recovery, azioni registrate e azioni finali.
`auto_v3_runs.csv` riassume velocità, danno, uscite, recovery, danger ed errore
dalla racing line.

## 7. Metodo Sperimentale

1. raccogliere almeno tre giri professionali puliti;
2. generare e ispezionare i quattro report;
3. verificare un giro completo senza danni né uscite;
4. verificare tre giri consecutivi puliti;
5. congelare V3;
6. valutare KNN soltanto in una V4 separata.

## 8. Cronologia

### 6 giugno 2026

- distinto il comando intenzionale dal comando post-ADAS;
- archiviato il dataset pre-ADAS;
- riscritta V3 per riprodurre le azioni realmente inviate a TORCS;
- eliminato il controller deterministico separato.

### 7 giugno 2026

- introdotto profilo operativo circolare da 5 m;
- utilizzati tutti i giri puliti per racing line, MAD e affidabilità;
- mantenute le azioni del giro migliore;
- aggiunte zone di frenata e danger map da 20 m;
- collegata la danger map esclusivamente a feedback e recovery;
- aggiunti modalità `--analyze-only`, report normalizzati e logging esteso.
- eliminati definitivamente i dataset legacy pre-ADAS;
- creato un nuovo `torcs_ps4_dataset.csv` vuoto con lo schema post-ADAS
  definitivo, pronto per tre nuove registrazioni professionali.
- eliminata la raccolta di segmenti parziali: il profilo multigiro utilizza
  esclusivamente dimostrazioni complete e pulite.
