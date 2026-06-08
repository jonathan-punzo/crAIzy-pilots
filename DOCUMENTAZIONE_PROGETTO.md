# V7 Behavioral Cloning e DAgger per TORCS

## Stato Corrente

La versione ufficiale è `craizy_auto_v7.py`. V6 resta congelata come
baseline KNN e non viene importata.

V7 segue l'impostazione delle slide del corso: lo stato sensoriale viene
mappato direttamente in azioni continue mediante supervised learning.
La rete PyTorch usa 25 input normalizzati, due layer nascosti da 128 e 64
neuroni con ReLU e due output `tanh`: sterzo e pedale firmato.

La prima prova runtime ha evidenziato una sensibilità eccessiva alle piccole
variazioni sensoriali: tra 25 e 55 km/h la dimostrazione usava sterzo compreso
tra 0 e 0,025, mentre il primo checkpoint arrivava a -0,70. Il training V7
robusto aggiunge perturbazioni limitate ai sensori. L'azione finale rispetta
inoltre i massimi incrementi per tick osservati nel controller post-ADAS,
impedendo salti non presenti nelle dimostrazioni.

Il training usa le azioni post-ADAS realmente inviate a TORCS. La loss MSE
è pesata per dare più importanza alle frenate e alle sterzate forti, che
sono meno frequenti ma decisive. `distFromStart` non è una feature.

La guida normale non contiene edge guard, brake guard, speed cap o altre
correzioni della traiettoria. Restano deterministici soltanto cambio,
clipping dei comandi e recovery d'emergenza.

Il problema del covariate shift viene affrontato con DAgger. In modalità
`--dagger` la rete guida normalmente; tenendo premuto `L1`, il pilota umano
prende il controllo. Gli stati visitati dalla rete e le azioni corrette
post-ADAS vengono salvati separatamente in `torcs_v7_dagger.csv`, senza
modificare il dataset originale. Il successivo `--train` usa entrambe le
fonti.

La prima valutazione leave-one-lap-out sui quattro giri produce:

| Modello | MAE sterzo | MAE pedale | MAE frenate |
|---|---:|---:|---:|
| V7 MLP robusta | 0,0583 | 0,0660 | 0,0982 |
| V6 KNN | 0,0677 | 0,1654 | 0,4272 |

L'inferenza V7 al 95 percentile richiede circa `0,04 ms`.

---

# Archivio: V6 ibrida sensoriale con behavioral cloning KNN per TORCS

## 1. Obiettivo

Il progetto realizza un driver autonomo Python per TORCS. La validazione
operativa avviene sul circuito Corkscrew, ma la versione corrente
`craizy_auto_v6.py` non usa coordinate della pista o settori codificati.
Le versioni V2-V5 restano esperimenti storici archiviati.

V6 è definibile come:

> controller autonomo ibrido basato su sensori, behavioral cloning con KNN
> regressivo e sistemi dinamici di stabilizzazione e recovery.

## Architettura V6 Corrente

La V6 abbandona sia il replay spaziale sia la base IBM. Il comportamento
normale viene appreso direttamente dalle dimostrazioni post-ADAS tramite
`KNeighborsRegressor`:

```text
stato sensoriale -> sterzo, pedale firmato
```

Lo stato contiene i 19 sensori pista, `trackPos`, `angle`, `speedX`,
`speedY`, `rpm` e la dispersione delle velocita' delle ruote. Le feature
sono normalizzate con scale fisiche fisse. `distFromStart` viene registrato
nei log, ma non entra nel modello: il controller non conosce il settore
della pista e deve reagire alla geometria e alla dinamica osservate.

Il pedale firmato vale `accel_action - brake_action`. Un valore positivo
produce acceleratore, uno negativo produce freno; i due comandi non possono
quindi essere significativi nello stesso tick. La marcia resta
deterministica e usa soglie con isteresi.

La fiducia del KNN e' ricavata dalla distanza media dei sette vicini. Vale
uno fino a `0,263`, decade linearmente e diventa zero a `0,697`. Fuori
distribuzione entra gradualmente una policy sensoriale prudente, non un
altro pilota storico.

Il safety controller non pianifica la traiettoria. Limita soltanto la
variazione dello sterzo a `0,04` e quella del pedale a `0,16` per tick,
riduce il gas in presenza di slittamento e applica una correzione moderata
quando posizione, angolo o velocita' laterale indicano perdita di stabilita'.
Sui rettilinei aperti oltre `100 km/h` lo sterzo usa uno smoothing piu'
forte per contrastare l'oscillazione tra vicini KNN.

Dopo la prima prova TORCS della V6 e' emerso un limite specifico della
regressione del pedale: stati sensoriali simili possono appartenere sia a un
rettilineo sia all'avvicinamento alla frenata, e la maggioranza dei campioni
di accelerazione puo' cancellare i pochi campioni di freno. Il predictive
brake guard calcola quindi una velocita' sicura dalla visibilita' centrale e
dalla sua tendenza di chiusura. Si attiva soltanto oltre `220 km/h`, con
sovravelocita' prevista di almeno `10 km/h` e una transizione verificata da
pista aperta a pista in chiusura. Non usa `distFromStart`.
La prima taratura arrestava correttamente l'accelerazione, ma il KNN
continuava poi a frenare fino a `69 km/h`, mentre le dimostrazioni passavano
lo stesso riferimento tra `126` e `144 km/h`. Il guard conserva quindi un
contesto temporaneo: raggiunta la velocita' sicura, annulla la frenata
residua del KNN e lascia scorrere la vettura.

Nella prova successiva l'ingresso curva e' rimasto stabile, ma in uscita il
KNN ha mantenuto sterzo circa `+0,60` mentre `trackPos` attraversava la pista
da `+0,44` a `-0,92`. Un edge guard proietta `trackPos` a 18 tick usando la
deriva osservata. Si attiva solo con `abs(steer) >= 0,55`, movimento verso
l'esterno e bordo previsto oltre `0,92`; riduce prima lo sterzo e applica
controsterzo soltanto oltre `0,82`. Non impone il centro pista e conserva
quindi le racing line professionali vicine al bordo.

Le prove successive mostrano che la V6 iniziava la prima sterzata a circa
`140 km/h`; nelle quattro dimostrazioni lo stesso tratto e' normalmente
percorso tra `106` e `125 km/h`, con un solo giro vicino a `142 km/h`.
Il dataset non viene alterato. La policy applica un massimo operativo di
`215 km/h` e riduce al 90% la velocita' sicura stimata dalla geometria
visibile. Si conserva quindi la traiettoria appresa rallentandone
coerentemente l'esecuzione.

Il recovery usa gli stati `NORMAL`, `STABILIZE`, `REVERSE`,
`FORWARD_ALIGN` e `REJOIN`. Ogni transizione ha durata e condizioni
esplicite, impedendo l'alternanza rapida tra prima e retromarcia. Il ritorno
alla policy appresa avviene con un blend progressivo.

La valutazione leave-one-lap-out sui quattro giri, 15.409 campioni, produce
MAE `0,0673` sullo sterzo e `0,1553` sul pedale firmato. Il 95 percentile
del tempo di inferenza e' circa `1,37 ms`, sotto il gate di `5 ms`.

## 2. Dataset

`craizy_manual.py` registra:

- intenzione del joystick;
- azione prodotta dagli ADAS e realmente inviata a TORCS;
- sensori e posizione;
- `run_id`, step e tempo giro.

V3, V4 e V5 usano le azioni post-ADAS. In V5 esse diventano il target della
regressione residuale rispetto al pilota deterministico IBM.

Il dataset operativo contiene esclusivamente giri completi. SHARE e OPTIONS
scartano l'intero tentativo e riavviano la gara. I prefissi antecedenti a una
sbandata non vengono conservati: un rollback temporale fisso non garantisce
che siano assenti le condizioni dinamiche che hanno causato l'errore.

Sono accettati soltanto giri con almeno 800 campioni, distanza di almeno
3500 m e danno invariato. L'uscita pista viene confermata dopo 3 tick
consecutivi con `abs(trackPos) >= 1` oppure sensori pista negativi. Servono
almeno tre giri. La lunghezza pista è la mediana delle lunghezze accettate;
un giro distante oltre 10 m dalla mediana viene escluso.

## 3. Launch E Profilo Operativo Da 5 Metri

Ogni registrazione parte poco prima del traguardo. V4 individua il primo
passaggio da `distFromStart > 3000` a `distFromStart < 500` e separa:

- `launch`: partenza da fermo fino al primo attraversamento;
- `racing`: giro successivo, dal primo al secondo attraversamento.

Il launch del giro migliore viene riprodotto nel tempo senza feedback
ordinario. La fase racing viene interpolata sulla griglia da 5 m. Con una
pista di circa 3600 m il profilo contiene circa 720 punti. Il lookup non
collega circolarmente arrivo e partenza: vicino a 3600 m conserva l'arrivo
ad alta velocità, mentre dopo il primo wrap usa l'inizio racing.

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

In V4 la danger map resta descrittiva e modula solo la prudenza dei sistemi
di sicurezza. Non aggiunge freno e non sostituisce i comandi post-ADAS.

## 6. Evoluzione V4

Il trace V3 del 7 giugno 2026 mostra il primo evento critico attorno a
`954,5 m`: velocità `202,2 km/h`, `trackPos=0,994` e ingresso del recovery
mentre l'auto era ancora appena dentro il limite pista. Il recovery cambiava
sterzo e pedali durante una dinamica ad alta velocità. Inoltre la marcia
calcolata dal recovery non veniva copiata nell'azione finale: durante uno
stuck il replay poteva quindi restare in quinta o sesta invece di usare la
retro.

V4 corregge il comportamento con quattro decisioni:

- azioni e feedback usano entrambi lo stato del giro migliore;
- mediana, MAD e danger restano strumenti di analisi e affidabilità;
- una stability guard riduce solo il gas quando cresce la deviazione;
- il recovery entra dopo tre tick fuori pista o cinque tick di dinamica
  laterale grave, salvo casi estremi.

Il limite della correzione sterzo è `0,06` sotto `100 km/h` e scende
linearmente fino a `0,025` a `200 km/h`. A velocità superiori a `140 km/h`
il recovery inizialmente lascia l'auto scorrere senza aggiungere una frenata
brusca. La marcia recovery viene sempre inviata quando il blend è attivo,
inclusa la retro nella procedura di sblocco.

## 7. Evoluzione V5

L'archivio `IBM-TORCs-main.zip`, fornito dai professori, contiene più piloti.
Il README identifica `fastest.py` come versione migliore per la nuova
vettura F1 e `v1.py` come versione legacy per la vecchia auto. Il file
`snakeoil3_jm2.py` dell'archivio è identico byte per byte a quello già usato
dal progetto.

V5 incorpora direttamente la logica di `fastest.py`, senza importare file
esterni o altri piloti. La base interpreta sensori pista, angolo,
`trackPos`, velocità e slittamento delle ruote per produrre sterzo,
acceleratore, freno e marcia. Sono corretti:

- divisore errato nelle medie dei nove sensori laterali;
- riduzione gas asimmetrica, ora basata su `abs(steer)`;
- presenza contemporanea di gas e freno significativi.

La base IBM resta sempre disponibile come fallback completo.

## 8. KNN Regressivo Residuale

V5 usa esclusivamente `KNeighborsRegressor`, con:

```text
k = 7
weights = distance
metric = euclidean
```

Il modello non predice la marcia e non classifica modalità discrete. I tre
target continui sono:

```text
delta_steer = steer_action - steer_base
delta_accel = accel_action - accel_base
delta_brake = brake_action - brake_base
```

La policy IBM viene simulata in ordine temporale anche durante la costruzione
del dataset, perché il suo acceleratore dipende dal comando precedente.

I quattro giri vengono divisi al primo passaggio
`distFromStart > 3000 -> distFromStart < 500`. Sono creati due regressori:

- launch: tempo dalla partenza, stato vettura, sensori frontali e azione IBM;
- racing: seno e coseno della posizione circolare, stato vettura, sensori
  frontali e azione IBM.

Tutte le feature sono normalizzate con scale fisiche fisse. La confidenza è
calcolata dalla distanza media dei sette vicini: vale uno sui campioni esatti
e decresce fino a zero fuori distribuzione. A confidenza zero l'output è
numericamente uguale alla base IBM. Il recovery sostituisce entrambi soltanto
in caso di uscita confermata, deviazione estrema o auto bloccata.

Nel launch alcuni vettori provenienti da giri diversi sono identici ma hanno
sterzi professionali differenti di pochi millesimi. In questi casi un
regressore deterministico restituisce la media pesata; l'errore massimo
misurato sui campioni esatti è `0,00167`.

## 9. Output E Valutazione

`python craizy_auto_v5.py --analyze-only` genera:

- `corkscrew_v5_loo.csv`;
- `corkscrew_v5_model_summary.csv`.

La valutazione leave-one-lap-out sui `15.409` campioni produce:

| Comando | MAE IBM | MAE IBM + KNN |
|---|---:|---:|
| Sterzo | 0,5931 | 0,1440 |
| Acceleratore | 0,4735 | 0,2098 |
| Freno | 0,0911 | 0,0369 |

Il 95° percentile dell'inferenza è circa `0,76 ms` nell'ultima analisi, sotto
il gate di `5 ms`.
Durante TORCS, `logs/auto_v5_trace_*.csv` registra azione IBM, residuo,
confidenza, azione appresa, azione finale, fase e recovery.

### Stabilizzazione Dell'8 Giugno 2026

La prima prova V5 è registrata in
`auto_v5_trace_20260607_233041_0761300.csv`. La deviazione nasce prima
dell'uscita, tra `380` e `405 m`: la confidenza KNN cala mentre la base IBM è
saturata, e lo sterzo cambia fino a `0,21` per tick. Nel dataset professionale
il 99° percentile della variazione è invece `0,04`.

La correzione mantiene il KNN regressivo ma aggiunge un safety envelope
spiegabile:

- profilo spaziale mediano di `trackPos`, angolo, `speedY`, velocità e sterzo;
- feedback limitato verso la traiettoria dimostrata;
- blend progressivo verso lo sterzo professionale quando cresce l'errore;
- rate limit dello sterzo pari a `0,04` per tick;
- riduzione gas e frenata massima `0,12` soltanto in deviazione;
- nessun acceleratore nel recovery con angolo o velocità laterale instabili.

Sul replay della telemetria critica, tra `380` e `393 m`, il vecchio comando
passava da `+0,162` a `+0,353`; il comando stabilizzato resta circa tra
`-0,033` e `-0,010`, opponendosi all'errore di linea senza salti.

## 10. Metodo Sperimentale

1. eseguire `--analyze-only` e verificare regressione e latenza;
2. completare un giro con `--base-only`;
3. completare un giro V5 con zero danni e zero uscite;
4. completare tre giri V5 consecutivi puliti;
5. confrontare il tempo V5 con la baseline IBM;
6. congelare V5 soltanto se non è peggiore della baseline.

### Configurazione Fisica TORCS

L'8 giugno 2026 e' stata analizzata l'installazione effettivamente avviata
dal collegamento `wtorcs`. La gara usa Corkscrew, un solo giro e
`scr_server 1` con la monoposto `car1-ow1`. I file `0/default.xml` e
`1/default.xml` di `scr_server` sono identici.

Il setup attivo privilegia la reattivita':

- carburante iniziale: 100 litri;
- sterzo: lock di 21 gradi e velocita' massima di 360 gradi/s;
- ali: 13 gradi anteriore e 10 gradi posteriore;
- trazione posteriore con differenziale limited slip;
- ripartizione frenante anteriore: 0,47;
- barre antirollio anteriori e posteriori: 0;
- molle: 2000 lbs/in davanti e 3000 lbs/in dietro.

Lo sterzo fisico potrebbe attraversare l'intero campo da `-1` a `+1` in
circa 0,117 secondi. La V5 limita quindi la velocita' del comando in software.
L'asfalto di Corkscrew ha attrito 1,1 e resistenza 0,015; la terra passa a
0,9 e 0,1. Una piccola oscillazione nata sull'asfalto viene percio'
amplificata rapidamente quando una ruota esce.

La configurazione TORCS non viene modificata: fisica, vettura e pista restano
invarianti rispetto al dataset professionale. Questi valori sono riportati
soltanto per interpretare la dinamica del veicolo e tarare il controller
Python.

## 11. Cronologia

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
- separata la partenza da fermo dal giro racing per evitare la sovrapposizione
  dei campioni attorno al traguardo;
- verificati sperimentalmente i segni del feedback e del recovery rispetto
  alle coordinate TORCS e alla risposta fisica della vettura;
- aggiunti deadband, conferma temporale e limite di variazione dello sterzo;
- sostituito il trace sovrascritto con log timestampati.
- analizzata la prima sbandata della V3 attorno a 955 m;
- creata V4 come pilota autonomo indipendente, senza importare V3;
- allineato il feedback allo stato del giro migliore;
- aggiunte stability guard, recovery confermato e marcia recovery effettiva.
- letto e verificato l'archivio IBM fornito dai professori;
- scelto `fastest.py` come base F1 e confermato il client snakeoil identico;
- creata V5 indipendente con policy IBM corretta;
- introdotti due `KNeighborsRegressor` residuali per launch e racing;
- aggiunti fallback per confidenza, recovery prioritario e logging V5;
- verificato offline il miglioramento su sterzo, gas e freno e la latenza
  inferiore a `5 ms`.
- analizzata la prima run V5 e localizzata la deviazione tra 380 e 415 m;
- aggiunti riferimento multigiro, feedback limitato e rate limit misurato;
- modificato il recovery per stabilizzare la vettura prima di accelerare.
