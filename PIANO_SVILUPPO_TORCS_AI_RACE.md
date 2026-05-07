# Piano di sviluppo - TORCS AI Racing League

Data di riferimento: 5 maggio 2026.

Questo piano e' costruito sui PDF forniti: regolamento/submission IBM AI Racing League, setup TORCS, slide UNISA step-by-step e lezioni sul client TORCS Python. Tutte le voci indicate come opzionali nei materiali vengono trattate qui come must-have, pur restando aderenti alle linee guida delle slide.

## 1. Obiettivo del progetto

Realizzare un AI driver Python per TORCS capace di completare il circuito Corkscrew da partenza ferma con tempo competitivo, comportamento stabile e documentazione chiara dell'approccio tecnico, dell'uso di IBM Granite / SkillsBuild e del lavoro di squadra.

Il progetto non deve essere solo una macchina veloce: deve dimostrare capacita' tecnica, comunicazione, collaborazione e rispetto delle regole della competizione.

## 2. Vincoli non negoziabili

- Il progetto deve usare TORCS versione Python.
- Il circuito ufficiale e' Corkscrew.
- Il giro valido per la presentazione deve partire da fermo.
- E' consentito modificare solo il codice Python dell'IA.
- Non si devono alterare dinamiche interne del simulatore, asset ufficiali o livree ufficiali.
- Il driver TORCS deve essere `scr_server 1`.
- Il repository GitHub deve essere accessibile e contenere il codice dell'AI driver.
- Ogni membro deve completare e raccogliere i certificati/badge SkillsBuild richiesti:
  - `Introduction to IBM AI Racing Competition`
  - `IBM Granite Models for Software Development`
  - `SUBMIT TO WIN!`
- Il video tecnico di team deve essere in inglese e durare massimo 3 minuti.
- Il video del giro veloce deve mostrare chiaramente team, universita' e giro su Corkscrew.
- Il team deve avere nome coerente su SkillsBuild, form Race, repository e materiali.
- Un solo rappresentante effettua il submit finale, ma tutti i materiali devono essere pronti e condivisi.

## 3. Must-have derivati dalle parti opzionali

- Aggiungere altri driver in TORCS durante i test, oltre a `scr_server 1`, per verificare stabilita' e gestione degli avversari.
- Creare un blog/report pubblico del percorso del team, con link a video giro, video team e repository.
- Creare una presenza social minima del team, coerente con la comunicazione della race.
- Preparare una livrea/loghi separati in RGB per team e universita', senza logo IBM, pronta per eventuali gare con partenza di massa.
- Inserire nella presentazione certificati anche eventuali corsi SkillsBuild aggiuntivi utili al progetto.
- Usare IBM Granite non solo come voce narrativa, ma come parte documentata del workflow: supporto a ideazione, review codice, spiegazione tecnica, tuning ragionato e preparazione contenuti.
- Creare Drive di team e GitHub di team fin dall'inizio, non a fine progetto.
- Documentare prove, tempi, parametri e decisioni tecniche in modo continuo.

## 4. Architettura tecnica proposta

### 4.1 Struttura repository

Repository consigliato:

```text
torcs-ai-race/
  README.md
  requirements.txt
  environment.yml
  src/
    torcs_jm_par.py
    driver/
      controller.py
      speed_planner.py
      steering.py
      traction.py
      gears.py
      recovery.py
      telemetry.py
      config.py
    training/
      collect_manual_data.py
      behavioral_cloning.py
      evaluate_model.py
    utils/
      lap_logger.py
      plots.py
  data/
    README.md
  results/
    README.md
  docs/
    granite_usage_log.md
    technical_strategy.md
    test_report.md
    submission_checklist.md
    certificates_links.md
  media/
    README.md
```

### 4.2 Componenti del driver

- `controller.py`: orchestration del ciclo `sensori -> decisione -> azioni`.
- `steering.py`: controllo sterzo basato su `angle`, `trackPos` e vettore `track`.
- `speed_planner.py`: velocita' target dinamica in base alla curvatura stimata dai sensori `track`.
- `traction.py`: controllo slittamento basato su `wheelSpinVel`, riduzione accelerazione e gestione stabilita'.
- `gears.py`: cambio marcia automatico usando `speedX`, `rpm` e soglie configurabili.
- `recovery.py`: recupero da fuori pista, testacoda, stallo o valori sensore non affidabili.
- `telemetry.py`: logging di sensori, azioni, tempi, danno, uscite pista e parametri.
- `config.py`: unico punto Python per `TARGET_SPEED`, `STEER_GAIN`, `CENTERING_GAIN`, `BRAKE_THRESHOLD`, `GEAR_SPEEDS`, traction control e modalita' test.
- `manual_control.py`: guida manuale per prove, debug e raccolta dataset.

## 5. Sensori e azioni da usare

Input principali:

- `angle`: allineamento auto-tracciato.
- `track`: 19 sensori frontali per distanza dal bordo pista.
- `trackPos`: distanza normalizzata dal centro pista.
- `speedX`, `speedY`, `speedZ`: velocita' longitudinale, laterale e verticale.
- `wheelSpinVel`: slittamento ruote.
- `rpm`, `gear`: gestione cambio.
- `damage`: metrica di sicurezza.
- `curLapTime`, `lastLapTime`, `distFromStart`, `distRaced`: valutazione prestazioni.
- `opponents`, `racePos`: test con altri driver.
- `z`: controllo stabilita' e condizioni anomale.

Output controllati:

- `steer`: sterzo limitato nel range ammesso.
- `accel`: accelerazione progressiva.
- `brake`: frenata predittiva e anti-bloccaggio semplice.
- `gear`: cambio automatico.
- `clutch`: default stabile salvo necessita' specifica.
- `meta`: non usare per reset durante run ufficiali.

## 6. Roadmap di sviluppo

### Fase 0 - Setup e governance (5-8 maggio)

Obiettivo: ambiente riproducibile e team operativo.

Attivita':

- Estrarre TORCS in `C:\torcs\` con struttura `gym_torcs` e `torcs`.
- Configurare Python 3.11.x con ambiente `torcs-env`.
- Installare dipendenze: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`, `gymnasium`, `torch`, `tensorflow`, `keras`.
- Verificare avvio TORCS: `Race -> Practice/Quick Race -> Configure Race`.
- Impostare `scr_server 1`.
- Verificare esecuzione `torcs_jm_par.py` prima dell'avvio gara.
- Creare GitHub di team, Drive di team e cartella certificati.
- Definire ruoli:
  - AI/control engineer
  - telemetry/evaluation engineer
  - ML/behavioral cloning engineer
  - integration/release engineer
  - communication/marketing lead
- Registrare decisioni in `docs/technical_strategy.md`.

Definition of Done:

- TORCS parte e il client Python controlla l'auto.
- Il repo contiene README iniziale, istruzioni setup e struttura progetto.
- Tutti i membri hanno accesso a GitHub e Drive.

### Fase 1 - Baseline guidabile (9-14 maggio)

Obiettivo: completare giri puliti su Corkscrew con log automatici.

Attivita':

- Modularizzare `drive_modular()`.
- Implementare controllo sterzo:
  - combinazione di `angle` e `trackPos`
  - clipping del comando
  - smoothing per evitare oscillazioni.
- Implementare velocita' target dinamica:
  - velocita' piu' alta sui rettilinei
  - frenata anticipata in curve strette
  - uso dei sensori `track` centrali/laterali.
- Implementare traction control sempre attivo.
- Implementare cambio marcia automatico.
- Salvare log CSV per ogni run:
  - parametri
  - lap time
  - danno
  - uscite pista
  - numero recuperi
  - best/mean/std lap time.

Definition of Done:

- Almeno 5 giri consecutivi completati su Corkscrew.
- Nessun reset manuale.
- Report iniziale in `docs/test_report.md`.

### Fase 2 - Robustezza e recovery (15-22 maggio)

Obiettivo: rendere l'auto affidabile, non solo veloce.

Attivita':

- Gestire fuori pista: `abs(trackPos) > 1`, sensori `track = -1`, bassa velocita', angolo elevato.
- Implementare recovery:
  - riduzione accelerazione
  - sterzo verso centro pista
  - retromarcia controllata se bloccata
  - ripartenza progressiva.
- Ridurre oscillazioni laterali con smoothing e limiti velocita'-sterzo.
- Usare `damage` come metrica di penalizzazione interna.
- Testare con altri driver aggiunti alla gara.
- Usare sensori `opponents` per evitare collisioni semplici durante test multi-driver.

Definition of Done:

- 10 giri consecutivi completati su Corkscrew.
- Run multi-driver completata senza crash gravi.
- Recovery documentato con esempi nei log.

### Fase 3 - Ottimizzazione prestazioni (23 maggio-5 giugno)

Obiettivo: migliorare il tempo sul giro mantenendo stabilita'.

Attivita':

- Eseguire tuning sistematico dei parametri:
  - `TARGET_SPEED`
  - `STEER_GAIN`
  - `CENTERING_GAIN`
  - `BRAKE_THRESHOLD`
  - `GEAR_SPEEDS`
  - soglie traction control.
- Mantenere una sola configurazione da gara direttamente in Python:
  - `src/driver/config.py`: massima prestazione sul giro secco.
- Mantenere un solo flusso manuale:
  - `manual_control.py`: guida manuale per test e raccolta dati.
- Generare grafici:
  - tempo giro per run
  - velocita' lungo il tracciato
  - sterzo/freno/accelerazione
  - trackPos nel tempo
  - danno e uscite pista.
- Usare IBM Granite per:
  - analizzare log e pattern di errore
  - proporre modifiche motivate ai parametri
  - fare review del codice e della spiegazione tecnica.

Definition of Done:

- Miglior tempo misurato e ripetibile.
- `src/driver/config.py` completa giri puliti e produce il miglior tempo ripetibile.
- `docs/granite_usage_log.md` contiene prompt, output sintetico e decisioni prese.

### Fase 4 - Behavioral cloning obbligatorio (6-13 giugno)

Obiettivo: includere una componente AI/ML coerente con le lezioni, anche se il controller finale resta ibrido.

Attivita':

- Raccogliere dataset da guida manuale:
  - feature: `track`, `trackPos`, `angle`, `speedX`, `speedY`, `wheelSpinVel`
  - label: sterzo, velocita' target, freno/accelerazione.
- Addestrare almeno un modello:
  - baseline K-NN o regressore scikit-learn
  - MLP/NN se il dataset e i tempi lo consentono.
- Valutare rischio di data distribution mismatch.
- Confrontare:
  - controller rule-based
  - behavioral cloning puro
  - controller ibrido: ML per velocita'/sterzo suggeriti, safety layer rule-based.
- Integrare solo componenti ML che migliorano stabilita' o tempo senza introdurre instabilita'.

Definition of Done:

- Notebook/script riproducibile per training e valutazione.
- Risultati comparativi nel test report.
- Decisione motivata sul ruolo del modello nel driver finale.

### Fase 5 - Freeze tecnico e materiali (14-20 giugno)

Obiettivo: congelare il driver e preparare materiali ufficiali.

Attivita':

- Freeze dei parametri in `src/driver/config.py`.
- Pulizia repository:
  - README completo
  - istruzioni setup
  - istruzioni run
  - spiegazione architettura
  - link video/materiali
  - licenze/crediti se necessari.
- Verificare che nessun file modifichi simulatori, fisica o asset ufficiali.
- Creare video giro veloce:
  - Corkscrew
  - partenza da fermo
  - nome team e universita' sempre chiari.
- Preparare video team in inglese, max 3 minuti:
  - chi siamo
  - ruoli
  - strategia IA
  - uso IBM Granite
  - uso SkillsBuild
  - risultati e metriche.
- Preparare blog/report pubblico.
- Preparare slide certificati uniche per tutto il team.
- Preparare loghi RGB team/universita' senza logo IBM.

Definition of Done:

- Codice congelato.
- Video, blog, repo e certificati pronti.
- Checklist submission quasi completa.

### Fase 6 - Revisione finale e submit (21-26 giugno)

Obiettivo: consegna interna completa entro il 26 giugno 2026.

Attivita':

- Eseguire run finale ufficiale e salvare:
  - video giro migliore
  - tempo
  - configurazione usata
  - commit hash.
- Fare review incrociata:
  - tecnico: codice, setup, riproducibilita'
  - comunicazione: video inglese, chiarezza, storytelling
  - regolamento: circuito, partenza, materiali obbligatori.
- Completare certificato `SUBMIT TO WIN!` per tutti.
- Il rappresentante verifica Microsoft Form e prepara invio.
- Submit entro 26 giugno 2026 come scadenza interna UNISA.

Definition of Done:

- Submit pronto e verificato.
- Materiali accessibili da link pubblici o condivisi correttamente.
- Nessun requisito mancante.

### Fase 7 - Buffer ufficiale (27 giugno-1 luglio)

Obiettivo: usare il margine fino alla scadenza ufficiale AI Race del 1 luglio 2026 solo per fix critici.

Attivita':

- Nessuna riscrittura del driver salvo bug bloccanti.
- Verifica link, permessi, video e repository.
- Backup di video, repository, configurazioni e certificati.
- Submit finale sul form IBM/Codemotion se non gia' completato.

## 7. Metriche di successo

Metriche tecniche:

- Best lap time su Corkscrew da partenza ferma.
- Media e deviazione standard su 10 giri.
- Percentuale giri completati.
- Numero uscite pista.
- Danno totale e danno per giro.
- Numero recovery attivati.
- Tempo perso in curve critiche.
- Stabilita' con altri driver.

Metriche di comunicazione:

- Video team in inglese entro 3 minuti.
- Spiegazione chiara di architettura, sensori, azioni, Granite e SkillsBuild.
- Repository leggibile e riproducibile.
- Blog/report con percorso, scelte, risultati e call to action SkillsBuild.
- Certificati e badge completi per tutti i membri.

## 8. Strategia AI e tuning

La strategia consigliata e' ibrida:

- Baseline rule-based modulare per garantire controllo e affidabilita'.
- Tuning parametrico sistematico per Corkscrew.
- Behavioral cloning come componente AI sperimentale e documentata.
- Safety layer sempre attivo per limitare comandi rischiosi.
- IBM Granite usato per supportare sviluppo, review e comunicazione tecnica.

Per la gara, priorita':

1. completare sempre il giro;
2. evitare danni e uscite pista;
3. ridurre oscillazioni;
4. aumentare velocita' nei tratti sicuri;
5. ottimizzare le curve critiche;
6. registrare il miglior giro stabile, non un giro casuale non ripetibile.

## 9. Rischi e mitigazioni

- Rischio: auto veloce ma instabile.
  Mitigazione: i parametri in `src/driver/config.py` vanno ottimizzati su best lap, ma validati su piu' tentativi per evitare un giro casuale non ripetibile.

- Rischio: behavioral cloning lento o fragile.
  Mitigazione: usarlo come modulo confrontato e documentato; mantenere safety layer rule-based.

- Rischio: data distribution mismatch.
  Mitigazione: raccogliere dataset anche in stati difficili, curve, recovery e velocita' diverse.

- Rischio: submission incompleta.
  Mitigazione: checklist materiali e freeze una settimana prima della scadenza interna.

- Rischio: link privati o non accessibili.
  Mitigazione: test permessi da account esterno prima del submit.

- Rischio: video troppo narrativo e poco tecnico.
  Mitigazione: script video con flusso tecnico: problema, sensori, policy, tuning, Granite, risultati.

## 10. Checklist finale

- Team registrato con nome coerente.
- Tutti registrati su SkillsBuild con mail universitaria.
- Tutti registrati al form AI Race.
- Tutti i badge/certificati obbligatori scaricati.
- Presentazione unica con certificati di tutti.
- Drive di team completo.
- GitHub pubblico/accessibile.
- README completo.
- AI driver Python funzionante.
- Solo codice Python IA modificato.
- TORCS configurato con `scr_server 1`.
- Corkscrew usato per run ufficiale.
- Giro ufficiale da partenza ferma.
- Video giro veloce pronto.
- Video team in inglese, massimo 3 minuti.
- Blog/report pubblico pronto.
- Presenza social minima pronta.
- Loghi RGB team/universita' pronti, senza IBM.
- Uso IBM Granite documentato.
- Test multi-driver completato.
- Report tecnico con metriche completato.
- Microsoft Form pronto per il rappresentante.
- Submit interno entro 26 giugno 2026.
- Submit ufficiale entro 1 luglio 2026.
