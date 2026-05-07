# Strategia tecnica

## Approccio

Il driver usa una policy ibrida:

- controllo rule-based modulare per affidabilita';
- speed planning dinamico basato sui sensori `track`;
- steering con `angle`, `trackPos` e bilanciamento dei sensori laterali;
- traction control su `wheelSpinVel`;
- recovery per fuori pista, auto bloccata e testacoda;
- opponent guard disponibile nei test multi-driver, disattivato nel profilo best lap;
- guida manuale Windows-friendly per test e raccolta dataset;
- behavioral cloning come esperimento AI/ML documentato.

## Sensori usati

- `angle`
- `track`
- `trackPos`
- `speedX`, `speedY`, `speedZ`
- `wheelSpinVel`
- `rpm`, `gear`
- `damage`
- `curLapTime`, `lastLapTime`, `distFromStart`, `distRaced`
- `opponents`, `racePos`
- `z`

## Azioni inviate

- `steer`
- `accel`
- `brake`
- `gear`
- `clutch`
- `meta`

## Configurazione

- `src/driver/config.py`: unico punto Python per i parametri automatici da gara su Corkscrew.
- `manual_control.py`: controller manuale per guida, debug e raccolta dataset.

## Uso di IBM Granite

Granite deve essere usato e documentato per:

- analisi dei log;
- revisione del codice;
- proposta di tuning parametri;
- spiegazione tecnica per video e blog;
- sintesi delle decisioni del team.
