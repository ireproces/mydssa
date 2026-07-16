# mydssa
> Repository per un progetto sviluppato per il corso di Decision Support Systems and Analytics.

Il codice implementa un solutore basato su **Ricerca Locale (Local Search)** per un problema di ottimizzazione sull'assegnazione di anestetici a pazienti.

## Struttura del Progetto
Il progetto è stato strutturato seguendo il principio della **separazione delle responsabilità**:

*   `anesthetic_assignment_ls.py`: è il **modulo principale della libreria**. Contiene tutta la logica del solutore, suddivisa in classi che rappresentano i diversi layer dell'architettura (Dati, Stato, Logica, Algoritmo).
*   `run.py`: è lo **script eseguibile** che orchestra gli esperimenti. Si occupa di generare i dati di test, configurare ed eseguire le diverse strategie di ricerca e salvare i report dei risultati.
*   `output/`: directory in cui vengono salvati i file `.txt` contenenti i report di ogni esecuzione.

## Architettura e Ottimizzazioni
L'architettura del solutore è suddivisa in layer, ciascuno con un ruolo specifico e ottimizzato per le massime prestazioni.

### Data Layer (`ProblemData`)
Un `@dataclass` che incapsula in modo efficiente tutti i parametri di input del problema (matrici di compatibilità, costi, pesi, ecc.) utilizzando array NumPy.\
Questo approccio centralizzato previene la duplicazione dei dati e migliora la gestione della memoria.

### State Layer (`Solution`)
Rappresenta una singola soluzione (il vettore delle decisioni `X`).\
La sua caratteristica chiave è l'uso della **Delta Evaluation**: il metodo `evaluate_move_delta` calcola l'impatto di una potenziale mossa (ΔZ) in tempo costante **O(1)**.
Questo è esponenzialmente più efficiente rispetto al ricalcolo da zero dell'intera funzione obiettivo per ogni vicino, ed è una delle ottimizzazioni più importanti dell'algoritmo.

### Logic Layer (`NeighborhoodGenerator`)
Definisce lo spazio di ricerca.\
Il suo metodo principale, `generate_moves`, è un **generatore** Python (`yield`) che produce le mosse del vicinato una alla volta (**lazy evaluation**).
Questo ottimizza drasticamente l'uso della memoria, specialmente per la strategia *First-Improvement*, che può così interrompere l'esplorazione non appena trova una mossa migliorativa.
Per ulteriore efficienza, la classe **pre-calcola e mette in cache** la lista degli anestetici compatibili per ogni paziente nel suo costruttore, evitando calcoli ridondanti all'interno del ciclo di ricerca.

### Algorithm Layer (`LocalSearchOptimizer`)
È il motore esecutivo che orchestra il processo di ricerca.

Implementa due strategie classiche di Ricerca Locale:
*   **First-Improvement**: valuta i vicini e accetta **immediatamente** la prima mossa che migliora la funzione obiettivo (ΔZ < 0). È molto veloce per iterazione, ideale per esplorare spazi di ricerca molto ampi.
*   **Best-Improvement**: valuta l'**intero vicinato** in ogni iterazione e applica la mossa che garantisce il miglioramento più grande (il ΔZ più negativo). È più intensiva dal punto di vista computazionale per iterazione, ma segue il percorso di discesa più ripido.

## Come Eseguire un Esperimento

Per lanciare un esperimento, esegui lo script `run.py` dal terminale:

```bash
python3 run.py
```

Lo script eseguirà le seguenti azioni:
1.  genererà un'istanza del problema con 532 pazienti e 207 anestetici;
2.  eseguirà l'algoritmo di ricerca locale con entrambe le strategie (`best` e `first`), partendo dalla stessa soluzione iniziale per un confronto equo;
3.  salverà un report di testo dettagliato con i dati di input, i log delle iterazioni e i risultati finali nella directory `output/`. Il nome del file includerà un timestamp per garantirne l'unicità.