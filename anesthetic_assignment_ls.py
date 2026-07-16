import numpy as np
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional

@dataclass
class ProblemData:
    alpha: np.ndarray  # Matrice di compatibilità (n_pazienti, n_anestetici)
    q: np.ndarray      # Quantità richiesta (n_pazienti, n_anestetici)
    s: np.ndarray      # Adeguatezza (n_pazienti, n_anestetici)
    c: np.ndarray      # Costo unitario (n_anestetici,)
    e: np.ndarray      # Impatto ambientale unitario (n_anestetici,)
    lambda_1: float    # Peso per il Costo totale
    lambda_2: float    # Peso per l'Impatto Ambientale totale
    lambda_3: float    # Peso per l'Adeguatezza (da massimizzare, quindi sottratto in Z) totale

    @property
    def n_patients(self) -> int:
        return self.alpha.shape[0]  # n: numero di righe

    @property
    def n_anesthetics(self) -> int:
        return self.alpha.shape[1]  # m: numero di colonne
    
class Solution:

    # --- 1. Inizializzazione e Creazione dello Stato ---

    # Costruttore.
    # Accetta un parametro opzionale X:
    # - se X non viene fornito, significa che l'algoritmo è appena stato avviato
    # - se X viene fornito, significa che l'algoritmo è già in esecuzione
    def __init__(self, data: ProblemData, X: Optional[np.ndarray] = None):
        self.data = data # collega l'istanza della soluzione ai dati del problema
        
        if X is None:
            self.X = self._generate_initial_feasible() # genera una soluzione iniziale ammissibile casuale
        else:
            self.X = X.copy() # copia la soluzione corrente

    # Metodo di generazione di una soluzione iniziale strettamente ammissibile.
    def _generate_initial_feasible(self) -> np.ndarray:
        X = np.zeros(self.data.n_patients, dtype=int)

        # itera su ogni paziente
        for i in range(self.data.n_patients):
            compatible_anesthetics = np.where(self.data.alpha[i] == 1)[0] # estrae solo gli indici degli anestetici compatibili
            
            if compatible_anesthetics.size == 0:
                raise ValueError(f"Il paziente {i} non ha alcun anestetico compatibile.")
            
            X[i] = np.random.choice(compatible_anesthetics) # sceglie casualmente uno degli anestetici compatibili
        return X

    # --- 2. Validazione ---

    # Metodo di controllo (sanity check).
    # Restituisce True se l'intero vettore X rispetta la matrice di compatibilità alpha, altrimenti False
    def is_feasible(self) -> bool:
        patient_indices = np.arange(self.data.n_patients) # array di indici per i pazienti, da 0 a n-1
        
        # seleziona i valori di alpha corrispondenti alle assegnazioni in X e controlla che siano tutti 1
        return np.all(self.data.alpha[patient_indices, self.X] == 1)
    
    # --- 3. Valutazione Assoluta ---

    # Metodo per il calcolo delle componeneti della funzione obiettivo Z, allo stato attuale della soluzione.
    def compute_metrics(self) -> Tuple[float, float, float]:
        patient_indices = np.arange(self.data.n_patients)

        chosen_anesthetics = self.X # self.X contiene già gli indici degli anestetici scelti

        # calcolo vettorizzato delle metriche totali: costo, impatto ambientale ed adeguatezza
        c_tot = np.sum(self.data.c[chosen_anesthetics] * self.data.q[patient_indices, chosen_anesthetics])
        e_tot = np.sum(self.data.e[chosen_anesthetics] * self.data.q[patient_indices, chosen_anesthetics])
        s_tot = np.sum(self.data.s[patient_indices, chosen_anesthetics])

        return c_tot, e_tot, s_tot

    # Metodo per il calcolo del valore della funzione obiettivo Z, allo stato attuale della soluzione.
    def objective_value(self) -> float:
        c_tot, e_tot, s_tot = self.compute_metrics()
        return (self.data.lambda_1 * c_tot) + \
               (self.data.lambda_2 * e_tot) - \
               (self.data.lambda_3 * s_tot)

    # --- 4. Valutazione Relativa (Delta) ---

    # Metodo per il calcolo del delta della funzione obiettivo Z.
    # Delta negativo indica un miglioramento della soluzione corrente.
    def evaluate_move_delta(self, patient_idx: int, new_anesthetic: int) -> float:
        old_anesthetic = self.X[patient_idx] # anestetico attualmente assegnato al paziente
        
        # calcola il delta delle componenti
        delta_c = (self.data.c[new_anesthetic] * self.data.q[patient_idx, new_anesthetic]) - \
                  (self.data.c[old_anesthetic] * self.data.q[patient_idx, old_anesthetic])
                  
        delta_e = (self.data.e[new_anesthetic] * self.data.q[patient_idx, new_anesthetic]) - \
                  (self.data.e[old_anesthetic] * self.data.q[patient_idx, old_anesthetic])
                  
        delta_s = self.data.s[patient_idx, new_anesthetic] - self.data.s[patient_idx, old_anesthetic]
        
        # combina le variazioni delle singole componenti per ottenere il delta della funzione obiettivo Z
        delta_z = (self.data.lambda_1 * delta_c) + \
                  (self.data.lambda_2 * delta_e) - \
                  (self.data.lambda_3 * delta_s)
                  
        return delta_z
    
    # Metodo di applicazione della mossa: aggiorna la soluzione corrente con il nuovo
    # anestetico per il paziente specificato.
    def apply_move(self, patient_idx: int, new_anesthetic: int) -> None:
        self.X[patient_idx] = new_anesthetic

class NeighborhoodGenerator:

    def __init__(self, data: ProblemData):
        self.data = data

        # pre-calcola e memorizza la lista di anestetici compatibili per ogni paziente
        self.compatible_anesthetics_map = [np.where(data.alpha[i] == 1)[0] for i in range(data.n_patients)]

    # Metodo di generazione lazy delle mosse del vicinato per la soluzione data.
    # Restituisce un iteratore di tuple (indice_paziente, nuovo_anestetico) ammissibili.
    def generate_moves(self, current_solution: Solution) -> Iterator[Tuple[int, int]]:

        # itera su ogni paziente
        for i in range(self.data.n_patients):
            current_anesthetic = current_solution.X[i] # anestetico attualmente assegnato al paziente i
            
            # itera sulla lista pre-calcolata di anestetici compatibili per il paziente i
            for new_anesthetic in self.compatible_anesthetics_map[i]:
                if new_anesthetic != current_anesthetic:
                    yield (i, new_anesthetic)


class LocalSearchOptimizer:

    # Costruttore.
    # Inizializza l'ottimizzare con i dati del problema e la strategia di ricerca locale scelta:
    # - "first" per First-Improvement;
    # - "best" per Best-Improvement.
    def __init__(self, data: ProblemData, strategy: str = "best"):

        # controllo di validità della strategia scelta
        if strategy not in ["first", "best"]:
            raise ValueError("Strategia non supportata. Usa 'first' o 'best'.")
        
        self.data = data
        self.strategy = strategy

        # inizializza il generatore di vicinato
        self.neighborhood = NeighborhoodGenerator(data)

    # Metodo di ricerca locale.
    # Esegue l'algoritmo fino alla convergenza sull'ottimo locale e restituisce la soluzione trovata.
    def solve(self, log_file=None) -> Tuple[Solution, int]:

        current_sol = Solution(self.data) # soluzione iniziale casuale ammissibile
        improved = True # boolean per controllare se ci sono stati miglioramenti nella soluzione corrente
        iteration = 0 # contatore delle iterazioni della ricerca locale
        current_z = current_sol.objective_value() # funzione obiettivo della soluzione iniziale
        
        print(f"--- Inizio Local Search ({self.strategy}-improvement) ---", file=log_file)
        print(f"Iter 0 (Sol. Iniziale) - Z: {current_z:.4f}", file=log_file)
        
        # ciclo principale, si ferma quando non si riescono più a trovare soluzioni migliorative
        # ogni esecuzione del ciclo rappresenta un passo di discesa verso un minimo locale
        while improved:

            improved = False
            best_delta = 0.0
            best_move = None
            
            # itera su tutte le mosse del vicinato generate da NeighborhoodGenerator
            for patient_idx, new_anesthetic in self.neighborhood.generate_moves(current_sol):

                # calcola il delta della funzione obiettivo Z per la mossa corrente
                delta_z = current_sol.evaluate_move_delta(patient_idx, new_anesthetic)
                
                # verifica se la mossa è migliorativa
                # (-1e-6 è una soglia robusta per gestire le imprecisioni dei numeri in virgola mobile)
                if delta_z < -1e-6:

                    # se first-improvement, interrompe l'esplorazione corrente alla prima soluzione migliorativa
                    if self.strategy == "first":
                        current_sol.apply_move(patient_idx, new_anesthetic) # applica la mossa
                        improved = True
                        current_z += delta_z # aggiorna Z
                        iteration += 1
                        print(f"Iter {iteration} - Mossa: p_{patient_idx} -> a_{new_anesthetic} | Nuovo Z: {current_z:.4f} (Delta: {delta_z:.4f})", file=log_file)
                        break
                    
                    # se best-improvement, esplora tutto il vicinato e memorizza la soluzione con il delta migliore
                    elif self.strategy == "best":
                        if delta_z < best_delta:
                            best_delta = delta_z
                            best_move = (patient_idx, new_anesthetic)
            
            # dopo aver scansionato l'intero vicinato se si è trovata una mossa migliorativa,
            # applica la mossa con il delta migliore
            if self.strategy == "best" and best_move is not None:
                current_sol.apply_move(best_move[0], best_move[1])
                improved = True
                current_z += best_delta # Aggiorna Z in O(1)
                iteration += 1
                print(f"Iter {iteration} - Mossa: p_{best_move[0]} -> a_{best_move[1]} | Nuovo Z: {current_z:.4f} (Delta: {best_delta:.4f})", file=log_file)
        
        print("[!] Ottimo Locale Raggiunto\n", file=log_file)
        return current_sol, iteration