import numpy as np
from datetime import datetime
import os

from anesthetic_assignment_ls import ProblemData, LocalSearchOptimizer

# Funzione che scrive i dati del problema nel file di log
def log_problem_data(data: ProblemData, file):
    header = " DATI SINTETICI DI INPUT "
    file.write(f"{header:=^80}\n\n")
    file.write(f"Numero pazienti: {data.n_patients}\n")
    file.write(f"Numero anestetici: {data.n_anesthetics}\n\n")
    file.write(f"Matrice di compatibilità (alpha):\n{data.alpha}\n\n")
    file.write(f"Matrice delle quantità (q):\n{np.round(data.q, 2)}\n\n")
    file.write(f"Matrice di adeguatezza (s):\n{data.s}\n\n")
    file.write(f"Vettore dei costi (c):\n{data.c}\n\n")
    file.write(f"Vettore impatto ambientale (e):\n{data.e}\n\n")
    file.write("Pesi funzione obiettivo:\n")
    file.write(f"  lambda_1 (costo): {data.lambda_1}\n")
    file.write(f"  lambda_2 (impatto): {data.lambda_2}\n")
    file.write(f"  lambda_3 (adeguatezza): {data.lambda_3}\n\n")
    file.write(f"{'='*80}\n\n")

# Funzione che esegue la ricerca locale e scrive i risultati nel file di log
def run_and_report(data: ProblemData, strategy: str, log_file=None):
    optimizer = LocalSearchOptimizer(data, strategy=strategy)
    solution, iterations = optimizer.solve(log_file=log_file)
    c_tot, e_tot, s_tot = solution.compute_metrics()

    header = f" REPORT FINALE ({strategy.upper()}-IMPROVEMENT) "
    print(f"{header:#^80}", file=log_file)
    print(f"Soluzione X ottima locale : {solution.X}", file=log_file)
    print(f"Ammissibilità rigorosa    : {'Verificata' if solution.is_feasible() else 'FALLITA'}", file=log_file)
    print(f"Numero di iterazioni      : {iterations}", file=log_file)
    print(f"Funzione Obiettivo Z      : {solution.objective_value():.2f}", file=log_file)
    print(f" -> Costo Totale (C_tot)  : {c_tot:.2f}", file=log_file)
    print(f" -> Impatto Tot. (E_tot)  : {e_tot:.2f}", file=log_file)
    print(f" -> Adeguatezza (S_tot)   : {s_tot:.2f}\n", file=log_file)

if __name__ == "__main__":
    np.random.seed(42)

    # 1. Creazione Mock Data (532 pazienti, 207 anestetici)
    n_patients = 532
    n_anesthetics = 207

    # a. alpha: matrice di compatibilità
    # Generiamo una matrice sparsa, assicurando che ogni paziente abbia
    # un numero sufficiente di opzioni compatibili per evitare problemi di ammissibilità.
    mock_alpha = np.zeros((n_patients, n_anesthetics), dtype=int)
    for i in range(n_patients):
        # Scegli un numero casuale di anestetici compatibili (es. da 5 a 20)
        num_compatible = np.random.randint(5, 21)
        # Scegli casualmente gli indici degli anestetici compatibili
        compatible_indices = np.random.choice(n_anesthetics, num_compatible, replace=False)
        mock_alpha[i, compatible_indices] = 1

    # b. quantità necessarie q (valori realistici generati casualmente)
    mock_q = np.random.uniform(10.0, 50.0, size=(n_patients, n_anesthetics))

    # c. adeguatezza s (valori da 1 a 10)
    mock_s = np.random.uniform(1.0, 10.0, size=(n_patients, n_anesthetics))

    # d. costi unitari c (valori per ogni anestetico)
    mock_c = np.random.uniform(5.0, 25.0, size=n_anesthetics)

    # e. impatto ambientale unitario e (valori per ogni anestetico)
    mock_e = np.random.uniform(1.0, 6.0, size=n_anesthetics)

    # 2. Inizializzazione Data Layer
    problem_data = ProblemData(
        alpha=mock_alpha,
        q=mock_q,
        s=mock_s,
        c=mock_c,
        e=mock_e,
        # l'assegnazione dei pesi riflette le priorità del decisore
        lambda_1=1.0,
        lambda_2=2.0,
        lambda_3=10.0
    )

    # 3. Creazione del file di report e esecuzione
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_esecuzione_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            log_problem_data(problem_data, file=f)

            np.random.seed(42)
            run_and_report(problem_data, "best", log_file=f)

            np.random.seed(42)
            run_and_report(problem_data, "first", log_file=f)
        print(f"Report di esecuzione salvato con successo nel file: {filepath}")
    except IOError as e:
        print(f"Errore durante la scrittura del file di report: {e}")