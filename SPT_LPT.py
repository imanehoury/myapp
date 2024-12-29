import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
############################################################################

def get_user_input():
    """Collect inputs for SPT and LPT scheduling."""
    st.write("### Enter Inputs for SPT/LPT Scheduling")
    
    # Number of machines
    m = st.number_input("Enter the number of machines:", min_value=1, value=2, step=1)
    
    # Number of jobs
    n = st.number_input("Enter the number of jobs:", min_value=1, value=4, step=1)
    
    # Processing times
    st.write("### Enter Processing Times (P)")
    P = []
    for i in range(int(m)):
        row = st.text_input(f"Enter processing times for Machine {i + 1} (separate by space):","1 "*n)
        if row:
            P.append(list(map(float, row.split())))

    # Validate dimensions
    if len(P) == m and all(len(row) == n for row in P):
        P = np.array(P)
        return P, int(m), int(n)
    else:
        st.warning("Please provide valid processing times for all machines.")
        return None, None, None

def get_user_inputz():
    """
    Collect user input for scheduling parameters
    """
    print("Flow Shop Scheduling Gantt Chart Generator")
    print("==========================================")
    
    # Get number of machines and jobs
    while True:
        try:
            m = int(input("Nombre de machines (entier positif) : "))
            n = int(input("Nombre de jobs (entier positif) : "))
            if m <= 0 or n <= 0:
                print("Veuillez entrer des entiers positifs supérieurs à zéro.")
                continue
            break
        except ValueError:
            print("Veuillez entrer des valeurs entières valides.")
    
    # Saisie des temps de traitement
    print("\nEntrez les temps de traitement pour chaque machine :")
    P = []
    for i in range(m):
        while True:
            try:
                row = list(map(int, input(f"Machine {i + 1} (temps séparés par des espaces) : ").split()))
                if len(row) != n:
                    print(f"Veuillez entrer exactement {n} valeurs.")
                    continue
                if any(t < 0 for t in row):
                    print("Veuillez entrer uniquement des valeurs positives.")
                    continue
                break
            except ValueError:
                print("Veuillez entrer uniquement des nombres entiers.")
        P.append(row)
    P = np.array(P)
  
    
    return P, m, n

############################################################################
def LPT(P):
    """
    Applique la règle LPT (Longest Processing Time) pour classer les jobs.
    """
    total_times = P.sum(axis=0)  # Temps total de chaque job
    seq = np.argsort(total_times)[::-1]  # Trie en ordre décroissant
    seq = [int(x + 1) for x in seq]  # Index des jobs (1-based)
    return seq

def SPT(P):
    """
    Applique la règle SPT (Shortest Processing Time) pour classer les jobs.
    """
    total_times = P.sum(axis=0)  # Temps total de chaque job
    seq = np.argsort(total_times)  # Trie en ordre croissant
    seq = [int(x + 1) for x in seq]
    return seq


def gantt_flowshop_with_metrics(P, seq):
    """
    Affiche un diagramme de Gantt pour le problème Flow Shop et calcule les métriques Cmax et TFT.
    """
    # Initialisation des paramètres
    m, n = P.shape
    P_array = np.array(P)
    seq = [x - 1 for x in seq]  # Ajustement de la séquence (0-based)
    # Création de la matrice C pour stocker les temps d'achèvement
    C = np.zeros((m, n))
    
    # Calcul des temps d'achèvement C[i][j] pour chaque machine et job
    for i in range(m):  # Pour chaque machine
        for j in range(n):  # Pour chaque job dans la séquence
            job_idx = seq[j]  # Index du job actuel
            if i == 0 and j == 0:  # Premier job sur la première machine
                C[i, j] = P_array[i, job_idx]
            elif i == 0:  # Première machine, jobs suivants
                C[i, j] = C[i, j - 1] + P_array[i, job_idx]
            elif j == 0:  # Première job sur les autres machines
                C[i, j] = C[i - 1, j] + P_array[i, job_idx]
            else:  # Autres cas : dépendance sur la machine précédente et le job précédent
                C[i, j] = max(C[i - 1, j], C[i, j - 1]) + P_array[i, job_idx]
    
    # Calcul des métriques Cmax et TFT
    Cmax = C[-1, -1]  # Temps d'achèvement pour le dernier job sur la dernière machine
    # Calcul de Total Flow Time (TFT)
    TFT = np.sum(C[-1])

    # Affichage du diagramme de Gantt
    plt.figure(figsize=(15, 8))
    colors_jobs = plt.cm.Set3(np.linspace(0, 1, n))  # Couleurs pour chaque job

    for i in range(m):  # Pour chaque machine
        for j in range(n):  # Pour chaque job dans la séquence
            job_idx = seq[j]
            start_time = C[i, j] - P_array[i, job_idx]
            processing_time = P_array[i, job_idx]
            plt.barh(i, processing_time, left=start_time, color=colors_jobs[job_idx], edgecolor='black')
            plt.text(start_time + processing_time / 2, i, f'J{job_idx + 1}',
                     ha='center', va='center', fontweight='bold', color='black')

        # Adjust axis limits to ensure annotations are visible
    arrow_offset = m+1 # Start placing arrows slightly above the highest machine index
    plt.ylim(-1, m+3)  # Extra space for annotations
    plt.xlim(0, np.max(C) + 5)  # Extend the x-axis slightly
    
    plt.xlabel('Time')
    plt.ylabel('Machines')
    plt.title(f'Flow Shop Gantt Chart (Cmax={Cmax}, TFT={TFT})')
    plt.yticks(range(m), [f'Machine {i + 1}' for i in range(m)])
    plt.grid(axis='x')
    plt.yticks(range(m), [f'Machine {i + 1}' for i in range(m)])
    plt.annotate("", xy=(0, arrow_offset), xytext=(Cmax, arrow_offset),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1.5, zorder=4))
    plt.text((Cmax) / 2, arrow_offset + 0.05, f"Cmax={Cmax}",
         ha='center', va='bottom', color='black', fontsize=10, zorder=4)


    # Set x-axis ticks to intervals of 1 unit
    max_time = int(np.max(C)) + 1  # Determine the maximum time
    plt.xticks(range(0, max_time + 1, 1))  # Set ticks from 0 to max_time with step 1
    plt.tight_layout()
    st.pyplot(plt)
    
    return TFT, Cmax, C


############################################################################

def calculate_performance_metrics_per_machine(P, m, n,Cmax):
    TFR = np.zeros(m)
    TAR = np.zeros(m)
    for i in range(m):
        sum_pi = np.sum(P[i, :])
        TFR[i] = sum_pi / Cmax
        TAR[i] = 1 - TFR[i]
    TFR *= 100
    TAR *= 100
    machines = [f'Machine {i+1}' for i in range(len(TFR))]
    x = np.arange(len(machines))
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    bars1 = plt.bar(x - bar_width / 2, TFR, bar_width, label='TFR (%)', color='#BA55D3', edgecolor='black', linewidth=1.5, zorder=3)
    bars2 = plt.bar(x + bar_width / 2, TAR, bar_width, label='TAR (%)', color='#AFEEEE', edgecolor='black', linewidth=1.5, zorder=3)
    
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}', ha='center', fontsize=10, zorder=4)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}', ha='center', fontsize=10, zorder=4)
    
    plt.title('Taux de Fonctionnement Réel (TFR) et d\'Arrêt Réel (TAR)')
    plt.xlabel('Machines')
    plt.ylabel('Pourcentage (%)')
    plt.xticks(x, machines)
    plt.legend()
    
    # Dashed grid lines with lower zorder
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
    plt.tight_layout()
    st.pyplot(plt)
    return TFR, TAR

################################################################################################################################################

def main():
    """
    Main function to run the Flow Shop Scheduling analysis
    """
    # Get user input
    P,m,n = get_user_input()
    while True:
        print("\nChoisissez le type de tri :")
        print("1 - Tri croissant (SPT)")
        print("2 - Tri décroissant (LPT)")
        choice = input("Entrez votre choix (1 ou 2) : ")
        if choice in ['1', '2']:
            break
        else:
            print("Veuillez entrer '1' pour SPT ou '2' pour LPT.")

    if choice == '1':
        seq = SPT(P)
        method = "SPT (Tri Croissant)"
    else:
        seq = LPT(P)
        method = "LPT (Tri Décroissant)"
        
    print("\nCalculating Scheduling Metrics:")
    TFT, Cmax ,C = gantt_flowshop_with_metrics(P, seq)
    # Print metrics
    print(f"\nMetrics Summary:")
    print("Séquence optimisée :", seq)
    print(f"Matrice des temps d'achèvement (C) :\n{C}")
    print(f"Makespan (Cmax): {Cmax}")
    print(f"Total Flow Time (TFT): {TFT}")
    
    TFR, TAR = calculate_performance_metrics_per_machine(P, m, n,Cmax)
    # You can access individual machine metrics like this:
    print("Machine TAR values:", TAR)
    print("Machine TFR values:", TFR)

# Run the main function
if __name__ == "__main__":
    main()


