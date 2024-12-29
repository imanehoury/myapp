import numpy as np
import matplotlib.pyplot as plt
############################################################################
import streamlit as st

import streamlit as st
import numpy as np

def get_user_input():
    st.write("### Paramètres pour le Flow Shop sans préparation et sans retard")

    # Saisie du nombre de machines et jobs
    m = st.number_input("Nombre de machines :", min_value=1, step=1, value=2)
    n = st.number_input("Nombre de jobs :", min_value=1, step=1, value=4)

    # Saisie des temps de traitement
    st.write("#### Temps de traitement (entrez les valeurs pour chaque machine et job)")
    P = []
    for i in range(m):
        row = st.text_input(f"Temps pour Machine {i + 1} (séparez les valeurs par des espaces) :", "1 "*n)
        if row.strip():
            try:
                row_data = list(map(int, row.split()))
                if len(row_data) != n:
                    st.error(f"Veuillez entrer exactement {n} valeurs pour Machine {i + 1}")
                    return None, None, None, None
                P.append(row_data)
            except ValueError:
                st.error("Les temps doivent être des entiers positifs.")
                return None, None, None, None
    
    if len(P) != m:
        st.error("Vous devez entrer les temps pour toutes les machines.")
        return None, None, None, None

    P = np.array(P)
    
    # Saisie du paramètre k si nécessaire
    k = 0
    if m > 2:
        k = st.number_input("Valeur de k (optionnel) :", min_value=0, step=1, value=1)
    
    return P, k, m, n

############################################################################

def makespan_johnson(P, seq):
    """
    Calcule le makespan pour une séquence donnée.
    Arguments :
    - P : Matrice numpy (2D) des temps de traitement (m x n).
    - seq : Liste représentant l'ordre des jobs.

    Retourne :
    - C : Matrice numpy des temps cumulés (m x n).
    - Cmax : Valeur du makespan.
    """
    m, n = P.shape
    C = np.zeros((m, n), dtype=int)
    # Calcul pour la première machine
    C[0, seq[0]] = P[0, seq[0]]
    for j in range(1, n):
        C[0, seq[j]] = C[0, seq[j - 1]] + P[0, seq[j]]

    # Calcul pour les autres machines
    for i in range(1, m):
        C[i, seq[0]] = C[i - 1, seq[0]] + P[i, seq[0]]
        for j in range(1, n):
            C[i, seq[j]] = max(C[i - 1, seq[j]], C[i, seq[j - 1]]) + P[i, seq[j]]

    # Makespan
    Cmax = C[m - 1, seq[-1]]
    return C, Cmax

############################################################################

def seq_johnson(P):
    """
    Applique l'algorithme de choix de séquence.
    Arguments :
    - P : Matrice numpy (2D) des temps de traitement (m x n).
    
    Retourne :
    - seq : Liste représentant l'ordre optimal des jobs.
    """
    m, n = P.shape
    if m != 2:
        raise ValueError("L'algorithme de choix de séquence est conçu pour 2 machines uniquement.")
    
    U, V = [], []

    # Diviser les jobs dans les groupes U et V
    for j in range(n):
        if P[0, j] <= P[1, j]:
            U.append((j, P[0, j]))
        else:
            V.append((j, P[1, j]))

    # Trier U par ordre croissant et V par ordre décroissant
    U.sort(key=lambda x: x[1])
    V.sort(key=lambda x: x[1], reverse=True)

    # Extraire uniquement les indices de jobs
    seq = [job[0] for job in U] + [job[0] for job in V]
    return seq


##################################################################################

def cds_sequence(P, k):
    """
    Implémente l'algorithme CDS pour un k spécifique.
    Retourne la séquence optimale selon la règle de Johnson.
    """
    m, n = P.shape
    jobs = np.arange(n)

    # Calcul des machines fictives M1 et M2 pour le k donné
    M1_fictive = np.sum(P[:k, :], axis=0)  # Somme des temps pour les k premières machines
    M2_fictive = np.sum(P[m-k:, :], axis=0)  # Somme des temps pour les k dernières machines

    print(M1_fictive,M2_fictive)
    # Application de la règle de Johnson
    seq = seq_johnson(np.array([M1_fictive, M2_fictive]))

    return seq
####################################################################################
def gantt_flowshop_with_metrics(P, seq):
    """
    Create Gantt chart and calculate scheduling metrics
    """
    m, n = P.shape
    jobs = list(range(n))
    # Convert P to numpy array for easier computation
    P_array = np.array(P)
    
    # Correctly get completion times matrix
    C, Cmax = makespan_johnson(P, seq)
    completion_times = C
    TFT = 0  # Total Flow Time
    
    # Calculate metrics
    for j in range(n):
        TFT += completion_times[-1, seq[j]]  # Add to total flow time
    
    # Plot Gantt chart
    plt.figure(figsize=(15, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, n))
    arrow_offset = m+1 # Start placing arrows slightly above the highest machine index
    for i in range(m):
        for j in range(n):
            start_time = completion_times[i, seq[j]] - P_array[i, seq[j]]
            plt.barh(i, P_array[i, seq[j]], left=start_time, color=colors[seq[j]], edgecolor='black')
            # Correct job label to start at J1
            plt.text(start_time + P_array[i, seq[j]] / 2, i, f'J{jobs[seq[j]] + 1}', ha='center', va='center', color='black')
    
    
    plt.annotate("", xy=(0, arrow_offset), xytext=(Cmax, arrow_offset),
    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5, zorder=4))
    plt.text((Cmax) / 2, arrow_offset + 0.05, f"Cmax={Cmax}",
    ha='center', va='bottom', color='black', fontsize=10, zorder=4)

    plt.ylim(-1, m+3)  # Extra space for annotations
    plt.xlim(0, np.max(C) + 5)  # Extend the x-axis slightly
    plt.xlabel('Time')
    plt.ylabel('Machines')
    plt.title(f'Flow Shop Gantt Chart (Cmax={Cmax}, TFT={TFT})')
    plt.yticks(range(m), [f'Machine {i + 1}' for i in range(m)])
    plt.grid(axis='x')
    st.pyplot(plt)
    plt.close()
    
    return TFT, Cmax, C

# Gantt chart plotting
def plot_gantt_chart(P, seq):
    m, n = P.shape
    C, Cmax = makespan_johnson(P, seq)
    plt.figure(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, n))
    for i in range(m):
        for j in range(n):
            start = C[i, seq[j]] - P[i, seq[j]]
            plt.barh(i, P[i, seq[j]], left=start, color=colors[j], edgecolor="black")
            plt.text(start + P[i, seq[j]] / 2, i, f"J{seq[j]+1}", ha="center", va="center")
    plt.xlabel("Time")
    plt.ylabel("Machines")
    plt.title(f"Gantt Chart (Makespan = {Cmax})")
    plt.yticks(range(m), [f"Machine {i+1}" for i in range(m)])
    plt.grid(axis="x")
    st.pyplot(plt)
    return Cmax
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
    st.pyplot(plt)
    plt.close()
    return TFR, TAR

################################################################################################################################################

def main():
    st.title("Flow Shop Scheduling: No Preparation, No Delay")
    st.write("Provide the necessary input data below:")

    # User Input
    m = st.number_input("Number of machines:", min_value=1, step=1)
    n = st.number_input("Number of jobs:", min_value=1, step=1)
    if m > 0 and n > 0:
        st.subheader("Processing Times")
        P = []
        for i in range(m):
            row = st.text_input(f"Processing times for Machine {i+1} (space-separated):", key=f"row_{i}")
            if row:
                P.append(list(map(int, row.split())))
        
        if len(P) == m:
            P = np.array(P)
            st.subheader("Input Data")
            st.write("Processing Matrix (P):")
            st.write(P)

            # Calculate sequence
            seq = sorted(range(n), key=lambda x: min(P[:, x]))
            st.subheader("Optimal Job Sequence:")
            st.write([s+1 for s in seq])

            # Generate Gantt chart
            st.subheader("Gantt Chart")
            C, Cmax = plot_gantt_chart(P, seq)

            # Display Makespan
            st.subheader("Makespan")
            st.write(f"Total Makespan (Cmax): {Cmax}")

            # Display Completion Matrix
            st.subheader("Completion Times Matrix:")
            st.write(C)
        else:
            st.warning("Please enter all processing times for each machine.")
# Run the main function
if __name__ == "__main__":
    main()