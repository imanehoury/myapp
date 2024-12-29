import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

############################################################################
def get_user_input():
    """
    Collect inputs for Flow Shop Scheduling via Streamlit interface.
    """
    st.write("### Flow Shop Scheduling - Input Parameters")
    
    # Number of machines
    m = st.number_input("Enter the number of machines:", min_value=1, value=2, step=1)
    
    # Number of jobs
    n = st.number_input("Enter the number of jobs:", min_value=1, value=4, step=1)
    
    # Processing times
    st.write("### Enter Processing Times (P)")
    P = []
    for i in range(int(m)):
        row = st.text_input(f"Enter processing times for Machine {i + 1} (separated by spaces):")
        if row:
            P.append(list(map(float, row.split())))

    # Validate dimensions
    if len(P) == m and all(len(row) == n for row in P):
        P = np.array(P)
    else:
        st.warning("Please provide valid processing times for all machines.")
        return None, None, None, None, None

    # Setup times
    st.write("### Enter Setup Times (S)")
    S = np.zeros((n, n, m), dtype=int)
    for k in range(int(m)):
        st.write(f"Setup times for Machine {k + 1}:")
        for i in range(int(n)):
            row = st.text_input(f"Setup times for Job {i + 1} (separated by spaces, Machine {k + 1}):")
            if row:
                values = list(map(float, row.split()))
                if len(values) == n:
                    S[i, :, k] = values
                else:
                    st.warning(f"Please enter exactly {n} values for Job {i + 1}, Machine {k + 1}.")
                    return None, None, None, None, None

    # Order (ascending or descending)
    order = st.radio("Choose the job order:", ["Ascending (c)", "Descending (d)"])
    order = 'c' if "Ascending" in order else 'd'

    return P, S, m, n, order

def get_user_inputz():
    """
    Collect user input for scheduling parameters
    """
    print("Flow Shop Scheduling Gantt Chart Generator")
    print("==========================================")
    
    # Get number of machines and jobs
    while True:
        try:
            m = int(input("Nombre de machines : "))
            n = int(input("Nombre de jobs : "))
            break
        except ValueError:
            print("Please enter valid integer values.")
    
    # Saisie des temps de traitement
    print("Entrez les temps de traitement pour chaque machine :")
    P = []
    for i in range(m):
        row = list(map(int, input(f"Machine {i + 1} (temps séparés par des espaces) : ").split()))
        while len(row) != n:
            print(f"Veuillez entrer {n} valeurs.")
            row = list(map(int, input(f"Machine {i + 1} (temps séparés par des espaces) : ").split()))
        P.append(row)
    P = np.array(P)

    
    # Saisie des temps de préparation
    print("Entrez les temps de préparation (S) :")
    S = np.zeros((n, n, m), dtype=int)
    for k in range(m):
        print(f"Machine {k + 1} :")
        for i in range(n):
            row = list(map(int, input(f"Préparation pour le job {i + 1} (séparés par des espaces) : ").split()))
            while len(row) != n:
                print(f"Veuillez entrer {n} valeurs.")
                row = list(map(int, input(f"Préparation pour le job {i + 1} (séparés par des espaces) : ").split()))
            S[i, :, k] = row
            
    ordre = input("Écrire 'c' pour un ordre croissant et 'd' pour un ordre décroissant: ").strip().lower()
    
    return P, S, m, n,ordre

############################################################################

def calculate_completion_times(P, S, seq):
    m, n = P.shape
    seq = seq - 1
    C = np.zeros((m, n))
    C[0, seq[0]] = P[0, seq[0]] + S[seq[0], seq[0], 0]

    for i in range(1, m):
        C[i, seq[0]] = max(C[i - 1, seq[0]], S[seq[0], seq[0], i]) + P[i, seq[0]]

    for j in range(1, n):
        C[0, seq[j]] = C[0, seq[j - 1]] + S[seq[j - 1], seq[j], 0] + P[0, seq[j]]

    for j in range(1, n):
        for i in range(1, m):
            C[i, seq[j]] = max(
                C[i - 1, seq[j]],
                C[i, seq[j - 1]] + S[seq[j - 1], seq[j], i]
            ) + P[i, seq[j]]
    
    Cmax = np.max(C)
    return Cmax, C 


# Function to determine the job sequence
def choix_sequence_decroissant(P, S):
    m, n = P.shape
    TP = np.zeros(n)
    A = np.arange(n)

    for j in range(n):
        for i in range(m):
            TP[j] += P[i, j] + S[j, j, i]

    indices = np.argsort(-TP)
    seq = A[indices]
    return seq + 1

def choix_sequence_croissant(P, S):
    seq=choix_sequence_decroissant(P,S)[::-1] 
    return seq + 1

############################################################################
def gantt_flowshop_with_metrics(P, S, seq): 
    """
    Create Gantt chart and calculate scheduling metrics, including setup times.
    """
    m, n = P.shape
    P_array = np.array(P)
    S_array = np.array(S)
    
    # Calculer les temps de fin
    Cmax, C = calculate_completion_times(P, S, seq)
    
    TFT = 0
    # Calculate metrics
    for j in range(n):
        job_index = seq[j] - 1  # Ajuster l'indice
        TFT += C[-1, job_index]  # Add to total flow time

    
    # Plot Gantt chart
    plt.figure(figsize=(15, 8)) 
    colors_jobs = plt.cm.Set3(np.linspace(0, 1, n), alpha=100)  # Colors for processing times
    colors_setup = plt.cm.Set3(np.linspace(0, 1, n), alpha=-10022)  # Setup times with reduced alpha
    arrow_offset = m+1 # Start placing arrows slightly above the highest machine index
    
    for i in range(m):
        for j in range(n):
            job_index = seq[j] - 1  # Ajuster l'indice
            
            # Calculate start time for processing
            setup_time = S_array[seq[j - 1] - 1, job_index, i] if j > 0 else S_array[job_index, job_index, i]
            start_time_setup = C[i, job_index] - P_array[i, job_index] - setup_time
            start_time_processing = C[i, job_index] - P_array[i, job_index]
            
            # Plot setup time (if non-zero)
            if setup_time > 0:
                plt.barh(i, setup_time, left=start_time_setup, color=colors_setup[job_index], edgecolor='black', alpha=0.8)
            
            # Plot processing time
            plt.barh(i, P_array[i, job_index], left=start_time_processing, color=colors_jobs[job_index], edgecolor='black')
            
            plt.text(start_time_processing + P_array[i, job_index]/2, i, f'J{seq[j]}', 
                 ha='center', va='center', fontweight='bold', color='black')


        # Adjust axis limits to ensure annotations are visible
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

def calculate_performance_metrics_per_machine(C, P, S, m, n, seq,Cmax):
    m, n = P.shape
    TFR = np.zeros(m)
    TAR = np.zeros(m)
    TAP = np.zeros(m)

    for i in range(m):
        sum_si = sum(S[seq[j - 1] - 1, seq[j] - 1, i] for j in range(1, n))
        sum_si = sum_si + S[seq[0] - 1 , seq[0] - 1, i]
        TAP[i] = sum_si / Cmax
        TFR[i] = np.sum(P[i, :]) / Cmax
        TAR[i] = 1 - TFR[i] - TAP[i]

    TFR *= 100
    TAR *= 100
    TAP *= 100
    TFT = np.sum(C[-1, :])
    machines = [f'Machine {i+1}' for i in range(len(TFR))]
    x = np.arange(len(machines))
    bar_width = 0.25

    plt.figure(figsize=(12, 7))
    
    bars1 = plt.bar(x - bar_width, TFR, bar_width, label='TFR (%)', color='green', edgecolor='black', linewidth=1.5, zorder=3)
    bars2 = plt.bar(x, TAR, bar_width, label='TAR (%)', color='red', edgecolor='black', linewidth=1.5, zorder=3)
    bars3 = plt.bar(x + bar_width, TAP, bar_width, label='TAP (%)', color='blue', edgecolor='black', linewidth=1.5, zorder=3)

    for bar in bars1 + bars2 + bars3:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}', ha='center', fontsize=10, zorder=4)

    plt.title('TFR, TAR, and TAP for Machines')
    plt.xlabel('Machines')
    plt.ylabel('Pourcentage (%)')
    plt.xticks(x, machines)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
    plt.tight_layout()
    st.pyplot(plt)
    return TFR, TAR, TAP

#########################################################################
def main():
    """
    Main function to run the Flow Shop Scheduling analysis.
    """
    # Get user input
    P, S, m, n, order = get_user_input()
    
    # Determine sequence based on number of machines
    if order == 'd' :
            seq = choix_sequence_decroissant(P, S)
    else :
            seq = choix_sequence_croissant(P , S)
    
    # Calculate and display metrics
    print("\nCalculating Scheduling Metrics:")
    TFT, Cmax, C = gantt_flowshop_with_metrics(P, S, seq)
    TFR, TAR, TAP = calculate_performance_metrics_per_machine(C, P, S, m, n, seq,Cmax)
    
    # Print metrics
    print("\nMetrics Summary:")
    print("Séquence optimisée :", seq)
    print(f"Matrice des temps d'achèvement (C) :\n{C}")
    print(f"Makespan (Cmax): {Cmax}")
    print(f"Total Flow Time (TFT): {TFT}")
    print("\nMachine Performance:")
    print("TAR values:", TAR)
    print("TFR values:", TFR)
    print("TAP values:", TAP)

# Run the main function
if __name__ == "__main__":
    main()