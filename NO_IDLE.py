import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
############################################################################


def get_user_input():
    """Collect inputs for No Idle scheduling."""
    st.write("### Enter Inputs for No Idle Scheduling")
    
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

def get_user_inputzz():
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
    Applique la règle LPT (Longest Processing Time)
    """
    total_times = P.sum(axis=0)
    seq = np.argsort(-total_times)  # Trie en ordre décroissant
    return list(seq)

def SPT(P):
    """
    Applique la règle SPT (Shortest Processing Time)
    """
    total_times = P.sum(axis=0)
    seq = np.argsort(total_times)  # Trie en ordre croissant
    return list(seq)

def C_no_idle(seq, P):
    """
    Calculate completion times for no-idle flowshop scheduling
    
    Args:
        seq: List of job indices (0-based)
        P: Processing times matrix
    """
    #seq = [job - 1 for job in seq]  # Adjust sequence to 0-based indexing
    m = len(P)
    n = len(seq)
    
    # Initialize L values
    L = np.zeros(m)
    
    # Calculate L values for each machine
    for i in range(1, m):
        d = np.zeros(n)
        for k in range(n):
            s1 = sum(P[i-1][seq[j]] for j in range(k+1))
            s2 = sum(P[i][seq[j]] for j in range(k))
            d[k] = s1 - s2
        L[i] = L[i-1] + max(d)
    
    # Initialize matrices
    S = np.zeros((m, n))  # Start times
    C = np.zeros((m, n))  # Completion times
    
    # Set initial values
    for i in range(m):
        S[i,0] = L[i]
        C[i,0] = S[i,0] + P[i,seq[0]]
    
    # Calculate remaining values
    for j in range(1, n):
        for i in range(m):
            S[i,j] = C[i,j-1]
            C[i,j] = S[i,j] + P[i,seq[j]]
    
    Cmax = C[m-1,n-1]
    return C, Cmax


def gantt_flowshop_with_metrics(P, seq):
    """
    Affiche le diagramme de Gantt et calcule les métriques
    """
    m, n = P.shape
    C, Cmax = C_no_idle(seq, P)
    TFT = np.sum(C[-1])

    plt.figure(figsize=(15, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, n))

    for i in range(m):
        for j in range(n):
            job = seq[j]
            start_time = C[i,j] - P[i,job]
            plt.barh(i, P[i,job], left=start_time, 
                    color=colors[job], edgecolor='black')
            plt.text(start_time + P[i,job]/2, i, 
                    f'J{job+1}', ha='center', va='center')

    arrow_offset = m + 1
    plt.ylim(-1, m + 3)
    plt.xlim(0, np.max(C) + 5)
    
    plt.xlabel('Time')
    plt.ylabel('Machines')
    plt.title(f'Flow Shop Gantt Chart (Cmax={Cmax:.1f}, TFT={TFT:.1f})')
    plt.yticks(range(m), [f'Machine {i+1}' for i in range(m)])
    plt.grid(axis='x')

    # Add Cmax annotation
    plt.annotate("", xy=(0, arrow_offset), xytext=(Cmax, arrow_offset),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    plt.text(Cmax/2, arrow_offset + 0.05, f"Cmax={Cmax:.1f}",
             ha='center', va='bottom')

    plt.tight_layout()
    st.pyplot(plt)
    
    return TFT, Cmax, C

############################################################################

def calculate_performance_metrics_per_machine(P, m, n, Cmax):
    """
    Calcule et affiche les métriques de performance par machine
    """
    TFR = np.zeros(m)
    TAR = np.zeros(m)
    
    for i in range(m):
        sum_pi = np.sum(P[i,:])
        TFR[i] = (sum_pi / Cmax) * 100
        TAR[i] = 100 - TFR[i]

    machines = [f'Machine {i+1}' for i in range(m)]
    x = np.arange(len(machines))
    
    plt.figure(figsize=(10, 6))
    width = 0.35
    
    plt.bar(x - width/2, TFR, width, label='TFR (%)', 
           color='#BA55D3', edgecolor='black', linewidth=1.5)
    plt.bar(x + width/2, TAR, width, label='TAR (%)', 
           color='#AFEEEE', edgecolor='black', linewidth=1.5)
    
    for i, v in enumerate(TFR):
        plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
    for i, v in enumerate(TAR):
        plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
    
    plt.title('Taux de Fonctionnement Réel (TFR) et d\'Arrêt Réel (TAR)')
    plt.xlabel('Machines')
    plt.ylabel('Pourcentage (%)')
    plt.xticks(x, machines)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)
    
    return TFR, TAR

################################################################################################################################################

def main():
    """
    Main function to run the Flow Shop Scheduling analysis
    """
    # Get user input
    P,m,n = get_user_inputzz()
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
    # Calculate and display metrics
    C, Cmax = C_no_idle(seq, P)
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


