import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from preparation_avec_retard_updated import get_user_input as get_input_preparation_avec_retard
from sans_preparation_avec_retard import get_user_input as get_input_sans_preparation_avec_retard
from preparation_sans_retard_updated import get_user_input as get_input_preparation_sans_retard
from sans_prep_sans_retard import get_user_input as get_input_sans_prep_sans_retard

from sans_prep_sans_retard import cds_sequence,seq_johnson,gantt_flowshop_with_metrics,calculate_performance_metrics_per_machine

from preparation_sans_retard_updated import gantt_flowshop_with_metrics as gantt_flowshop_with_metrics_retard
from preparation_sans_retard_updated import calculate_performance_metrics_per_machine as calculate_performance_metrics_per_machine_prep_sansretard
from preparation_sans_retard_updated import choix_sequence_croissant as choix_sequence_croissant_sansretard
from preparation_sans_retard_updated import choix_sequence_decroissant as choix_sequence_decroissant_sansretard

from sans_preparation_avec_retard import gantt_flowshop_with_metrics as gantt_flowshop_with_metrics_sansprep_retard
from sans_preparation_avec_retard import calculate_performance_metrics_per_machine as calculate_performance_metrics_per_machine_sansprep_retard
from sans_preparation_avec_retard import cds_sequence as cds_sequence_retard
from sans_preparation_avec_retard import seq_johnson as seq_johnson_retard

from preparation_avec_retard_updated import gantt_flowshop_with_metrics as gantt_flowshop_with_metrics_prepretard
from preparation_avec_retard_updated import calculate_performance_metrics_per_machine as calculate_performance_metrics_per_machine_prepretard
from preparation_avec_retard_updated import choix_sequence_croissant as choix_sequence_croissant_avecretard
from preparation_avec_retard_updated import choix_sequence_decroissant as choix_sequence_decroissant_avecretard

from SPT_LPT import get_user_input as get_input_spt_lpt, SPT, LPT, gantt_flowshop_with_metrics as gantt_spt_lpt, calculate_performance_metrics_per_machine as calculate_performance_metrics_per_machine_sptlpt 
from NO_IDLE import get_user_input as get_input_no_idle,SPT as SPT_no_idle, LPT as LPT_no_idle, gantt_flowshop_with_metrics as gantt_no_idle, calculate_performance_metrics_per_machine as calculate_performance_metrics_per_machine_no_idle


def main():
    st.title("Flow Shop Scheduling")
    st.write("Interactive scheduling application")

    # Choose the scheduling type
    scheduling_type = st.selectbox("Choose Scheduling Type:", ["Flow Shop", "Job Shop"])    
    if scheduling_type == "Flow Shop":
        st.write("You selected Flow Shop Scheduling.")

        # Sub-choice: Preparation/Delay vs Idle/No Idle
        main_choice = st.radio("Choose Analysis Type:", ["Preparation and Delay", "SPT/LPT"])
        # If "Preparation and Delay" is selected
        if main_choice == "Preparation and Delay":
            # Options for flow shop
            with_prep = st.radio("Include Preparation?", ["Yes", "No"])
            with_delay = st.radio("Include Delay?", ["Yes", "No"])

            if with_prep == "Yes" and with_delay == "Yes":
                st.write("Flow Shop with Preparation and Delay.")
                P, S, due_dates, m, n, order = get_input_preparation_avec_retard()
                # Call relevant processing functions
                if P is None or S is None or due_dates is None:
                    st.warning("Les paramètres sont incorrects. Veuillez corriger vos entrées.")
                else:
                    # Déterminer la séquence
                    seq=None
                    # Déterminer la séquence en fonction de l'ordre choisi
                    if order == "Croissant":
                        seq = choix_sequence_croissant_avecretard(P, S)
                        st.write("### Séquence choisie : Croissant")
                    else:
                        seq = choix_sequence_decroissant_avecretard(P, S)
                        st.write("### Séquence choisie : Décroissant")

                    # Calculer les métriques et afficher le Gantt
                    TFT, TT, Cmax, C = gantt_flowshop_with_metrics_prepretard(P, S, due_dates, seq)
                    sequence_modifiee = [int(x) for x in seq]

                    # Affichage des résultats
                    st.write("### Résultats")
                    st.write(f"Séquence optimale : {sequence_modifiee}")
                    st.write(f"Makespan (Cmax) : {Cmax}")
                    st.write(f"Total Flow Time (TFT) : {TFT}")
                    st.write(f"Total Tardiness (TT) : {TT}")
                    st.dataframe(C)

                    # Calcul et affichage des performances
                    TFR, TAR, TAP = calculate_performance_metrics_per_machine_prepretard(C, P, S, m, n, seq, Cmax)
                    
                    # Création du DataFrame
                    performance_df = pd.DataFrame({
                    "TFR (%)": [f"{tfr:.3f}" for tfr in TFR],
                    "TAR (%)": [f"{tar:.3f}" for tar in TAR],
                        "TAP (%)": [f"{tap:.3f}" for tap in TAP]
                    })
                    performance_df.index = performance_df.index + 1
                
                    # Affichage avec st.table
                    st.write("### Performances par machine")
                    st.table(performance_df)
                        

            elif with_prep == "No" and with_delay == "Yes":
                st.write("Flow Shop without Preparation but with Delay.")
                algo_choice = st.radio("Choisissez l'algorithme :", ["Johnson", "CDS"])
                P, due_dates, k, m, n = get_input_sans_preparation_avec_retard()
                # Vérification des entrées
                if P is None or due_dates is None:
                    st.warning("Les paramètres sont incorrects. Veuillez corriger vos entrées.")
                else:
                    # Calculer la séquence et les métriques selon le choix de l'algorithme
                    if algo_choice == "Johnson":
                        if m > 2:
                            st.warning("L'algorithme Johnson est conçu pour 2 machines uniquement.")
                        else:
                            seq = seq_johnson_retard(P)
                            st.write("### Algorithme : Johnson")
                    elif algo_choice == "CDS":
                        if m <= 2:
                            st.warning("L'algorithme CDS est plus adapté pour 3 machines ou plus.")
                        else:
                            seq = cds_sequence_retard(P, k)
                            st.write("### Algorithme : CDS")
            

                    # Calculer et afficher les métriques
                    TFT, TT, Cmax, C = gantt_flowshop_with_metrics_sansprep_retard(P, due_dates, seq)
                    sequence_modifiee = [int(x) + 1 for x in seq]

                    # Afficher les résultats
                    st.write("### Résultats")
                    st.write(f"Séquence optimale : {sequence_modifiee}")
                    st.write(f"Makespan (Cmax) : {Cmax}")
                    st.write(f"Total Tardiness (TT) : {TT}")
                    st.write(f"Total Flow Time (TFT) : {TFT}")
                    st.dataframe(C)

                    # Calcul des performances
                    TFR, TAR = calculate_performance_metrics_per_machine_sansprep_retard(P, m, n, Cmax)
                    # Création du DataFrame
                    performance_df = pd.DataFrame({
                    "TFR (%)": [f"{tfr:.3f}" for tfr in TFR],
                    "TAR (%)": [f"{tar:.3f}" for tar in TAR]
                    })
                    performance_df.index = performance_df.index + 1

                    # Affichage avec st.table
                    st.write("### Performances par machine")
                    st.table(performance_df)
                    

            elif with_prep == "Yes" and with_delay == "No":
                st.write("Flow Shop with Preparation but without Delay.")
                P, S, m, n, order = get_input_preparation_sans_retard()
                # Vérification des entrées
                if P is None or S is None:
                    st.warning("Les paramètres sont incorrects. Veuillez corriger vos entrées.")
                else:
                # Déterminer la séquence
                    seq=None
                    if order == "Croissant":
                        seq = choix_sequence_croissant_sansretard(P, S)
                        st.write("### Algorithme : Séquence croissante")
                    else:
                        seq = choix_sequence_decroissant_sansretard(P, S)
                        st.write("### Algorithme : Séquence décroissante")
            
                    # Calculer et afficher les métriques
                    TFT, Cmax, C = gantt_flowshop_with_metrics_retard(P, S, seq)
                    sequence_modifiee = [int(x) for x in seq]

                    # Afficher les résultats
                    st.write("### Résultats")
                    st.write(f"Séquence optimale : {sequence_modifiee}")
                    st.write(f"Makespan (Cmax) : {Cmax}")
                    st.write(f"Total Flow Time (TFT) : {TFT}")
                    st.dataframe(C)

                    # Calcul et affichage des performances
                    TFR, TAR, TAP = calculate_performance_metrics_per_machine_prep_sansretard(C, P, S, m, n, seq, Cmax)
                    # Création du DataFrame
                    performance_df = pd.DataFrame({
                    "TFR (%)": [f"{tfr:.3f}" for tfr in TFR],
                    "TAR (%)": [f"{tar:.3f}" for tar in TAR],
                        "TAP (%)": [f"{tap:.3f}" for tap in TAP]
                    })
                    performance_df.index = performance_df.index + 1

                    # Affichage avec st.table
                    st.write("### Performances par machine")
                    st.table(performance_df)


            elif with_prep == "No" and with_delay == "No":
                st.write("Flow Shop without Preparation and without Delay.")
                # Ajouter un choix entre les algorithmes
                algo_choice = st.radio("Choisissez l'algorithme :", ["Johnson", "CDS"])
                P, k, m, n = get_input_sans_prep_sans_retard()
                # Vérification des entrées
                if P is None:
                    st.warning("Les paramètres sont incorrects. Veuillez corriger vos entrées.")
                else:
                    # Calculer la séquence et les métriques selon le choix de l'algorithme
                    if algo_choice == "Johnson":
                        if m > 2:
                            st.warning("L'algorithme Johnson est conçu pour 2 machines uniquement.")
                        else:
                            seq = seq_johnson(P)
                            st.write("### Algorithme : Johnson")
                    elif algo_choice == "CDS":
                        if m <= 2:
                            st.warning("L'algorithme CDS est plus adapté pour 3 machines ou plus.")
                        else:
                            seq = cds_sequence(P, k)
                            st.write("### Algorithme : CDS")
        
                    TFT, Cmax, C = gantt_flowshop_with_metrics(P, seq)
                    sequence_modifiee = [x + 1 for x in seq]

                    # Afficher les résultats
                    st.write("### Résultats")
                    st.write(f"Séquence optimale : {sequence_modifiee}")
                    st.write(f"Makespan (Cmax) : {Cmax}")
                    st.write(f"Total Flow Time (TFT) : {TFT}")
                    st.dataframe(C)

                    # Calcul et affichage des performances
                    TFR, TAR= calculate_performance_metrics_per_machine(P, m, n,Cmax)
                    # Création du DataFrame
                    performance_df = pd.DataFrame({
                    "TFR (%)": [f"{tfr:.3f}" for tfr in TFR],
                    "TAR (%)": [f"{tar:.3f}" for tar in TAR]
                    })
                    performance_df.index = performance_df.index + 1

                    # Affichage avec st.table
                    st.write("### Performances par machine")
                    st.table(performance_df)
                    
        # If "Idle and No Idle" is selected
        elif main_choice == "SPT/LPT":
            # Step 1: Choose Sequence Type (SPT or LPT)
            seq_type = st.radio("Choose Sequence Type:", ["SPT", "LPT"])
            activate_no_idle = st.radio("Activate No Idle?", ["Yes", "No"])
            # Step 3: Get inputs using `get_user_input`
            P, m, n = get_input_no_idle()  # Handles input collection and validation
            if seq_type=="SPT" and activate_no_idle=="Yes":
                seq=SPT_no_idle(P)
                print(f"Sequence (seq): {seq}")
                print(f"Processing times matrix (P) shape: {P.shape}")
                st.write("### Gant Chart")
                TFT, Cmax,C= gantt_no_idle(P,seq)
                sequence_modifiee = [int(x) + 1 for x in seq]
                st.write(f"Optimized Sequence : {sequence_modifiee}")
                st.write(f"No Idle Makespan (Cmax): {Cmax}")
                st.write(f"Total Flow Time (TFT) : {TFT}")
                st.write("### Completion Time Matrix (C)")
                st.dataframe(C)
                # Calcul des performances
                TFR, TAR = calculate_performance_metrics_per_machine_no_idle(P, m, n, Cmax)
                # Création du DataFrame
                performance_df = pd.DataFrame({
                "TFR (%)": [f"{tfr:.3f}" for tfr in TFR],
                "TAR (%)": [f"{tar:.3f}" for tar in TAR]
                })
                performance_df.index = performance_df.index + 1
                # Affichage avec st.table
                st.write("### Performances par machine")
                st.table(performance_df)

            elif seq_type=="SPT" and activate_no_idle=="No":
                seq=SPT(P)
                st.write("### Gant Chart")
                TFT, Cmax,C= gantt_spt_lpt(P,seq)
                sequence_modifiee = [int(x) for x in seq]
                st.write(f"Optimized Sequence : {sequence_modifiee}")
                st.write(f"No Idle Makespan (Cmax): {Cmax}")
                st.write(f"Total Flow Time (TFT) : {TFT}")
                st.write("### Completion Time Matrix (C)")
                st.dataframe(C)
                # Calcul des performances
                TFR, TAR = calculate_performance_metrics_per_machine_sptlpt(P, m, n, Cmax)
                # Création du DataFrame
                performance_df = pd.DataFrame({
                "TFR (%)": [f"{tfr:.3f}" for tfr in TFR],
                "TAR (%)": [f"{tar:.3f}" for tar in TAR]
                })
                performance_df.index = performance_df.index + 1
                # Affichage avec st.table
                st.write("### Performances par machine")
                st.table(performance_df)
            elif seq_type=="LPT" and activate_no_idle=="Yes":
                seq=LPT_no_idle(P)
                st.write("### Gant Chart")
                TFT, Cmax,C= gantt_no_idle(P,seq)
                sequence_modifiee = [int(x) + 1 for x in seq]
                st.write(f"Optimized Sequence : {sequence_modifiee}")
                st.write(f"No Idle Makespan (Cmax): {Cmax}")
                st.write(f"Total Flow Time (TFT) : {TFT}")
                st.write("### Completion Time Matrix (C)")
                st.dataframe(C)
                # Calcul des performances
                TFR, TAR = calculate_performance_metrics_per_machine_no_idle(P, m, n, Cmax)
                # Création du DataFrame
                performance_df = pd.DataFrame({
                "TFR (%)": [f"{tfr:.3f}" for tfr in TFR],
                "TAR (%)": [f"{tar:.3f}" for tar in TAR]
                })
                performance_df.index = performance_df.index + 1
                # Affichage avec st.table
                st.write("### Performances par machine")
                st.table(performance_df)
            elif seq_type=="LPT" and activate_no_idle=="No": 
                seq=LPT(P) 
                st.write("### Gant Chart") 
                TFT, Cmax,C= gantt_spt_lpt(P,seq)
                sequence_modifiee = [int(x) for x in seq]
                st.write(f"Optimized Sequence : {sequence_modifiee}")
                st.write(f"No Idle Makespan (Cmax): {Cmax}")
                st.write(f"Total Flow Time (TFT) : {TFT}")
                st.write("### Completion Time Matrix (C)")
                st.dataframe(C)
                # Calcul des performances
                TFR, TAR = calculate_performance_metrics_per_machine_sptlpt(P, m, n, Cmax)
                # Création du DataFrame
                performance_df = pd.DataFrame({
                "TFR (%)": [f"{tfr:.3f}" for tfr in TFR],
                "TAR (%)": [f"{tar:.3f}" for tar in TAR]
                })
                performance_df.index = performance_df.index + 1
                # Affichage avec st.table
                st.write("### Performances par machine")
                st.table(performance_df)     

    else:
        st.write("Job Shop Scheduling is not implemented yet.")

if __name__ == "__main__":
    main()
