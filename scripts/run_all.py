"""

    run_all.py

    Run full analysis for all validated sessions.
    
    SESSIONS
    * 'Early' and 'Late'
        - 071102, 071107, 071109, 071213, 071214, 071218, 071221, 071227, 080213, 071228, 080206, 080208, 080227, 080220, 080222

    PARAMETERS
    - min_neu: Minimum number of neurons in a profile
    - corr_thresh: Correlation threshold for creating lax correlation graphs
    - corr_thresh_strict: Correlation threshold for creating strict correlation graphs
    - bin_width: Bin width for correlation measure
    - num_shuffles: Number of shuffles graphs to create

    OUTPUT STRUCTURE    
    ├── ../results
    │   ├── /YYYY-MM-DD
    │   │   ├── /run_all
    │   │   │   ├── /[SESSION]
    │   │   │   │   ├── /Profile
    │   │   │   │   ├── /Shuffle
    │   │   │   │   ├── /Correlation
    
"""

import os
from datetime import datetime
import graphcheck as gc
import pandas as pd


# ---------------------------
# 15 sessions: Early + Late
sessions = ["071102", "071107", "071109", "071213", "071214", "071218", "071221", "071227", "080213", "071228", "080206", "080208", "080227", "080220", "080222"]


min_neu = 3
corr_thresh = 0.2
corr_thresh_strict = 0.5
corr_bin_width = 100
num_shuffles = 10

date_str = datetime.today().strftime('%Y-%m-%d')
date_time_str = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

file_name = "run_all"
path_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + "/"
path_out = path_local + "results/" + date_str + "/" + file_name + "/"
if not os.path.exists(path_out):
    os.makedirs(path_out)


# === IMPORTANT ===
# This analysis needs both raw and profile data: use `spykesim-lite` to generate the profile data first
path_data_root = "/path/to/data/root/"  # replace with the actual path to your data root
path_prof_root = "/path/to/profile_data/root/"  # replace with the actual path to your profile data root


alphas = ["0.0693", "0.1386"]

for alpha in alphas:
    for session in sessions:
        print("Initializing session " + session + " with alpha " + alpha)

        config = gc.Config(path_data_root, path_prof_root, session, alpha)
        path_out_session = path_out + session + "/"

        if not os.path.exists(path_out_session):
            os.makedirs(path_out_session)
            
        loaded_data_both = []

        for stage in ["Early", "Late"]: 
            print("Initializing data for " + stage + " stage")

            profile_dict = gc.open_dict(
                config.path_profiles_dict[stage], 
                config.filename_dict[stage]
            )

            loaded_data = gc.LoadData(session, min_neu, stage, alpha, config.path_data)
            loaded_data.load_profiles(profile_dict)
            loaded_data.import_lever_data(config.path_data)
            loaded_data.import_event_data(config.path_data)
            loaded_data.import_spikes(config.path_data)
            loaded_data_both.append(loaded_data)

        # # Profile

        path_out_profile = path_out_session + "Profile/"
        if not os.path.exists(path_out_profile):
            os.makedirs(path_out_profile)

        profile_graphs = {}
        profile_graphs["Early"] = gc.DataGraphProfile(loaded_data_both[0])
        profile_graphs["Late"] = gc.DataGraphProfile(loaded_data_both[1])
        
        for stage in ["Early", "Late"]: 
            print("Analyzing data for " + stage + " stage")
            profile_graphs[stage].create_all()
            profile_graphs[stage].write_report_terminal()

        persistent_assemblies_profile = gc.find_persistent_assemblies(profile_graphs["Early"], profile_graphs["Late"])

        for stage in ["Early", "Late"]: 
            print("Exporting data for " + stage + " stage")
            profile_graphs[stage].export_all(path_out_profile)
            profile_graphs[stage].plot_all(path_out_profile)

        # Shuffle

        path_out_shuffle = path_out_session + "Shuffle/"
        if not os.path.exists(path_out_shuffle):
            os.makedirs(path_out_shuffle)

        all_shuffle_graphs = []
        all_persistent_assemblies_shuffle = []

        for i in range(num_shuffles):
            shuffle_graphs = {}
            shuffle_graphs["Early"] = gc.DataGraphShuffle(loaded_data_both[0], random_seed = i)
            shuffle_graphs["Late"] = gc.DataGraphShuffle(loaded_data_both[1], random_seed = i)

            for stage in ["Early", "Late"]:
                shuffle_graphs[stage].create_all()
                shuffle_graphs[stage].write_report_terminal()

            persistent_assemblies_shuffle = gc.find_persistent_assemblies(shuffle_graphs["Early"], shuffle_graphs["Late"])
            all_persistent_assemblies_shuffle.append(persistent_assemblies_shuffle)

            for stage in ["Early", "Late"]:
                shuffle_graphs[stage].export_all(path_out_shuffle)
                shuffle_graphs[stage].plot_all(path_out_shuffle)
            
            all_shuffle_graphs.append(shuffle_graphs)

        # Correlation

        path_out_correlation = path_out_session + "Correlation/"
        if not os.path.exists(path_out_correlation):
            os.makedirs(path_out_correlation)

        correlation_graphs = {}
        correlation_graphs["Early"] = gc.DataGraphCorrelation(loaded_data_both[0], corr_bin_width = corr_bin_width, corr_thresh = corr_thresh)
        correlation_graphs["Late"] = gc.DataGraphCorrelation(loaded_data_both[1], corr_bin_width = corr_bin_width, corr_thresh = corr_thresh)

        for stage in ["Early", "Late"]:
            correlation_graphs[stage].create_all()
            correlation_graphs[stage].write_report_terminal()

        persistent_assemblies_correlation = gc.find_persistent_assemblies(correlation_graphs["Early"], correlation_graphs["Late"])

        for stage in ["Early", "Late"]:    
            correlation_graphs[stage].export_all(path_out_correlation)
            correlation_graphs[stage].plot_all(path_out_correlation)

        correlation_graphs_strict = {}
        correlation_graphs_strict["Early"] = gc.DataGraphCorrelation(loaded_data_both[0], corr_bin_width = corr_bin_width, corr_thresh = corr_thresh_strict)
        correlation_graphs_strict["Late"] = gc.DataGraphCorrelation(loaded_data_both[1], corr_bin_width = corr_bin_width, corr_thresh = corr_thresh_strict)

        for stage in ["Early", "Late"]:
            correlation_graphs_strict[stage].create_all()
            correlation_graphs_strict[stage].write_report_terminal()

        persistent_assemblies_correlation_strict = gc.find_persistent_assemblies(correlation_graphs_strict["Early"], correlation_graphs_strict["Late"])

        for stage in ["Early", "Late"]:
            correlation_graphs_strict[stage].export_all(path_out_correlation)
            correlation_graphs_strict[stage].plot_all(path_out_correlation)

        # Export

        all_data = list(profile_graphs.values()) + list(correlation_graphs.values()) + list(correlation_graphs_strict.values())
        for shuffle_graphs in all_shuffle_graphs:
            all_data += list(shuffle_graphs.values())
        df_stages = gc.create_stages_df(all_data)
        gc.export_stages_df(df_stages, path_out, alpha)

        concatenated_df_neuron = pd.concat([datagraph.df_neuron for datagraph in all_data]).reset_index(drop=True)
        concatenated_df_edge = pd.concat([datagraph.df_edge for datagraph in all_data]).reset_index(drop=True)
        concatenated_df_profile = pd.concat([datagraph.df_profile for datagraph in all_data]).reset_index(drop=True)
        concatenated_df_reward = pd.concat([datagraph.df_reward for datagraph in all_data]).reset_index(drop=True)
        concatenated_df_reward_combined = pd.concat([datagraph.df_reward_combined for datagraph in all_data]).reset_index(drop=True)

        gc.export_concatenated_df(concatenated_df_neuron, path_out, "neuron", alpha)
        gc.export_concatenated_df(concatenated_df_edge, path_out, "edge", alpha)
        gc.export_concatenated_df(concatenated_df_profile, path_out, "profile", alpha)
        gc.export_concatenated_df(concatenated_df_reward, path_out, "reward", alpha)
        gc.export_concatenated_df(concatenated_df_reward_combined, path_out, "reward_combined", alpha)

        all_data_comparison = [profile_graphs, correlation_graphs, correlation_graphs_strict]
        for shuffle_graphs in all_shuffle_graphs:
            all_data_comparison.append(shuffle_graphs)
        all_persistent_assemblies = [persistent_assemblies_profile, persistent_assemblies_correlation, persistent_assemblies_correlation_strict] + all_persistent_assemblies_shuffle

        df_comparison = gc.create_comparison_df(all_data_comparison, all_persistent_assemblies)
        gc.export_comparison_df(df_comparison, path_out, alpha)