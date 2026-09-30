import pandas as pd


def import_run_all(sessions, alpha, path, path_movement, export=False):
    df_stage = pd.DataFrame()
    df_comparison = pd.DataFrame()
    df_edge_concat = pd.DataFrame()
    df_neuron_concat = pd.DataFrame()
    df_profile_concat = pd.DataFrame()
    df_reward_concat = pd.DataFrame()
    df_reward_combined_concat = pd.DataFrame()


    for session in sessions:
        df_stage_session = pd.read_csv(path + f"{session}_{alpha}_df_stages.csv", index_col=0)
        df_edge_concat_session = pd.read_csv(path + f"{session}_{alpha}_df_edge_concatenated.csv", index_col=0)
        df_neuron_concat_session = pd.read_csv(path + f"{session}_{alpha}_df_neuron_concatenated.csv", index_col=0)
        df_profile_concat_session = pd.read_csv(path + f"{session}_{alpha}_df_profile_concatenated.csv", index_col=0)
        df_reward_concat_session = pd.read_csv(path + f"{session}_{alpha}_df_reward_concatenated.csv", index_col=0)
        df_reward_combined_concat_session = pd.read_csv(path + f"{session}_{alpha}_df_reward_combined_concatenated.csv", index_col=0)
        df_comparison_session = pd.read_csv(path + f"{session}_{alpha}_df_comparison.csv", index_col=0)

        df_stage = pd.concat([df_stage, df_stage_session]).reset_index(drop=True)
        df_edge_concat = pd.concat([df_edge_concat, df_edge_concat_session]).reset_index(drop=True)
        df_neuron_concat = pd.concat([df_neuron_concat, df_neuron_concat_session]).reset_index(drop=True)
        df_profile_concat = pd.concat([df_profile_concat, df_profile_concat_session]).reset_index(drop=True)
        df_reward_concat = pd.concat([df_reward_concat, df_reward_concat_session]).reset_index(drop=True)
        df_reward_combined_concat = pd.concat([df_reward_combined_concat, df_reward_combined_concat_session]).reset_index(drop=True)
        df_comparison = pd.concat([df_comparison, df_comparison_session]).reset_index(drop=True)


    # adding movement categories: motivated/unmotivated, rewarded/unrewarded
    df_movement_classes = pd.read_csv(path_movement + "allsessions_success_failure_clean.csv", index_col=0)
    df_movement_classes_short = df_movement_classes[["Session", "Stage", "Motivation", "Reward", "Category"]]


    # adding profile lengths
    df_profiles_data = df_profile_concat[df_profile_concat["Method"] == "Profile"].copy()
    df_profile_concat_short = df_profiles_data[["Session", "Stage", "Alpha", "Profile ID", "Length"]].copy()
    df_profile_concat_short_unique = df_profile_concat_short.drop_duplicates()


    # adding category columns to other dataframes
    df_reward_concat_full = df_reward_concat.merge(df_movement_classes_short, on=("Session", "Stage"), how="left")
    df_reward_combined_concat_full = df_reward_combined_concat.merge(df_movement_classes_short, on=("Session", "Stage"), how="left")


    df_stage = df_stage.merge(df_movement_classes_short, on=("Session", "Stage"), how="left")
    df_neuron_concat = df_neuron_concat.merge(df_movement_classes_short, on=("Session", "Stage"), how="left")


    # adding profile lengths to reward dataframes
    df_reward_concat_prof_len = df_reward_concat.merge(df_profile_concat_short_unique, on=("Session", "Stage", "Alpha", "Profile ID"), how="left")


    # not sure if this is necessary
    df_profile_concat["Session"] = df_profile_concat["Session"].astype(str)

    
    # additional columns for plots
    df_stage["% Neurons in Profiles"] = 100 * df_stage["# Neurons in Graph"] / df_stage["# Neurons Below FR Max"]
    df_stage["% Superficial Profiles"] = 100 * df_stage["# Shallow Profiles"] / df_stage["# Profiles"]
    df_stage["% Deep Profiles"] = 100 * df_stage["# Deep Profiles"] / df_stage["# Profiles"]
    df_stage["% Both Layers"] = 100 * (df_stage["# Profiles"] - df_stage["# Deep Profiles"] - df_stage["# Shallow Profiles"]) / df_stage["# Profiles"]


    if export:
        # exporting concatenated dataframes
        df_neuron_concat.to_csv(path + f"ALLSESSIONS_{alpha}_df_neuron_concatenated_full.csv")
        df_edge_concat.to_csv(path + f"ALLSESSIONS_{alpha}_df_edge_concatenated_full.csv")
        df_profile_concat.to_csv(path + f"ALLSESSIONS_{alpha}_df_profile_concatenated_full.csv")
        df_stage.to_csv(path + f"ALLSESSIONS_{alpha}_df_stages_concatenated.csv")
        df_comparison.to_csv(path + f"ALLSESSIONS_{alpha}_df_comparison_concatenated.csv")
        df_reward_concat_full.to_csv(path + f"ALLSESSIONS_{alpha}_df_reward_concatenated_full.csv")
        df_reward_combined_concat_full.to_csv(path + f"ALLSESSIONS_{alpha}_df_reward_combined_concatenated_full.csv")    


    # changing method names
    df_stage.loc[:, "Method"] = df_stage["Method"].apply(lambda x: "Shuffle" if x.startswith("Shuffle") else x)
    df_neuron_concat.loc[:, "Method"] = df_neuron_concat["Method"].apply(lambda x: "Shuffle" if x.startswith("Shuffle") else x)
    df_edge_concat.loc[:, "Method"] = df_edge_concat["Method"].apply(lambda x: "Shuffle" if x.startswith("Shuffle") else x)
    df_comparison.loc[:, "Method"] = df_comparison["Method"].apply(lambda x: "Shuffle" if x.startswith("Shuffle") else x)


    df_stage.replace("Correlation_BW_100_TH_0.2", "Corr/Lax", inplace=True)
    df_stage.replace("Correlation_BW_100_TH_0.5", "Corr/Strict", inplace=True)
    df_neuron_concat.replace("Correlation_BW_100_TH_0.2", "Corr/Lax", inplace=True)
    df_neuron_concat.replace("Correlation_BW_100_TH_0.5", "Corr/Strict", inplace=True)
    df_edge_concat.replace("Correlation_BW_100_TH_0.2", "Corr/Lax", inplace=True)
    df_edge_concat.replace("Correlation_BW_100_TH_0.5", "Corr/Strict", inplace=True)


    return df_stage, df_comparison, df_neuron_concat, df_edge_concat, df_profile_concat, df_reward_concat_full, df_reward_combined_concat_full, df_reward_concat_prof_len