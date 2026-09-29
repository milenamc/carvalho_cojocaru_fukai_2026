import numpy as np
import pandas as pd
import networkx as nx
from typing import List

from .graph import Graph
from .utils import calculate_success_failure, find_all_nearest_sequences


def create_df_neuron(graph: Graph, session: str, min_neu: int, stage: str, 
                     alpha: str, method: str):
    """
    Creates neuron dataframe
    """
    dataframe = [
        {
            "Session": session,
            "Min Profile Length": min_neu,
            "Stage": stage,
            "Alpha": alpha,
            "Method": method,
            "ID": node_id,
            "Electrode": graph.G.nodes[node_id].get('electrode', None),
            "Degree": attr['degree'],
            "Betweenness Centrality": attr['betweenness_cent'],
            "Degree Centrality": attr['degree_cent'],
            "Eigenvector Centrality": attr['eigenvector_cent'],
            "Clustering": attr['clustering'],
            "Weighted Clustering": attr['weighted_clustering'],
            "Pagerank": attr['pagerank'],
            "Norm. Hub Value": attr['hub_value'],
            "Norm. Authority Value": attr['authority_value'],
            "Community": graph.G.nodes[node_id].get('community', None)
        }
        for node_id, attr in graph.attributes.items()
    ]

    df_neuron = pd.DataFrame(dataframe, columns=[
        "Session", "Min Profile Length", "Stage", "Alpha", 
        "Method", "ID", "Electrode", "Degree",
        "Betweenness Centrality", "Degree Centrality", "Eigenvector Centrality",
        "Clustering", "Weighted Clustering", "Pagerank", "Norm. Hub Value", 
        "Norm. Authority Value", "Community"
    ])

    return df_neuron


def create_df_edge(graph: Graph, end_elect1: int, session: str, min_neu: int, 
                   stage: str, alpha: str, method: str):
    """
    Creates edge dataframe
    """
    G = graph.G
    edgedict = nx.get_edge_attributes(G, 'weight')
    normdict = nx.get_edge_attributes(G, 'norm_weight')

    pairs = list(edgedict.keys())
    weights = list(edgedict.values())
    norm_weights = list(normdict.values())

    dataframe = []

    for i in range(len(pairs)):
        if ((pairs[i][0] <= end_elect1) and (pairs[i][1] <= end_elect1)):
            edge_location = "Superficial"
        elif ((pairs[i][0] > end_elect1) and (pairs[i][1] > end_elect1)):
            edge_location = "Deep"
        else:
            edge_location = "Crosslayer"

        row = {
            "Session": session,
            "Min Profile Length": min_neu,
            "Stage": stage,
            "Alpha": alpha,
            "Method": method,
            "IDs": tuple(sorted(pairs[i])),
            "Weight": weights[i],
            "Norm Weight": norm_weights[i],
            "Location": edge_location
        }
        
        dataframe.append(row)

    df_edge = pd.DataFrame(dataframe, columns=[
        "Session", "Min Profile Length", "Stage", "Alpha", 
        "Method", "IDs", "Weight", "Norm Weight", "Location"
    ])

    return df_edge


def create_df_profile(profile_dict: dict, session: str, min_neu: int, stage: str, 
                      alpha: str, method: str):
    """
    Creates profile dataframe
    """
    n_sup_profiles = 0
    n_deep_profiles = 0

    neuron_per_profile = []
    count_per_profile = [] 

    dataframe = []

    for idx, profile in enumerate(profile_dict.values()):
        if profile['layer'] == 1:
            location = "Superficial"
            n_sup_profiles += 1
        elif profile['layer'] == 2:
            location = "Deep"
            n_deep_profiles += 1
        else:
            location = "Crosslayer"

        neuron_per_profile.append(profile['size'])
        count_per_profile.append(len(profile['times']))

        row = {
            "Session": session,
            "Min Profile Length": min_neu,
            "Stage": stage,
            "Alpha": alpha,
            "Method": method,
            "Profile ID": idx,
            "IDs": profile['neurons'].tolist(),
            "Length": profile['size'],
            "Spike count": profile['spkcount'],
            "Location": location,
            "Sequence count": len(profile['times']),
            "Times (ms)": profile['times'].tolist(),
        }

        dataframe.append(row)

    df_profile = pd.DataFrame(dataframe, columns=[
        "Session", "Min Profile Length", "Stage", "Alpha", "Method", 
        "Profile ID", "IDs", "Length", "Spike count", 
        "Location", "Sequence count", "Times (ms)"
    ])

    profile_stats = {
        "n_profiles": len(profile_dict),
        "n_neurons": neuron_per_profile, "seq_count": count_per_profile,
        "sup_prof": n_sup_profiles, "deep_prof": n_deep_profiles
    }

    return df_profile, profile_stats


def create_df_profile_shuffle(shuffled_profiles: List[List[int]], end_elect1: int, 
                              session: str, min_neu: int, stage: str, 
                              alpha: str, method: str):
    """
    Creates profile dataframe for shuffled profiles
    """  
    n_sup_profiles = 0
    n_deep_profiles = 0

    neuron_per_profile = []
    count_per_profile = [] 

    dataframe = []

    for idx, shuffled_profile in enumerate(shuffled_profiles):
        neuron_per_profile.append(len(shuffled_profile))
        count_per_profile.append(0)

        if all([x <= end_elect1 for x in shuffled_profile]):
            location = "Superficial"
            n_sup_profiles += 1
        elif all([x > end_elect1 for x in shuffled_profile]):
            location = "Deep"
            n_deep_profiles += 1
        else:
            location = "Crosslayer"

        row = {
            "Session": session,
            "Min Profile Length": min_neu,
            "Stage": stage,
            "Alpha": alpha,
            "Method": method,
            "Profile ID": idx,
            "IDs": shuffled_profile,
            "Length": len(shuffled_profile),
            "Spike count": None,
            "Location": location,
            "Sequence count": None,
            "Times (ms)": None,
            
        }  

        dataframe.append(row)

    df_profile = pd.DataFrame(dataframe, columns=[
        "Session", "Min Profile Length", "Stage", "Alpha", "Method", 
        "Profile ID", "IDs", "Length", "Spike count", 
        "Location", "Sequence count", "Times (ms)"
    ])

    profile_stats = {
        "n_profiles": len(shuffled_profiles),
        "n_neurons": neuron_per_profile, "seq_count": count_per_profile,
        "sup_prof": n_sup_profiles, "deep_prof": n_deep_profiles
    }

    return df_profile, profile_stats


def create_df_reward(profile_dict: dict, event: np.ndarray, session: str, 
                     min_neu: int, stage: str, alpha: str, method: str,
                     interval_sec: float = 1.0): # previously 1.5
    times_success, times_failure = calculate_success_failure(event)

    df_reward_row = []

    for i, profile in enumerate(profile_dict.values()): # i: profile ID
        profile_times = profile['times'].tolist() # previously: [x + start_time_ms for x in profile['times']] 

        for j, reward_time in enumerate(times_success): # j: reward ID
            nearest_sequences = find_all_nearest_sequences(profile_times, reward_time, interval_sec)

            for found_sequence in nearest_sequences:
                # [0]: sequence time, [1]: time distance to reward

                df_reward_row.append({
                    "Session": session, 
                    "Min Profile Length": min_neu, 
                    "Stage": stage, 
                    "Alpha": alpha,
                    "Method": method,
                    "Profile ID": i, 
                    "Reward ID": j, 
                    "Event": "Success_Signal", 
                    "Event Time (ms)": reward_time,
                    "Sequence Time (ms)": found_sequence[0], 
                    "Time Distance (ms)": found_sequence[1]
                })

            # 200 ms before reward signal time, lever movement is assumed to be completed
            movement_time = reward_time - 200
            nearest_sequences = find_all_nearest_sequences(profile_times, movement_time, interval_sec)

            for found_sequence in nearest_sequences:
                # [0]: sequence time, [1]: time distance to reward

                df_reward_row.append({
                    "Session": session, 
                    "Min Profile Length": min_neu, 
                    "Stage": stage,
                    "Alpha": alpha, 
                    "Method": method,
                    "Profile ID": i, 
                    "Reward ID": j, 
                    "Event": "Success", 
                    "Event Time (ms)": movement_time,
                    "Sequence Time (ms)": found_sequence[0], 
                    "Time Distance (ms)": found_sequence[1]
                })
        
        for k, failure_time in enumerate(times_failure): 
            nearest_sequences = find_all_nearest_sequences(profile_times, failure_time, interval_sec)

            for found_sequence in nearest_sequences:
                # [0]: sequence time, [1]: time distance to reward

                df_reward_row.append({
                    "Session": session, 
                    "Min Profile Length": min_neu, 
                    "Stage": stage,
                    "Alpha": alpha,
                    "Method": method, 
                    "Profile ID": i, 
                    "Reward ID": k, 
                    "Event": "Failure", 
                    "Event Time (ms)": failure_time, 
                    "Sequence Time (ms)": found_sequence[0], 
                    "Time Distance (ms)": found_sequence[1]
                })

    df_reward = pd.DataFrame(df_reward_row, columns=[
        "Session", "Min Profile Length", "Stage", "Alpha", "Method", 
        "Profile ID", "Reward ID", "Event", "Event Time (ms)", 
        "Sequence Time (ms)", "Time Distance (ms)"
    ])

    return df_reward


def create_df_reward_combined(profile_dict: dict, event: np.ndarray, session: str, 
                              min_neu: int, stage: str, alpha: str, method: str,
                              interval_sec: float = 1.0): # previously 1.5
    times_success, times_failure = calculate_success_failure(event)

    df_reward_row = []

    profile_times_combined = []
    for i, profile in enumerate(profile_dict.values()): # i: profile ID
        profile_times_combined += profile['times'].tolist() # previously: [x + start_time_ms for x in profile['times']]
    profile_times = np.unique(profile_times_combined)

    for j, reward_time in enumerate(times_success): # j: reward ID
        nearest_sequences = find_all_nearest_sequences(profile_times, reward_time, interval_sec)

        for found_sequence in nearest_sequences:
            # [0]: sequence time, [1]: time distance to reward

            df_reward_row.append({
                "Session": session, 
                "Min Profile Length": min_neu, 
                "Stage": stage, 
                "Alpha": alpha,
                "Method": method,
                "Reward ID": j, 
                "Event": "Success_Signal", 
                "Event Time (ms)": reward_time,
                "Sequence Time (ms)": found_sequence[0], 
                "Time Distance (ms)": found_sequence[1]
            })

        movement_time = reward_time - 200 # 200 ms before reward signal time, lever movement completed
        nearest_sequences = find_all_nearest_sequences(profile_times, movement_time, interval_sec)

        for found_sequence in nearest_sequences:
            # [0]: sequence time, [1]: time distance to reward

            df_reward_row.append({
                "Session": session, 
                "Min Profile Length": min_neu, 
                "Stage": stage, 
                "Alpha": alpha,
                "Method": method,
                "Reward ID": j, 
                "Event": "Success", 
                "Event Time (ms)": movement_time,
                "Sequence Time (ms)": found_sequence[0], 
                "Time Distance (ms)": found_sequence[1]
            })
    
    for k, failure_time in enumerate(times_failure): 
        nearest_sequences = find_all_nearest_sequences(profile_times, failure_time, interval_sec)

        for found_sequence in nearest_sequences:
            # [0]: sequence time, [1]: time distance to reward

            df_reward_row.append({
                "Session": session, 
                "Min Profile Length": min_neu, 
                "Stage": stage, 
                "Alpha": alpha,
                "Method": method, 
                "Reward ID": k, 
                "Event": "Failure",
                "Event Time (ms)": failure_time, 
                "Sequence Time (ms)": found_sequence[0], 
                "Time Distance (ms)": found_sequence[1]
            })

    df_reward_combined = pd.DataFrame(df_reward_row, columns=[
        "Session", "Min Profile Length", "Stage", "Alpha", "Method", 
        "Reward ID", "Event", "Event Time (ms)", 
        "Sequence Time (ms)", "Time Distance (ms)"
    ])

    return df_reward_combined