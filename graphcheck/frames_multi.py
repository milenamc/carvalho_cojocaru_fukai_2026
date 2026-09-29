import numpy as np
import pandas as pd
import networkx as nx
from typing import List


from .datagraph import DataGraph


def create_stages_df(data_list: List[DataGraph]):
    dataframe = []

    for i, data in enumerate(data_list):
        row = data.create_df_stages_row()
        dataframe.append(row)
    
    df_stages = pd.DataFrame(dataframe, columns=[
        "Session", "Min Profile Length", "Stage", "Alpha", "Method", "# Neurons Below FR Max", "# Neurons in Graph",
        "# Profiles", "Avg Neurons per Profile", "# Shallow Profiles", "# Deep Profiles", 
        "Avg Occurrence Profile", "Avg Degree", "Max Degree", "Avg Weight", "Avg Weight Norm", 
        "Avg Clustering", "Avg Weighted Clustering", "# Connected Components", "Size Conn Components", 
        "Avg Shortest Path", "# Communities", "Size Communities", "Smallworldness", "Density"
    ])
        
    return df_stages


def create_comparison_df(data_list: List[dict], persistent_assemblies_list: List[List[dict]]):
    rows = []
    for i, data_session in enumerate(data_list):
        rows.append(create_comparison_df_row(data_session["Early"], data_session["Late"], persistent_assemblies_list[i]))
    df_comparison = pd.DataFrame(rows)
    return df_comparison


def create_comparison_df_row(data_early: DataGraph, data_late: DataGraph, persistent_assemblies: List[dict]):
    if data_early.data.session != data_late.data.session:
        raise ValueError("Data objects must belong to the same session")
    elif data_early.data.session != persistent_assemblies[0]['session']:
        raise ValueError("Data objects and persistent assemblies must belong to the same session")
    
    if data_early.data.min_neu != data_late.data.min_neu:
        raise ValueError("Data objects must have the same minimum profile length")
    elif data_early.data.min_neu != persistent_assemblies[0]['min_neu']:
        raise ValueError("Data objects and persistent assemblies must have the same minimum profile length")
    
    if data_early.method != data_late.method:
        raise ValueError("Data objects must have the same neuron type (original or shuffle)")
    elif data_early.method != persistent_assemblies[0]['method']:
        raise ValueError("Data objects and persistent assemblies must have the same neuron type (original or shuffle)")

    if data_early.data.stage == data_late.data.stage:
        raise ValueError("Data objects must belong to different stages")
    # TO-FIX: Currently assuming that the persistent assemblies were calculated for different stages (early and late, in that order)
    
    df_neurons_outer = pd.merge(data_early.df_neuron, data_late.df_neuron, on='ID', how="outer", suffixes=[" (Early)"," (Late)"]).sort_values(by=['ID'])
    df_neurons_inner = pd.merge(data_early.df_neuron, data_late.df_neuron, on='ID', how="inner", suffixes=[" (Early)"," (Late)"]).sort_values(by=['ID'])

    df_edges_outer = pd.merge(data_early.df_edge, data_late.df_edge, on="IDs", how="outer", suffixes=[" (Early)"," (Late)"]).sort_values(by=['IDs'])
    df_edges_inner = pd.merge(data_early.df_edge, data_late.df_edge, on="IDs", how="inner", suffixes=[" (Early)"," (Late)"]).sort_values(by=['IDs'])

    row = {
        "Session": data_early.data.session,
        "Min Profile Length": data_early.data.min_neu,
        "Alpha": data_early.data.alpha,
        "Method": data_early.method,
        "# Neurons in Graphs": len(df_neurons_outer),
        "# Persistent Neurons": len(df_neurons_inner),
        "% Persistent Neurons": len(df_neurons_inner) / len(df_neurons_outer) * 100,
        "# Neuron Pairs": len(df_edges_outer),
        "# Persistent Pairs": len(df_edges_inner),
        "% Persistent Pairs": len(df_edges_inner) / len(df_edges_outer) * 100,
        "# Communities Early": len(data_early.graph.communities),
        "Size Communities Early": [len(i) for i in data_early.graph.communities],
        "# Communities Late": len(data_late.graph.communities),
        "Size Communities Late": [len(i) for i in data_late.graph.communities],
        "# Persistent Assemblies": len(persistent_assemblies) if persistent_assemblies[0]['size'] > 0 else 0,
        "Size Persistent Assemblies": [pers_assemb['size'] for pers_assemb in persistent_assemblies],
        "ID Persistent Assemblies": [pers_assemb['pers_assemb'] for pers_assemb in persistent_assemblies]
    }
    
    return row


def export_stages_df(df_stages: pd.DataFrame, path_out: str, alpha: str):
    sessions = np.unique(df_stages["Session"])
    sessions_str = ""
    for session in sessions:
        sessions_str += str(session)+"_"
    df_stages.to_csv(f"{path_out}{sessions_str}{alpha}_df_stages.csv")


def export_concatenated_df(df_concatenated: pd.DataFrame, path_out: str, type: str, alpha: str):
    sessions = np.unique(df_concatenated["Session"])
    sessions_str = ""
    for session in sessions:
        sessions_str += str(session)+"_"
    df_concatenated.to_csv(f"{path_out}{sessions_str}{alpha}_df_{type}_concatenated.csv")


def export_comparison_df(df_comparison: pd.DataFrame, path_out: str, alpha: str):
    sessions = np.unique(df_comparison["Session"])
    sessions_str = ""
    for session in sessions:
        sessions_str += str(session)+"_"
    df_comparison.to_csv(f"{path_out}{sessions_str}{alpha}_df_comparison.csv")