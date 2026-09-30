import os
import networkx as nx
import numpy as np
import pandas as pd
from typing import List


from .utils import Log, calculate_correlation_matrix
from .loaddata import LoadData
from .graph import Graph, ProfileGraph, ShuffledProfileGraph, CorrelationGraph, find_persistent_assemblies_graph
from .plot import plot_graph, plot_components, plot_degrees, plot_neuron_properties, plot_edge_properties
from .frames_single import create_df_neuron, create_df_edge, create_df_profile, create_df_reward, create_df_reward_combined, create_df_profile_shuffle


class DataGraph:
    def __init__(self, data: LoadData, method: str):
        """
        Class for creating and storing graphs and dataframes
        """
        
        self.data = data
        self.method = method

        self.graph = None

        self.df_neuron = None
        self.df_edge = None
        self.df_profile = None
        self.df_reward = None
        self.df_reward_combined = None
        self.profile_stats = None


    def create_graph(self):
        raise NotImplementedError("Method not implemented for superclass DataGraph. Use subclasses instead")


    def set_graph(self, graph: Graph):
        self.graph = graph


    def _get_sanitized_graph_copy(self):
        """
        Creates a GEXF-safe graph copy. Added to address recent updates in NetworkX
        that enforce stricter type checking for GEXF export.
        """
        def _cast(value):
            if isinstance(value, np.bool_):
                return bool(value)
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value

        G = self.graph.G.copy()

        # first convert NumPy types
        for _, attrs in G.nodes(data=True):
            for k, v in attrs.items():
                attrs[k] = _cast(v)

        for _, _, attrs in G.edges(data=True):
            for k, v in attrs.items():
                attrs[k] = _cast(v)

        # normalize node numeric attribute types
        node_attr_types = {}

        for _, attrs in G.nodes(data=True):
            for key, value in attrs.items():
                # bool is a subclass of int, so explicitly exclude it
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    node_attr_types.setdefault(key, set()).add(type(value))

        for key, types in node_attr_types.items():
            if int in types and float in types:
                # if both int and float types are present, convert all int values to float
                for _, attrs in G.nodes(data=True):
                    if (
                        key in attrs
                        and isinstance(attrs[key], int)
                        and not isinstance(attrs[key], bool)
                    ):
                        attrs[key] = float(attrs[key])

        # same for edge attributes
        edge_attr_types = {}

        for _, _, attrs in G.edges(data=True):
            for key, value in attrs.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    edge_attr_types.setdefault(key, set()).add(type(value))

        for key, types in edge_attr_types.items():
            if int in types and float in types:
                for _, _, attrs in G.edges(data=True):
                    if (
                        key in attrs
                        and isinstance(attrs[key], int)
                        and not isinstance(attrs[key], bool)
                    ):
                        attrs[key] = float(attrs[key])

        return G


    def export_graph(self, path_out: str):
        sanitized_G = self._get_sanitized_graph_copy()
        self.graph.write_gephi(
            f"{path_out}{self.data.session}_{self.data.stage}_{self.data.alpha}_{self.method}",
            G=sanitized_G
        )


    def create_df_neuron(self):
        self.df_neuron = create_df_neuron(self.graph, self.data.session, self.data.min_neu, self.data.stage, self.data.alpha, self.method)


    def export_df_neuron(self, path_out: str):
        if isinstance(self.df_neuron, pd.DataFrame):
            self.df_neuron.to_csv(f"{path_out}{self.data.session}_{self.data.stage}_{self.data.alpha}_{self.method}_df_neuron.csv",)
        else:
            print("Skipping export of neuron dataframe. Dataframe has not been created")
    

    def create_df_edge(self):
        self.df_edge = create_df_edge(self.graph, self.data.end_elect1, self.data.session, self.data.min_neu, self.data.stage, self.data.alpha, self.method)


    def export_df_edge(self, path_out: str):
        if isinstance(self.df_edge, pd.DataFrame):
            self.df_edge.to_csv(f"{path_out}{self.data.session}_{self.data.stage}_{self.data.alpha}_{self.method}_df_edge.csv")
        else:
            print("Skipping export of edge dataframe. Dataframe has not been created")


    def create_df_profile(self):
        # Only makes sense for Profile and Shuffle methods. Implemented in the relevant subclasses
        pass


    def export_df_profile(self, path_out: str):
        if isinstance(self.df_profile, pd.DataFrame):
            self.df_profile.to_csv(f"{path_out}{self.data.session}_{self.data.stage}_{self.data.alpha}_{self.method}_df_profile.csv")
        else:
            print("Skipping export of profile dataframe. Dataframe has not been created")   


    def create_df_reward(self):
        # Only makes sense for Profile method. Implemented in the relevant subclass
        pass


    def create_df_reward_combined(self):
        # Only makes sense for Profile method. Implemented in the relevant subclass
        pass


    def export_df_reward(self, path_out: str):
        if isinstance(self.df_reward, pd.DataFrame):
            self.df_reward.to_csv(
                f"{path_out}{self.data.session}_{self.data.stage}_{self.data.alpha}_{self.method}_df_reward.csv",
                float_format="%.4f"
            )
        else:
            print("Skipping export of reward dataframe. Dataframe has not been created")


    def export_df_reward_combined(self, path_out: str):
        if isinstance(self.df_reward_combined, pd.DataFrame):
            self.df_reward_combined.to_csv(
                f"{path_out}{self.data.session}_{self.data.stage}_{self.data.alpha}_{self.method}_df_reward_combined.csv",
                float_format="%.4f"
            )
        else:
            print("Skipping export of combined reward dataframe. Dataframe has not been created")


    def create_all(self):
        self.create_graph()
        self.create_df_neuron()
        self.create_df_edge()
        self.create_df_profile()
        self.create_df_reward()
        self.create_df_reward_combined()


    def export_all(self, path_out: str):
        self.export_graph(path_out)
        self.export_df_neuron(path_out)
        self.export_df_edge(path_out)
        self.export_df_profile(path_out)
        self.export_df_reward(path_out)
        self.export_df_reward_combined(path_out)


    def plot_graphs(self, path_out: str):
        fig_title = f"Session {self.data.session} ({self.data.stage}) | {self.method}"
        path_out_local = f"{path_out}{self.data.session}_{self.data.stage}_{self.data.alpha}_{self.method}"
        plot_graph(self.graph.G, path_out_local, fig_title)
        plot_components(self.graph.G, path_out_local, fig_title)
        plot_degrees(self.graph.G, path_out_local, fig_title)


    def plot_dfs(self, path_out: str):
        fig_title = f"Session {self.data.session} ({self.data.stage}) | {self.method}"
        path_out_local = f"{path_out}{self.data.session}_{self.data.stage}_{self.data.alpha}_{self.method}"
        if isinstance(self.df_neuron, pd.DataFrame):
            plot_neuron_properties(self.df_neuron, path_out_local, fig_title)
        if isinstance(self.df_edge, pd.DataFrame):
            plot_edge_properties(self.df_edge, path_out_local, fig_title)
        # if isinstance(self.df_reward, pd.DataFrame):
        #     plot_reward_properties(self.df_reward, path_out_local, fig_title)
        #     plot_event_alignment(self.df_reward, path_out_local, fig_title)
        #     if self.data.lever_data is not None:
        #         plot_lever_dynamics(self.df_reward, self.data.lever_data, path_out_local, fig_title)


    def plot_all(self, path_out: str):
        self.plot_graphs(path_out)
        self.plot_dfs(path_out)


    def write_graph_terminal(self):
        self.graph.write_terminal()


    def write_graph_log(self, log: Log):
        self.graph.write_log(log)


    def write_report_terminal(self):
        """
        Write report to terminal
        """
        print(f"Session {self.data.session} ({self.data.stage}) | Min. Neurons: {self.data.min_neu}")
        print(f"Method: {self.method}")
        print("Graph created" if self.graph else "No graph created")
        print("Neuron dataframe created" if isinstance(self.df_neuron, pd.DataFrame) else "No neuron dataframe created")
        print("Edge dataframe created" if isinstance(self.df_edge, pd.DataFrame) else "No edge dataframe created")
        print("Profile dataframe created" if isinstance(self.df_profile, pd.DataFrame) else "No profile dataframe created")
        print("Reward dataframe created" if isinstance(self.df_reward, pd.DataFrame) else "No reward dataframe created")
        print("Combined reward dataframe created" if isinstance(self.df_reward_combined, pd.DataFrame) else "No reward dataframe created")


    def write_report_log(self, log: Log):
        """
        Write report to log file
        """
        log.lwrite(f"Session {self.data.session} ({self.data.stage}) | Min. Neurons: {self.data.min_neu}")
        log.lwrite(f"Method: {self.method}")
        log.lwrite("Graph created" if self.graph else "No graph created")
        log.lwrite("Neuron dataframe created" if isinstance(self.df_neuron, pd.DataFrame) else "No neuron dataframe created")
        log.lwrite("Edge dataframe created" if isinstance(self.df_edge, pd.DataFrame) else "No edge dataframe created")
        log.lwrite("Profile dataframe created" if isinstance(self.df_profile, pd.DataFrame) else "No profile dataframe created")
        log.lwrite("Reward dataframe created" if isinstance(self.df_reward, pd.DataFrame) else "No reward dataframe created")
        log.lwrite("Combined reward dataframe created" if isinstance(self.df_reward_combined, pd.DataFrame) else "No reward dataframe created")



    def set_persistent_assemblies(self, persistent_assemblies: List[dict]):
        self.graph.set_persistent_assemblies(persistent_assemblies)


    def create_df_stages_row(self):
        G = self.graph.G.copy()
        degrees = [val for (node, val) in G.degree()]
        weights = list(nx.get_edge_attributes(G,'weight').values())

        neurons_below_fr_max = self.data.get_num_neurons_below_fr_max()

        avg_neurons_per_profile = sum(self.profile_stats['n_neurons']) / self.profile_stats['n_profiles']
        avg_occurrence_profile = sum(self.profile_stats['seq_count']) / self.profile_stats['n_profiles']
        neurons_in_graph = len(degrees)

        avg_degree = sum(degrees) / len(degrees)
        max_degree = max(degrees)
        avg_weight = sum(weights) / len(weights)
        avg_weight_norm = sum(weights) / max(weights) / len(weights)
        avg_clustering = nx.average_clustering(G)
        avg_weighted_clustering = nx.average_clustering(G, weight='norm_weight')

        size_conn_components = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        n_conn_components = len(size_conn_components)
        
        avg_shortest_path = []
        for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
            avg_shortest_path.append(nx.average_shortest_path_length(C, weight='weight'))

        row = {
            "Session": self.data.session,
            "Min Profile Length": self.data.min_neu,
            "Stage": self.data.stage,
            "Alpha": self.data.alpha,
            "Method": self.method,
            "# Neurons Below FR Max": neurons_below_fr_max,
            "# Neurons in Graph": neurons_in_graph,            
            "# Profiles": self.profile_stats['n_profiles'],
            "Avg Neurons per Profile": avg_neurons_per_profile,
            "# Shallow Profiles": self.profile_stats['sup_prof'],
            "# Deep Profiles": self.profile_stats['deep_prof'],
            "Avg Occurrence Profile": avg_occurrence_profile,
            "Avg Degree": avg_degree,
            "Max Degree": max_degree,
            "Avg Weight": avg_weight,
            "Avg Weight Norm": avg_weight_norm,
            "Avg Clustering": avg_clustering,
            "Avg Weighted Clustering": avg_weighted_clustering,
            "# Connected Components": n_conn_components,
            "Size Conn Components": size_conn_components,
            "Avg Shortest Path": avg_shortest_path,
            "# Communities": len(self.graph.communities),
            "Size Communities": [len(i) for i in self.graph.communities],
            "Smallworldness": self.graph.smallworldness,
            "Density": self.graph.density
        }

        return row


class DataGraphProfile(DataGraph):
    def __init__(self, data: LoadData, method: str = "Profile"):
        super().__init__(data, method)


    def create_graph(self):
        profile_dict = self.data.get_profile_dict()
        self.graph = ProfileGraph(profile_dict)


    def create_df_profile(self):
        profile_dict = self.data.get_profile_dict()
        self.df_profile, self.profile_stats = create_df_profile(profile_dict, self.data.session, self.data.min_neu, self.data.stage, self.data.alpha, self.method)


    def create_df_reward(self, interval_sec: float = 1.0): # previously 1.5
        profile_dict = self.data.get_profile_dict()
        event = self.data.get_event_data()
        self.df_reward = create_df_reward(
            profile_dict, event, self.data.session, self.data.min_neu, 
            self.data.stage, self.data.alpha, self.method, interval_sec)


    def create_df_reward_combined(self, interval_sec: float = 1.0): # previously 1.5
        profile_dict = self.data.get_profile_dict()
        event = self.data.get_event_data()
        self.df_reward_combined = create_df_reward_combined(
            profile_dict, event, self.data.session, self.data.min_neu, 
            self.data.stage, self.data.alpha, self.method, interval_sec
        )


class DataGraphShuffle(DataGraph):
    def __init__(self, data: LoadData, method: str = "Shuffle", random_seed: int = 0):
        super().__init__(data, method + f"_RS_{random_seed}")
        self.shuffled_profiles = []
        self.random_seed = random_seed


    def create_graph(self):
        profile_dict = self.data.get_profile_dict()
        self.graph = ShuffledProfileGraph(profile_dict, self.random_seed)
        self.shuffled_profiles = self.graph.get_shuffled_profiles()


    def create_df_profile(self):
        self.df_profile, self.profile_stats = create_df_profile_shuffle(
            self.shuffled_profiles, self.data.end_elect1, self.data.session, 
            self.data.min_neu, self.data.stage, self.data.alpha, self.method
        )


class DataGraphCorrelation(DataGraph):
    def __init__(self, data: LoadData, method: str = "Correlation", 
                 corr_bin_width: int = 1, corr_thresh: float = 0.0):
        super().__init__(data, method + f"_BW_{corr_bin_width}_TH_{corr_thresh}")

        self.corr_bin_width = corr_bin_width
        self.corr_thresh = corr_thresh

        self.corrmat = None  
        self.neuron_list = None


    def create_graph(self):
        df_spikes = self.data.get_df_spikes()

        self.corrmat, self.neuron_list = calculate_correlation_matrix(
            df_spikes, self.data.start_time_ms,
            self.data.duration_ms, self.corr_bin_width)

        self.graph = CorrelationGraph(self.corrmat, self.neuron_list, 
                                      self.data.end_elect1, self.corr_thresh)


    def create_df_stages_row(self):
        G = self.graph.G

        degrees = [val for (node, val) in G.degree()]
        weights = list(nx.get_edge_attributes(G,'weight').values())
        neurons_in_graph = len(degrees)

        neurons_below_fr_max = self.data.get_num_neurons_below_fr_max()

        avg_degree = sum(degrees) / len(degrees)
        max_degree = max(degrees)
        avg_weight = sum(weights) / len(weights)
        avg_weight_norm = sum(weights) / max(weights) / len(weights)
        avg_clustering = nx.average_clustering(G)
        avg_weighted_clustering = nx.average_clustering(G, weight='norm_weight')

        size_conn_components = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        n_conn_components = len(size_conn_components)
        
        avg_shortest_path = []
        for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
            avg_shortest_path.append(nx.average_shortest_path_length(C, weight='weight'))

        row = {
            "Session": self.data.session,
            "Min Profile Length": self.data.min_neu,
            "Stage": self.data.stage,
            "Alpha": self.data.alpha,
            "Method": self.method,
            "# Neurons Below FR Max": neurons_below_fr_max,
            "# Neurons in Graph": neurons_in_graph,            
            "# Profiles": None,
            "Avg Neurons per Profile": None,
            "# Shallow Profiles": None,
            "# Deep Profiles": None,
            "Avg Occurrence Profile": None,
            "Avg Degree": avg_degree,
            "Max Degree": max_degree,
            "Avg Weight": avg_weight,
            "Avg Weight Norm": avg_weight_norm,
            "Avg Clustering": avg_clustering,
            "Avg Weighted Clustering": avg_weighted_clustering,
            "# Connected Components": n_conn_components,
            "Size Conn Components": size_conn_components,
            "Avg Shortest Path": avg_shortest_path,
            "# Communities": len(self.graph.communities),
            "Size Communities": [len(i) for i in self.graph.communities],
            "Smallworldness": self.graph.smallworldness,
            "Density": self.graph.density
        }

        return row
    
    def get_corrmat(self):
        return self.corrmat.copy()


def find_persistent_assemblies(data_early: DataGraph, data_late: DataGraph):
    """
    Find persistent assemblies of two data objects belonging to the same session
    """
    if data_early.data.session != data_late.data.session:
        raise ValueError("Data objects must belong to the same session")
    if data_early.data.stage == data_late.data.stage:
        raise ValueError("Data objects must belong to different stages")
    if data_early.method != data_late.method:
        raise ValueError("Data objects must be of the same method")
    
    persistent_assemblies = find_persistent_assemblies_graph(
        data_early.graph, data_late.graph, 
        data_early.data.session, data_early.data.min_neu, 
        data_early.data.alpha, data_early.method
    )
    data_early.set_persistent_assemblies(persistent_assemblies)
    data_late.set_persistent_assemblies(persistent_assemblies)
    return persistent_assemblies