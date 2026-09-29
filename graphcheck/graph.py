import numpy as np
import networkx as nx
import pandas as pd
import random
from typing import List, Set

from scipy.ndimage import gaussian_filter1d

from .utils import df2binarray_csc


class Graph:
    def __init__(self):
        self.G = nx.Graph()
        self.communities = []
        self.neuron_attributes = {}
        self.attributes = {}

        self.sigma = None
        self.omega = None
        self.smallworldness = None
        self.density = None


    def calculate_distance(self):
        """
        Calculates the distance between nodes as the inverse of the weight
        """
        for _, _, data in self.G.edges(data=True):
            data["distance"] = 1. / data["weight"]


    def normalize_weights(self, epsilon: float = 0.01):
        """
        Normalizes the edge weights of a graph using shifted min-max normalization
        """
        edges = list(self.G.edges(data=True))
        weights = [data.get("weight", 1.0) for _, _, data in edges]

        min_weight = min(weights)
        max_weight = max(weights)

        # Avoid division by zero
        range_weight = max_weight - min_weight if max_weight > min_weight else 1

        if max_weight == min_weight:
            for _, _, data in edges:
                data["norm_weight"] = 1.0
            return

        range_weight = max_weight - min_weight

        for (_, _, data), w in zip(edges, weights):
            norm = epsilon + (1 - epsilon) * ((w - min_weight) / range_weight)
            data["norm_weight"] = max(epsilon, min(1.0, norm))


    def find_communities(self) -> List[Set[int]]:
        """
        Find communities via Louvain method
        """
        # DEPRECATED: communities via greedy modularity 
        # return sorted(nxcom.greedy_modularity_communities(self.G), key=len, reverse=True)        

        # ALSO DEPRECATED: using community_louvain module
        # partition = community_louvain.best_partition(self.G, weight='weight')
        # return get_louvain_sorted(partition)
        
        communities = nx.community.louvain_communities(
            self.G, 
            weight="weight", 
            resolution=1, 
            seed=12345
        )
        communities = sorted(communities, key=lambda c: min(c))
        return communities


    def extract_neuron_attributes(self, profile_dict: dict) -> dict:
        neuron_attributes = {}
                    
        for key in profile_dict:
            neu_idx = profile_dict[key]['neurons'].astype(int).tolist()
            for j in range(len(neu_idx)):
                if neu_idx[j] not in neuron_attributes:
                    neuron_attributes[neu_idx[j]] = {
                        # Previously: frequency information
                        #   "frequency": 1000 * profile_dict[i]['fr_ord'][j],
                        #   "freq_cat": "LFR" if 1000 * profile_dict[i]['fr_ord'][j] <= 1. else "HFR",

                        "electrode": profile_dict[key]['electrode'][j],
                        "persistent_assembly": 0
                    }
        return neuron_attributes


    def calculate_graph_attributes(self) -> dict:
        """
        Calculates attributes for the graph nodes
        """
        attributes = {}

        betweenness_centrality = nx.betweenness_centrality(self.G, weight="distance")
        degree_centrality = nx.degree_centrality(self.G)
        eigenvector_centrality = nx.eigenvector_centrality(
            self.G, weight="weight", max_iter=1000, tol=1e-6
        )
        clustering = nx.clustering(self.G)
        weighted_clustering = nx.clustering(self.G, weight="norm_weight")
        pagerank = nx.pagerank(self.G, weight="weight")
        degrees = dict(self.G.degree())
        hubs, authorities = nx.hits(self.G, normalized=True)

        for node in self.G.nodes():
            attributes[node] = {
                "betweenness_cent": betweenness_centrality[node],
                "degree_cent": degree_centrality[node],
                "eigenvector_cent": eigenvector_centrality[node],
                "clustering": clustering[node],
                "weighted_clustering": weighted_clustering[node],
                "pagerank": pagerank[node],
                "degree": degrees[node],
                "hub_value": hubs[node],
                "authority_value": authorities[node]
            }
        return attributes


    def set_attributes(self, attributes: dict):
        """
        Sets attributes for the graph nodes
        """
        nx.set_node_attributes(self.G, attributes)


    def set_community_attributes(self, communities: List[Set[int]]):
        """
        Sets community attributes for the graph nodes
        """
        attributes = {}
        for i, comm in enumerate(communities):
            for node in comm:
                attributes[node] = {
                    "community": i
                }
        nx.set_node_attributes(self.G, attributes)


    def set_persistent_assemblies(self, persistent_assemblies: List[dict]):
        """
        Sets persistent assemblies for the graph nodes
        """
        for i, pers_assemb in enumerate(persistent_assemblies):
            for node in pers_assemb['pers_assemb']:
                self.G.nodes[node]['persistent_assembly'] = i + 1


    def calculate_smallworldness(self, n_iter = 10) -> List[float]:
        """
        Calculates small-world-ness of the components of the graph

        Ref:
        Humphries, M.D., & Gurney, K. (2008). 
        Network ‘small-world-ness’: A quantitative method for determining canonical network equivalence. 
        PLoS One, 3(4), e0002051. doi:10.1371/journal.pone.0002051.
        """
        smallworldness = []
        for G in (self.G.subgraph(c).copy() for c in nx.connected_components(self.G)):
            if len(G) >= 5:
                # Compute the average shortest path length of the graph
                L = nx.average_shortest_path_length(G)
                
                # Compute the average clustering coefficient of the graph
                C = nx.average_clustering(G)
                
                # Create random graphs and average their metrics
                n = G.number_of_nodes()
                m = G.number_of_edges()
                L_randoms = []
                C_randoms = []
                for _ in range(n_iter):
                    random_graph = nx.gnm_random_graph(n, m)
                    if nx.is_connected(random_graph):
                        L_randoms.append(nx.average_shortest_path_length(random_graph))
                        C_randoms.append(nx.average_clustering(random_graph))
                
                # Compute the average metrics for random graphs
                L_random = np.mean(L_randoms)
                C_random = np.mean(C_randoms)
                
                # Calculate the small-worldness
                if L_random != 0 and C_random != 0:
                    smallworldness.append((C / C_random) / (L / L_random))
        return smallworldness
    

    def get_smallworldness(self):
        return self.smallworldness


    def write_terminal(self):
        """
        Writes graph information to terminal
        """
        print("Number of nodes: " + str(self.G.number_of_nodes()))
        print("Number of edges: " + str(self.G.number_of_edges()))
        print("Density: " + str(self.density))
        print("Number of communities: " + str(len(self.communities)))
        print("Community sizes: " + str([len(i) for i in self.communities]))
        print("Community members:")
        for comm in self.communities:
            print("\t"+str(list(comm)))
        print("Smallworldness: " + str(self.smallworldness))
        print("Sigma: " + str(self.sigma))
        print("Omega: " + str(self.omega))     


    def write_log(self, log):
        """
        Writes graph information to the log file
        """
        log.lwrite("Number of nodes: " + str(self.G.number_of_nodes()))
        log.lwrite("Number of edges: " + str(self.G.number_of_edges()))
        log.lwrite("Density: " + str(self.density))
        log.lwrite("Number of communities: " + str(len(self.communities)))
        log.lwrite("Community sizes: " + str([len(i) for i in self.communities]))
        log.lwrite("Community members:")
        for comm in self.communities:
            log.lwrite("\t"+str(list(comm)))
        log.lwrite("Smallworldness: " + str(self.smallworldness))
        log.lwrite("Sigma: " + str(self.sigma))
        log.lwrite("Omega: " + str(self.omega))


    def write_gephi(self, path_out):
        """
        Writes the graph to a Gephi-compatible file
        """
        nx.write_gexf(self.G, path_out + "_graph_export.gexf")


class ProfileGraph(Graph):
    def __init__(self, profile_dict: dict):
        super().__init__()
        self.G = self.create_profile_graph(profile_dict)

        self.calculate_distance()
        self.normalize_weights()

        self.neuron_attributes = self.extract_neuron_attributes(profile_dict)
        self.set_attributes(self.neuron_attributes)

        self.attributes = self.calculate_graph_attributes()
        self.set_attributes(self.attributes)

        self.communities = self.find_communities()
        self.set_community_attributes(self.communities)

        self.smallworldness = self.calculate_smallworldness()
        self.density = nx.density(self.G)


    def create_profile_graph(self, profile_dict):
        """
        Creates graph from profile dictionary
        """
        G = nx.Graph()
        for key in profile_dict:
            neu_idx = profile_dict[key]['neurons'].astype(int).tolist()
            G.add_nodes_from(neu_idx)
            
            for j in range(len(neu_idx)):
                for k in range(len(neu_idx)):
                    if j < k:
                        if G.has_edge(neu_idx[j], neu_idx[k]):
                            G[neu_idx[j]][neu_idx[k]]['weight'] += 1
                        else:
                            G.add_edge(neu_idx[j], neu_idx[k], weight=1)
        
        return G


class ShuffledProfileGraph(Graph):
    def __init__(self, profile_dict: dict, random_seed: int = 0):
        super().__init__()
        original_profiles = self.extract_profiles(profile_dict)
        self.shuffled_profiles = shuffle_profiles(original_profiles, random_seed)

        self.G = self.create_shuffled_profile_graph(self.shuffled_profiles)

        self.calculate_distance()
        self.normalize_weights()

        self.neuron_attributes = self.extract_neuron_attributes(profile_dict)
        self.set_attributes(self.neuron_attributes)
        
        self.attributes = self.calculate_graph_attributes()
        self.set_attributes(self.attributes)
        
        self.communities = self.find_communities()
        self.set_community_attributes(self.communities)
        
        self.smallworldness = self.calculate_smallworldness()
        self.density = nx.density(self.G)


    def extract_profiles(self, profile_dict):
        profiles = []
        for key in profile_dict:
            neu_idx = profile_dict[key]['neurons'].astype(int).tolist()
            profiles.append(neu_idx)
        return profiles

        
    def create_shuffled_profile_graph(self, shuffled_profiles: List[List[int]]):
        """
        Creates graph from profile dictionary
        """
        G = nx.Graph()
        for i, shuffled_profile in enumerate(shuffled_profiles):
            G.add_nodes_from(shuffled_profile)
            
            for j in range(len(shuffled_profile)):
                for k in range(len(shuffled_profile)):
                    if j < k:
                        if G.has_edge(shuffled_profile[j], shuffled_profile[k]):
                            G[shuffled_profile[j]][shuffled_profile[k]]['weight'] += 1
                        else:
                            G.add_edge(shuffled_profile[j], shuffled_profile[k], weight=1)

        return G
    

    def get_shuffled_profiles(self):
        return self.shuffled_profiles


def find_persistent_assemblies_graph(graph_a: Graph, graph_b: Graph, session: str, 
                                     min_neu: int, alpha: str, method: str):
    """
    Find groups of neurons that belong to the same community in both graphs.
    Could be an indication of a stable cell assembly.
        
    Definition of persistent: at least three overlapping neurons
    between the identified communities, tested two by two
        
    TO-DO: Confirm the existence of edges between them
    """
    persistent_assemblies = []
    
    for i, comm_early in enumerate(graph_a.communities):
        comm_i = set(comm_early)
        for j, comm_late in enumerate(graph_b.communities):
            comm_j = set(comm_late)
            if (len(comm_i & comm_j) >= 3):
                persistent_assemblies.append({
                    'session': session, 'min_neu': min_neu, 'alpha': alpha, 'method': method,
                    'size': len(comm_i & comm_j), 
                    'idx_comm_early': i, 'idx_comm_late': j, 
                    'comm_early': comm_i, 'comm_late': comm_j,
                    'pers_assemb': list(comm_i & comm_j)})
                
    if len(persistent_assemblies) == 0:
        persistent_assemblies.append({
            'session': session, 'min_neu': min_neu, 'alpha': alpha, 'method': method,
            'size': 0, 'idx_comm_early': None, 'idx_comm_late': None, 
            'comm_early': None, 'comm_late': None,
            'pers_assemb': []})
    
    return persistent_assemblies


def shuffle_profiles(original_profiles, random_seed = 0):
    rng = np.random.default_rng(random_seed)

    neurons = set([x for row in original_profiles for x in row])
    original_profiles_len = [len(row) for row in original_profiles]

    shuffled_profiles = []

    for prof_len in original_profiles_len:
        new_shuffle = rng.choice(list(neurons), size=prof_len, replace=False)
        shuffled_profiles.append(new_shuffle.astype(int).tolist())

    shuffled_profiles_len = [len(row) for row in shuffled_profiles]
       
    if shuffled_profiles_len != original_profiles_len:
        raise ValueError("Shuffled profiles have different lengths than original profiles")

    return shuffled_profiles


class CorrelationGraph(Graph):
    def __init__(self, corrmat, neuron_list, end_elect1, corr_thresh: float = 0.0):
        super().__init__()

        self.corrmat = corrmat

        self.G = self.create_correlation_graph(self.corrmat, neuron_list, corr_thresh)
        
        self.calculate_distance()
        self.normalize_weights()

        nodes = list(self.G.nodes())
        self.neuron_attributes = self.extract_neuron_attributes(nodes, end_elect1)
        self.set_attributes(self.neuron_attributes)

        self.attributes = self.calculate_graph_attributes()
        self.set_attributes(self.attributes)

        self.communities = self.find_communities()
        self.set_community_attributes(self.communities)

        self.smallworldness = self.calculate_smallworldness()
        self.density = nx.density(self.G)        


    def create_correlation_graph(self, corrmat: np.ndarray, neuron_list: List[int], 
                                 corr_thresh: float = 0.0):
        G = nx.Graph()
        G.add_nodes_from(neuron_list)

        # mask = np.eye(corrmat.shape[0], dtype=bool)
        # off_diagonal_corrmat = corrmat[~mask]
        # max_corr_offdiag = np.max(off_diagonal_corrmat)

        for i, neuron_i in enumerate(neuron_list):
            for j, neuron_j in enumerate(neuron_list):
                if i < j and corrmat[i, j] > corr_thresh:
                    weight = corrmat[i, j] # / max_corr_offdiag
                    G.add_edge(neuron_i, neuron_j, weight=weight)
        G.remove_nodes_from(list(nx.isolates(G)))
        return G
    
    
    def extract_neuron_attributes(self, nodes: List[int], end_elect1: int) -> dict:
        attributes = {}
        for neuron in nodes:
            electrode = 1 if neuron <= end_elect1 else 2
            attributes[neuron] = {
                "electrode": electrode,
                "persistent_assembly": 0 }
        return attributes
    

    # def calculate_graph_attributes(self) -> dict:
    #     """
    #     Calculates attributes for the graph nodes
    #     """
    #     attributes = {}

    #     betweenness_centrality = nx.betweenness_centrality(self.G, weight="distance")
    #     degree_centrality = nx.degree_centrality(self.G)
    #     # eigenvector_centrality = nx.eigenvector_centrality(self.G)
    #     clustering = nx.clustering(self.G)
    #     weighted_clustering = nx.clustering(self.G, weight="norm_weight")
    #     pagerank = nx.pagerank(self.G, weight="weight")
    #     degrees = dict(self.G.degree())
    #     # hubs, authorities = nx.hits(self.G, normalized=True)

    #     empty = {node: 0 for node in self.G.nodes()}

    #     for node in self.G.nodes():
    #         attributes[node] = {
    #             "betweenness_cent": betweenness_centrality[node],
    #             "degree_cent": degree_centrality[node],
    #             "eigenvector_cent": empty[node],
    #             "clustering": clustering[node],
    #             "weighted_clustering": weighted_clustering[node],
    #             "pagerank": pagerank[node],
    #             "degree": degrees[node],
    #             "hub_value": empty[node],
    #             "authority_value": empty[node]
    #         }
    #     return attributes


    def get_corrmat(self):
        return self.corrmat