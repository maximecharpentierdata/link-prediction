from typing import List, Tuple, Callable

import networkx as nx
from tqdm import tqdm
import numpy as np


def graph_feature_extractor(graph: nx.Graph, samples: List[Tuple[str, str]]) -> np.ndarray:
    """Compute graph based features of given edges

    Args:
        graph (nx.Graph): Graph
        samples (List[Tuple[str, str]]): List of edges

    Returns:
        np.ndarray: Features of the edges
    """
    feature_vector = [] 
    
    deg_centrality = nx.degree_centrality(graph)
    
    # --- Extract manually diverse features relative to each edge contained in samples --- 

    for edge in tqdm(samples):
        source_node, target_node = edge[0], edge[1]

        # Degree Centrality
        source_degree_centrality = deg_centrality[source_node]
        target_degree_centrality = deg_centrality[target_node]
        
        # Betweeness centrality  
        # diff_bt = betweeness_centrality[target_node] - betweeness_centrality[source_node]

        # Preferential Attachement 
        pref_attach = list(nx.preferential_attachment(graph, [(source_node, target_node)]))[0][2]

        # AdamicAdar
        aai = list(nx.adamic_adar_index(graph, [edge]))[0][-1]

        # Jaccard
        jacard_coeff = list(nx.jaccard_coefficient(graph, [edge]))[0][-1]
        
        # Create edge feature vector with all metric computed above
        feature_vector.append(np.array([source_degree_centrality, target_degree_centrality, 
                                        pref_attach, aai, jacard_coeff])) 
        
    return np.array(feature_vector)


def feature_extractor(graph: nx.Graph, samples: List[Tuple[str, str]], features_method: List[Callable]=[graph_feature_extractor]) -> np.ndarray:
    """Compute features of given edges

    Args:
        graph (nx.Graph): Graph
        samples (List[Tuple[str, str]]): List of edges
        features_method (List[Callable[...]], optional): List of features methods. Defaults to [graph_feature_extractor].

    Returns:
        np.ndarray: _description_
    """
    feature_vectors = []
    for feature in features_method:
        feature_vectors.append(feature(graph, samples))
    feature_vector = np.concatenate(feature_vectors, axis=1)
    return feature_vector