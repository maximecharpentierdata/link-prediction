from typing import Callable, List, Tuple

import networkx as nx
import numpy as np

from src.graph.features import graph_learned_features_extractor


def feature_extractor(
    graph: nx.Graph,
    samples: List[Tuple[str, str]],
    features_method: List[Callable],
    node_information_path: str = "data/node_information.csv",
    wv=None,
) -> np.ndarray:
    """Compute features of given edges

    Args:
        graph (nx.Graph): Graph
        samples (List[Tuple[str, str]]): List of edges
        features_method (List[Callable[...]], optional): List of features methods. Defaults to [graph_feature_extractor].
        node_information_path (str): Path to the node details

    Returns:
        np.ndarray: _description_
    """
    feature_vectors = []
    for feature_method in features_method:
        if feature_method == graph_learned_features_extractor:
            feature_vectors.append(
                feature_method(graph, samples, node_information_path, wv)
            )
        else:
            feature_vectors.append(
                feature_method(graph, samples, node_information_path)
            )
    feature_vector = np.concatenate(feature_vectors, axis=1)
    return feature_vector
