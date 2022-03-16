import hashlib
import os
from typing import List, Tuple

import gensim.models.keyedvectors as word2vec
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm


def graph_feature_extractor(
    graph: nx.Graph, samples: List[Tuple[str, str]], path: str = None
) -> np.ndarray:
    """Compute graph based features of given edges

    Args:
        graph (nx.Graph): Graph
        samples (List[Tuple[str, str]]): List of edges
        path (str): Unused

    Returns:
        np.ndarray: Features of the edges
    """
    feature_vector = []

    deg_centrality = nx.degree_centrality(graph)
    # betweeness_centrality = nx.betweenness_centrality(graph)

    # --- Extract manually diverse features relative to each edge contained in samples ---

    for edge in tqdm(samples):
        source_node, target_node = edge[0], edge[1]

        # Degree Centrality
        source_degree_centrality = deg_centrality[source_node]
        target_degree_centrality = deg_centrality[target_node]

        # Betweeness centrality
        # diff_bt = betweeness_centrality[target_node] - betweeness_centrality[source_node]

        # Preferential Attachement
        pref_attach = list(
            nx.preferential_attachment(graph, [(source_node, target_node)])
        )[0][2]

        # AdamicAdar
        aai = list(nx.adamic_adar_index(graph, [edge]))[0][-1]

        # Jaccard
        jacard_coeff = list(nx.jaccard_coefficient(graph, [edge]))[0][-1]

        # Create edge feature vector with all metric computed above
        feature_vector.append(
            np.array(
                [
                    source_degree_centrality,
                    target_degree_centrality,
                    pref_attach,
                    aai,
                    jacard_coeff,
                ]
            )
        )

    return np.array(feature_vector)


# Graph learned features


def generate_random_walk(graph: nx.Graph, root: str, length: int) -> List[str]:
    """Generates a random walk

    Args:
        graph (nx.Graph): Graph
        root (str): Root node of the walk
        length (int): Length of the walk

    Returns:
        List[str]: Walk
    """
    walk = [root]
    while len(walk) < length:
        neighbors = list(graph.neighbors(walk[-1]))
        if len(neighbors) > 0:
            choice = np.random.choice(len(neighbors), 1)[0]
            walk.append(neighbors[choice])
        else:
            return walk
    return walk


def deep_walk(graph: nx.Graph, walks_number: int, walk_length: int) -> List[List[str]]:
    """Generates multiple random walks

    Args:
        graph (nx.Graph): Graph
        walks_number (int): Number of random walks
        walk_length (int): Length of each random walk

    Returns:
        List[List[str]]: Random walks
    """
    walks = []

    nodes = list(graph.nodes())
    roots = [nodes[index] for index in np.random.choice(len(nodes), walks_number)]
    for root in tqdm(roots):
        walks.append(generate_random_walk(graph, root, walk_length))
    return walks


def run_graph_learning(graph: nx.Graph) -> word2vec.KeyedVectors:
    """Trains a Word2Vec model on a graph

    Args:
        graph (nx.Graph): Graph

    Returns:
        word2vec.KeyedVectors: Trained Word2Vec
    """
    nodes = list(graph.nodes)

    num_of_walks = 3000
    walk_length = 10000
    embedding_size = 32
    window_size = 6

    parameters = nodes + [num_of_walks, walk_length, embedding_size, window_size]

    file_hash = hashlib.sha224(str(parameters).encode()).hexdigest()

    model_filename = f"cache/graph.embedding_{file_hash}"

    if not os.path.isfile(model_filename):
        # Perform random walks - call function
        walks = deep_walk(graph, num_of_walks, walk_length)
        # Learn representations of nodes - use Word2Vec
        model = Word2Vec(
            walks, window=window_size, vector_size=embedding_size, hs=1, sg=1
        )
        # Save the embedding vectors
        model.wv.save_word2vec_format(model_filename)
    else:
        model = Word2Vec()
        model.wv = word2vec.KeyedVectors.load_word2vec_format(model_filename)
    return model.wv


def graph_learned_features_extractor(
    graph: nx.Graph, samples: List[Tuple[str, str]], path: str, wv
) -> np.ndarray:
    """Computes the learned graph features

    Args:
        graph (nx.Graph): Graph
        samples (List[Tuple[str, str]]): Edge samples
        path (str): Path for the node information file
        wv (_type_): Word2Vec model

    Returns:
        np.ndarray: Features vector
    """
    feature_func = lambda x, y: abs(x - y)

    feature_vector = []

    for edge in tqdm(samples):
        try:
            embed_1 = wv[edge[0]]
        except:
            embed_1 = np.zeros(32)
        try:
            embed_2 = wv[edge[1]]
        except:
            embed_2 = np.zeros(32)

        feature_vector.append(feature_func(embed_1, embed_2))

    return np.array(feature_vector)
