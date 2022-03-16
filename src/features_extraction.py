import hashlib
import os
import pickle
from typing import Callable, List, Tuple

import gensim.models.keyedvectors as word2vec
import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Manual Graph-based features


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


# Metadata based features


def metadata_features_extractor(
    graph: nx.Graph, samples: List[Tuple[str, str]], path: str
):
    feature_vector = []
    df = pd.read_csv(path, header=None)
    df.columns = [
        "id",
        "publication_year",
        "title",
        "authors",
        "journal_name",
        "abstract",
    ]

    df = df.groupby("id")

    for edge in tqdm(samples):
        source_node, target_node = edge[0], edge[1]
        source_row = df.get_group(int(source_node)).iloc[0]
        target_row = df.get_group(int(target_node)).iloc[0]

        # source_row = df[df["id"] == int(source_node)].iloc[0]
        # target_row = df[df["id"] == int(target_node)].iloc[0]

        if type(source_row["authors"]) is not str:
            source_authors = []
        else:
            source_authors = source_row["authors"].split(", ")
        if type(target_row["authors"]) is not str:
            target_authors = []
        else:
            target_authors = target_row["authors"].split(", ")

        number_of_common_authors = len(
            set(source_authors).intersection(set(target_authors))
        )

        source_year = int(source_row["publication_year"])
        target_year = int(target_row["publication_year"])

        year_difference = abs(source_year - target_year)

        source_title = source_row["title"].split()
        target_title = target_row["title"].split()

        title_difference = len(set(source_title).intersection(set(target_title)))

        feature_vector.append(
            np.array([number_of_common_authors, year_difference, title_difference])
        )
    return np.array(feature_vector)


# Textual based features


def transform_abstracts(samples, path):
    df = pd.read_csv(path, header=None)
    df.columns = [
        "id",
        "publication_year",
        "title",
        "authors",
        "journal_name",
        "abstract",
    ]

    df = df.groupby("id")

    sample_hash = hashlib.sha224(str(samples).encode()).hexdigest()

    if os.path.isfile(f"../cache/{sample_hash}"):
        with open(f"../cache/{sample_hash}", "rb") as file:
            cache_data = pickle.load(file)
        source_abstracts_encoded = cache_data["source_abstracts_encoded"]
        target_abstracts_encoded = cache_data["target_abstracts_encoded"]
    else:
        model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        source_abstracts = []
        target_abstracts = []
        for edge in tqdm(samples):
            source_node, target_node = edge[0], edge[1]
            source_row = df.get_group(int(source_node)).iloc[0]
            target_row = df.get_group(int(target_node)).iloc[0]
            source_abstracts.append(source_row["abstract"])
            target_abstracts.append(target_row["abstract"])

        source_abstracts_encoded = model.encode(
            source_abstracts, normalize_embeddings=True
        )
        target_abstracts_encoded = model.encode(
            source_abstracts, normalize_embeddings=True
        )

        cache_data = dict(
            source_abstracts_encoded=source_abstracts_encoded,
            target_abstracts_encoded=target_abstracts_encoded,
        )

        with open(f"../cache/{sample_hash}", "wb") as file:
            pickle.dump(cache_data, file)

    return source_abstracts_encoded, target_abstracts_encoded


def abstract_extractor_reduced(
    graph: nx.Graph, samples: List[Tuple[str, str]], path: str
):
    source_abstracts_encoded, target_abstracts_encoded = transform_abstracts(
        samples, path
    )
    return np.sum(source_abstracts_encoded * target_abstracts_encoded, axis=1)[
        ..., np.newaxis
    ]


def abstract_extractor_complete(
    graph: nx.Graph, samples: List[Tuple[str, str]], path: str
):
    source_abstracts_encoded, target_abstracts_encoded = transform_abstracts(
        samples, path
    )
    return np.concatenate([source_abstracts_encoded, target_abstracts_encoded], axis=1)


# Graph learned features


def generate_random_walk(graph, root, L):
    """
    :param graph: networkx graph
    :param root: the node where the random walk starts
    :param L: the length of the walk
    :return walk: list of the nodes visited by the random walk
    """
    walk = [root]
    while len(walk) < L:
        neighbors = list(graph.neighbors(walk[-1]))
        if len(neighbors) > 0:
            choice = np.random.choice(len(neighbors), 1)[0]
            walk.append(neighbors[choice])
        else:
            return walk
    return walk


def deep_walk(graph, N, L):
    """
    :param graph: networkx graph
    :param N: the number of walks for each node
    :param L: the walk length
    :return walks: the list of walks
    """
    walks = []

    nodes = list(graph.nodes())
    roots = [nodes[index] for index in np.random.choice(len(nodes), N)]
    for root in tqdm(roots):
        walks.append(generate_random_walk(graph, root, L))
    return walks


def run_graph_learning(graph: nx.Graph):
    nodes = list(graph.nodes)

    num_of_walks = 3000
    walk_length = 10000
    embedding_size = 32
    window_size = 6

    parameters = nodes + [num_of_walks, walk_length, embedding_size, window_size]

    file_hash = hashlib.sha224(str(parameters).encode()).hexdigest()

    model_filename = f"../cache/graph.embedding_{file_hash}"

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
):
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


def feature_extractor(
    graph: nx.Graph,
    samples: List[Tuple[str, str]],
    features_method: List[Callable] = [graph_feature_extractor],
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
