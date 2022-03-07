from typing import List, Tuple, Callable

import networkx as nx
from tqdm import tqdm
import numpy as np
import pandas as pd


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


def feature_extractor(
    graph: nx.Graph,
    samples: List[Tuple[str, str]],
    features_method: List[Callable] = [graph_feature_extractor],
    node_information_path: str = "data/node_information.csv",
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
        feature_vectors.append(feature_method(graph, samples, node_information_path))
    feature_vector = np.concatenate(feature_vectors, axis=1)
    return feature_vector
