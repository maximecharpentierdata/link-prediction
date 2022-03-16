from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


# Metadata based features
def metadata_features_extractor(
    graph: nx.Graph, samples: List[Tuple[str, str]], path: str
) -> np.ndarray:
    """Generates metadata based features

    Args:
        graph (nx.Graph): Graph
        samples (List[Tuple[str, str]]): Edge samples
        path (str): Path for node information

    Returns:
        np.ndarray: Features
    """
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
def _compute_tfidf(
    graph: nx.Graph, samples: List[Tuple[str, str]], path: str, num_features: int = 2500
):
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
    source_abstracts = []
    target_abstracts = []
    for edge in tqdm(samples):
        source_node, target_node = edge[0], edge[1]
        source_row = df.get_group(int(source_node)).iloc[0]
        target_row = df.get_group(int(target_node)).iloc[0]
        source_abstracts.append(source_row["abstract"])
        target_abstracts.append(target_row["abstract"])

    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", max_features=num_features
    )
    all_abstracts = source_abstracts + target_abstracts
    all_features = vectorizer.fit_transform(all_abstracts)

    source_features = all_features[: len(source_abstracts), :].todense()
    target_features = all_features[len(source_abstracts) :, :].todense()

    multiplied = np.multiply(source_features, target_features)

    summed = np.sum(multiplied, axis=1)

    return source_features, target_features


def tfidf_abstract_extractor(
    graph: nx.Graph, samples: List[Tuple[str, str]], path: str
) -> np.ndarray:
    """Computes TFIDF based features

    Args:
        graph (nx.Graph): Graph
        samples (List[Tuple[str, str]]): Edge samples
        path (str): Path for node information

    Returns:
        np.ndarray: Features
    """
    source_features, target_features = _compute_tfidf(graph, samples, path)
    multiplied = np.multiply(source_features, target_features)
    summed = np.sum(multiplied, axis=1)
    return summed
