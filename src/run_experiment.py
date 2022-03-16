import argparse
import csv
from typing import Callable, List

import gensim
import networkx as nx
import numpy as np

from src.classification_utils import (
    generate_submission_data_for_neural_networks,
    make_classification_neural_network)
from src.features_extraction import (abstract_extractor_complete,
                                     abstract_extractor_reduced,
                                     feature_extractor,
                                     graph_feature_extractor,
                                     graph_learned_features_extractor,
                                     metadata_features_extractor,
                                     run_graph_learning)
from src.graph_preprocessing import create_graph, load_graph_data


def fetch_submission_features(
    graph: nx.Graph, features_method: List[Callable], wv: gensim.models.KeyedVectors
) -> np.ndarray:
    with open("../data/testing_set.txt", "r") as f:
        reader = csv.reader(f)
        samples = list(reader)
    samples = [sample[0].split() for sample in samples]

    features = feature_extractor(
        graph,
        samples,
        features_method,
        node_information_path="../data/node_information.csv",
        wv=wv,
    )
    return features


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_features", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--graph_features", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--graph_learned_features", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--abstract_features", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--full_abstract_features", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--experiment", type=str, default="1.0_0.8")
    parser.add_argument("--classification", type=str, default="nn")
    parser.add_argument("--submission_name", type=str)
    return parser.parse_args()


def main():
    with open("../data/node_information.csv", "r") as f:
        reader = csv.reader(f)
        node_info = list(reader)

    nodes = [element[0] for element in node_info]

    with open("../data/training_set.txt", "r") as f:
        reader = csv.reader(
            f,
            delimiter=" ",
        )
        train_dataset = list(reader)

    arguments = parse_arguments()

    if arguments.graph_learned_features:
        print("Generating full graph and running Word2Vec model...")
        full_graph = create_graph(nodes, train_dataset)
        wv = run_graph_learning(full_graph)

    graph_data_path = f"../experiments/{arguments.experiment}"
    (
        sub_graph,
        residual_g,
        train_samples,
        train_labels,
        test_samples,
        test_labels,
    ) = load_graph_data(graph_data_path)

    features_methods = []

    if arguments.metadata_features:
        features_methods.append(metadata_features_extractor)
    if arguments.graph_features:
        features_methods.append(graph_feature_extractor)
    if arguments.graph_learned_features:
        features_methods.append(graph_learned_features_extractor)
    if arguments.abstract_features:
        features_methods.append(abstract_extractor_reduced)
    if arguments.full_abstract_features:
        features_methods.append(abstract_extractor_complete)

    if features_methods == []:
        raise ValueError("No feature methods specified")

    print("Computing training features...")
    train_features = feature_extractor(
        sub_graph,
        train_samples,
        features_methods,
        node_information_path="../data/node_information.csv",
        wv=wv,
    )

    print("Computing testing features...")
    test_features = feature_extractor(
        sub_graph,
        test_samples,
        features_methods,
        node_information_path="../data/node_information.csv",
        wv=wv,
    )

    if arguments.classification == "nn":
        model, th, (mean, std) = make_classification_neural_network(
            train_features, train_labels, test_features, test_labels
        )
        generate_submission_data_for_neural_networks(
            model, th, mean, std, arguments.submission_name
        )


if __name__ == "__main__":
    main()
