import argparse
import csv
from typing import Callable, List

import gensim
import networkx as nx
import numpy as np

from src.classification.classical import (
    generate_submission_data_classical_model,
    make_classication_lr,
    make_classication_rf,
    make_classication_svm,
    make_classication_xgboost,
)
from src.classification.nn import (
    generate_submission_data_for_neural_networks,
    make_classification_neural_network,
)
from src.features_extraction import feature_extractor
from src.graph.features import (
    graph_feature_extractor,
    graph_learned_features_extractor,
    run_graph_learning,
)
from src.graph.preprocessing import create_graph, load_graph_data
from src.metadata.features import metadata_features_extractor, tfidf_abstract_extractor


def fetch_submission_features(
    graph: nx.Graph, features_method: List[Callable], wv: gensim.models.KeyedVectors
) -> np.ndarray:
    with open("data/testing_set.txt", "r") as f:
        reader = csv.reader(f)
        samples = list(reader)
    samples = [sample[0].split() for sample in samples]

    features = feature_extractor(
        graph,
        samples,
        features_method,
        node_information_path="data/node_information.csv",
        wv=wv,
    )
    return features


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the option for using metadata features (authors, years, title)",
    )
    parser.add_argument(
        "--graph_features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the option for using basic graph features",
    )
    parser.add_argument(
        "--graph_learned_features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the option for using advanced learned graph features with Word2Vec",
    )
    parser.add_argument(
        "--tfidf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the option for using advanced textual information from the abstract with TFIDF",
    )
    parser.add_argument(
        "--experiment", type=str, default="1.0_0.8", help="Choose the experiment"
    )
    parser.add_argument(
        "--classification",
        type=str,
        default="nn",
        choices=["nn", "lr", "rf", "xgboost", "svm"],
        help="Choose a classifier",
    )
    parser.add_argument(
        "--submission_name", type=str, help="Name for the submission file"
    )
    return parser.parse_args()


def main():
    with open("data/node_information.csv", "r") as f:
        reader = csv.reader(f)
        node_info = list(reader)

    nodes = [element[0] for element in node_info]

    with open("data/training_set.txt", "r") as f:
        reader = csv.reader(
            f,
            delimiter=" ",
        )
        train_dataset = list(reader)

    arguments = parse_arguments()

    full_graph = create_graph(nodes, train_dataset)

    if arguments.graph_learned_features:
        print("Running Word2Vec model...")
        wv = run_graph_learning(full_graph)
    else:
        wv = None

    print("Loading graph data...")
    graph_data_path = f"experiments/{arguments.experiment}"
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
    if arguments.tfidf:
        features_methods.append(tfidf_abstract_extractor)

    if features_methods == []:
        raise ValueError("No feature methods specified")

    print("Computing training features...")
    train_features = feature_extractor(
        sub_graph,
        train_samples,
        features_methods,
        node_information_path="data/node_information.csv",
        wv=wv,
    )

    print("Computing testing features...")
    test_features = feature_extractor(
        sub_graph,
        test_samples,
        features_methods,
        node_information_path="data/node_information.csv",
        wv=wv,
    )

    if arguments.classification == "nn":
        model, th, (mean, std) = make_classification_neural_network(
            train_features, train_labels, test_features, test_labels
        )

        submission_features = fetch_submission_features(
            full_graph, features_methods, wv
        )

        generate_submission_data_for_neural_networks(
            model, th, submission_features, mean, std, arguments.submission_name
        )

    classical = False
    if arguments.classification == "lr":
        classical = True
        classification = make_classication_lr
    if arguments.classification == "svm":
        classical = True
        classification = make_classication_svm
    if arguments.classification == "rf":
        classical = True
        classification = make_classication_rf
    if arguments.classification == "xgboost":
        classical = True
        classification = make_classication_xgboost

    if classical:
        model, th, scaler = classification(
            train_features, train_labels, test_features, test_labels
        )

        submission_features = fetch_submission_features(
            full_graph, features_methods, wv
        )

        generate_submission_data_classical_model(
            model, th, submission_features, scaler, arguments.submission_name
        )
    else:
        raise RuntimeError("No classication technique specified")


if __name__ == "__main__":
    main()
