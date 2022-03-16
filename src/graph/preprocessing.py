import argparse
import csv
import json
import os
import random
from typing import List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm


def create_graph(
    node_list: List[int], edges_dataset: List[Tuple[str, str, str]]
) -> nx.Graph:
    """Creates graph from list of nodes and list of edges

    Args:
        node_list (List[int]): List of nodes
        edges_dataset (List[Tuple[str, str, str]]): List of edges

    Returns:
        nx.Graph: Created graph
    """
    graph = nx.Graph()
    for node in node_list:
        graph.add_node(node)
    for source, target, label in edges_dataset:
        if label == "1":
            graph.add_edge(source, target)
    return graph


def make_sub_graphs(graph: nx.Graph) -> List[nx.Graph]:
    """Creates all connected components graphs

    Args:
        graph (nx.Graph): Input graph

    Returns:
        List[nx.Graph]: List of connected subgraphs
    """
    sub_graphs = []
    for connected_component in nx.connected_components(graph):
        sub_graphs.append(nx.subgraph(graph, connected_component))
    return sub_graphs


def sample_non_edges(graph: nx.Graph, n_sample: int) -> List[Tuple[str, str]]:
    """Samples non existing edges from a graph

    Args:
        graph (nx.Graph): Input graph
        n_sample (int): Number of negative edges

    Returns:
        List[Tuple[str, str]]: List of non existing edges
    """
    samples = []
    nodes = list(graph.nodes())
    edges = list(graph.edges())
    while len(samples) < n_sample:
        source, target = random.sample(nodes, 2)
        if (source, target) not in edges:
            samples.append((source, target))

    return samples


def take_sub_graph(graph: nx.Graph, ratio: float) -> nx.Graph:
    """Creates a random connected subgraph

    Args:
        graph (nx.Graph): Input graph
        ratio (float): Size ratio of the subgraph

    Returns:
        nx.Graph: Connected subgraph
    """
    nodes = list(graph.nodes())
    np.random.shuffle(nodes)
    sub_graph = graph.subgraph(nodes[: int(len(nodes) * ratio)])
    sub_graphs = make_sub_graphs(sub_graph)
    sub_graphs = sorted(
        sub_graphs, key=lambda graph: graph.number_of_nodes(), reverse=True
    )
    return sub_graphs[0]


def generate_samples(graph, train_set_ratio):
    """
    Graph pre-processing step required to perform supervised link prediction
    Create training and test sets
    """

    # --- Step 0: The graph must be connected ---
    if nx.is_connected(graph) is not True:
        raise ValueError("The graph contains more than one connected component!")

    # --- Step 1: Generate positive edge samples for testing set ---
    residual_g = graph.copy()
    test_pos_samples = []

    # Shuffle the list of edges
    edges = list(residual_g.edges())
    np.random.shuffle(edges)

    # Define number of positive samples desired
    test_set_size = int((1.0 - train_set_ratio) * graph.number_of_edges())
    num_of_pos_test_samples = 0

    # Remove random edges from the graph, leaving it connected
    # Fill in the blanks

    i = 0
    with tqdm(total=test_set_size) as pbar:
        while len(test_pos_samples) < test_set_size:
            residual_g.remove_edge(edges[i][0], edges[i][1])
            if nx.is_connected(residual_g):
                test_pos_samples.append(edges[i])
                num_of_pos_test_samples += 1
                pbar.update(1)
            else:
                residual_g.add_edge(edges[i][0], edges[i][1])
            i += 1

    # Check if we have the desired number of positive samples for testing set
    if num_of_pos_test_samples != test_set_size:
        raise ValueError("Enough positive edge samples could not be found!")

    # --- Step 2: Generate positive edge samples for training set ---
    # The remaining edges are simply considered for positive samples of the training set
    train_pos_samples = list(residual_g.edges())

    # --- Step 3: Generate the negative samples for testing and training sets ---
    # Fill in the blanks

    train_neg_samples = sample_non_edges(graph, len(train_pos_samples))
    test_neg_samples = sample_non_edges(graph, len(test_pos_samples))

    # --- Step 4: Combine sample lists and create corresponding labels ---
    # For training set
    train_samples = train_pos_samples + train_neg_samples
    train_labels = [1 for _ in train_pos_samples] + [0 for _ in train_neg_samples]
    # For testing set
    test_samples = test_pos_samples + test_neg_samples
    test_labels = [1 for _ in test_pos_samples] + [0 for _ in test_neg_samples]

    return residual_g, train_samples, train_labels, test_samples, test_labels


def save_graph_data(
    path: float,
    sub_graph: nx.Graph,
    residual_g: nx.Graph,
    train_samples: List[Tuple[str, str]],
    train_labels: List[int],
    test_samples: List[Tuple[str, str]],
    test_labels: List[int],
):
    """Save all training data from graphs

    Args:
        path (str): Path of saving data
        sub_graph (nx.Graph): Graph (with test edges)
        residual_g (nx.Graph): Residual graph (without test edges)
        train_samples (List[Tuple[str, str]]): List of training edges
        train_labels (List[int]): List of training labels
        test_samples (List[Tuple[str, str]]): List of testing edges
        test_labels (List[int]): List of testing labels
    """
    os.makedirs(path)
    nx.write_gml(sub_graph, os.path.join(path, "sub_graph.gml"))
    nx.write_gml(residual_g, os.path.join(path, "residual_g.gml"))

    with open(os.path.join(path, "training_graph_data.json"), "w") as file:
        json.dump(
            {
                "train_samples": train_samples,
                "train_labels": train_labels,
                "test_samples": test_samples,
                "test_labels": test_labels,
            },
            file,
        )


def load_graph_data(path: str):
    sub_graph = nx.read_gml(os.path.join(path, "sub_graph.gml"))
    residual_g = nx.read_gml(os.path.join(path, "residual_g.gml"))

    with open(os.path.join(path, "training_graph_data.json"), "r") as file:
        data_dict = json.load(file)
    train_samples = data_dict["train_samples"]
    train_labels = data_dict["train_labels"]
    test_samples = data_dict["test_samples"]
    test_labels = data_dict["test_labels"]

    return sub_graph, residual_g, train_samples, train_labels, test_samples, test_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subgraphratio", type=float, default=0.8, help="Part of the graph to use"
    )
    parser.add_argument(
        "--tainsetratio",
        type=float,
        default=0.8,
        help="Ratio of the training part of the dataset",
    )
    return parser.parse_args()


def main(sub_graph_ratio: float, tain_set_ratio: float):
    """Runs full script

    Args:
        sub_graph_ratio (float): Size ratio of the subgraph
        tain_set_ratio (float): Size ratio of the training set
    """
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

    graph = create_graph(nodes, train_dataset)
    print(f"{graph.number_of_edges()} edges and {graph.number_of_nodes()} nodes.")

    sub_graphs = make_sub_graphs(graph)
    sub_graph = sub_graphs[0]
    sub_graph = take_sub_graph(sub_graph, sub_graph_ratio)

    (
        residual_g,
        train_samples,
        train_labels,
        test_samples,
        test_labels,
    ) = generate_samples(graph=sub_graph, train_set_ratio=tain_set_ratio)

    path = f"experiments/{sub_graph_ratio}_{tain_set_ratio}"

    save_graph_data(
        path,
        sub_graph,
        residual_g,
        train_samples,
        train_labels,
        test_samples,
        test_labels,
    )


if __name__ == "__main__":

    namespace = parse_args()

    SUB_GRAPH_RATIO = namespace.subgraphratio
    TRAIN_SET_RATIO = namespace.tainsetratio

    main(SUB_GRAPH_RATIO, TRAIN_SET_RATIO)
