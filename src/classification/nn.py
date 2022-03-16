from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.classification.utils import _generate_submission_data, _show_roc_and_f1


def _train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
) -> List[float]:
    """Train function for one epoch

    Args:
        dataloader (DataLoader): Dataloader
        model (nn.Module): Model
        loss_fn (Callable): Loss function
        optimizer (torch.optim.Optimizer): Optimizer

    Returns:
        List[float]: Losses
    """
    model.train()
    losses = []
    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            losses.append(loss.item())
    return losses


def _test(dataloader: DataLoader, model: nn.Module, loss_fn: Callable) -> List[float]:
    """Test function for one epoch

    Args:
        dataloader (DataLoader): Dataloader
        model (nn.Module): Model
        loss_fn (Callable): Loss function

    Returns:
        List[float]: Test losses
    """
    model.eval()
    losses = []
    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        if batch % 100 == 0:
            losses.append(loss.item())
    return losses


def make_classification_model_neural_network(
    input_features: np.ndarray,
) -> Tuple[nn.Module, torch.optim.Optimizer, Callable]:
    """Generates Classification Neural Network

    Args:
        input_features (np.ndarray): Input features

    Returns:
        Tuple[nn.Module, torch.optim.Optimizer, Callable]: Model,
        optimizer and loss function
    """
    model = nn.Sequential(
        nn.Linear(input_features, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 1),
        nn.Sigmoid(),
    )
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()
    return model, optimizer, loss_fn


def prepare_data_for_neural_networks(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[DataLoader, DataLoader, Tuple[torch.Tensor, torch.Tensor]]:
    """Prepares data for the training of the neural network classifier

    Args:
        train_features (np.ndarray): Train features
        train_labels (np.ndarray): Train labels
        test_features (np.ndarray): Test features
        test_labels (np.ndarray): Test labels

    Returns:
        Tuple[DataLoader, DataLoader, Tuple[torch.Tensor, torch.Tensor]]: Train loader,
        Test loader, (mean, std) for then standardization
    """
    train_tensor_features = torch.Tensor(train_features)
    train_tensor_labels = torch.Tensor(train_labels)[
        ..., None
    ]  # .type(torch.LongTensor)

    test_tensor_features = torch.Tensor(test_features)
    test_tensor_labels = torch.Tensor(test_labels)[..., None]  # .type(torch.LongTensor)

    full_tensor = torch.concat([train_tensor_features, test_tensor_features])

    mean, std = full_tensor.mean(dim=0), full_tensor.std(dim=0)

    train_tensor_features = (train_tensor_features - mean) / std
    test_tensor_features = (test_tensor_features - mean) / std

    train_dataset = TensorDataset(train_tensor_features, train_tensor_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(test_tensor_features, test_tensor_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader, (mean, std)


def make_classification_neural_network(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[nn.Module, float, Tuple[torch.Tensor, torch.Tensor]]:
    """Runs the full classification pipeline with a neural network

    Args:
        train_features (np.ndarray): Train features
        train_labels (np.ndarray): Train labels
        test_features (np.ndarray): Test features
        test_labels (np.ndarray): Test labels

    Returns:
        Tuple[nn.Module, float, Tuple[torch.Tensor, torch.Tensor]]: Model,
        threshold, (mean, std) for the standardization
    """
    model, optimizer, loss_fn = make_classification_model_neural_network(
        train_features.shape[1]
    )
    train_loader, test_loader, (mean, std) = prepare_data_for_neural_networks(
        train_features, train_labels, test_features, test_labels
    )
    train_losses = []
    test_losses = []

    num_epochs = 10

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}...")
        epoch_train_losses = _train(train_loader, model, loss_fn, optimizer)
        epoch_test_losses = _test(test_loader, model, loss_fn)
        train_losses.append(np.mean(epoch_train_losses))
        print("Train ", train_losses[-1])
        test_losses.append(np.mean(epoch_test_losses))
        print("Test ", test_losses[-1])

    model.eval()
    test_preds = [model(feature) for feature, label in test_loader]
    test_preds = torch.Tensor(test_preds).tolist()
    th = _show_roc_and_f1(test_labels, test_preds)
    return model, th, (mean, std)


def generate_submission_data_for_neural_networks(
    model: nn.Module,
    th: float,
    features: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    name: str,
):
    """Generate the submission csv for the Kaggle Challenge

    Args:
        model (nn.Module): Model
        th (float): Threshold
        features (np.ndarray): Features
        mean (torch.Tensor): Mean for the standardization
        std (torch.Tensor): Std for the standardization
        name (str): Name of the submission csv file
    """
    features = (features - mean.numpy()) / std.numpy()
    submit_loader = DataLoader(features, batch_size=1, shuffle=False)

    model.eval()
    submit_preds = [
        int(model(feature.float())[0].item() > th) for feature in submit_loader
    ]
    _generate_submission_data(submit_preds, name)
