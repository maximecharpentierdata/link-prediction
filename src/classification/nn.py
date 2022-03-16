from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.classification.utils import generate_submission_data, show_roc_and_f1


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
) -> List[float]:
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


def test(dataloader: DataLoader, model: nn.Module, loss_fn: Callable):
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
        epoch_train_losses = train(train_loader, model, loss_fn, optimizer)
        epoch_test_losses = test(test_loader, model, loss_fn)
        train_losses.append(np.mean(epoch_train_losses))
        print("Train ", train_losses[-1])
        test_losses.append(np.mean(epoch_test_losses))
        print("Test ", test_losses[-1])

    model.eval()
    test_preds = [model(feature) for feature, label in test_loader]
    test_preds = torch.Tensor(test_preds).tolist()
    th = show_roc_and_f1(test_labels, test_preds)
    return model, th, (mean, std)


def generate_submission_data_for_neural_networks(
    model: nn.Module,
    th: float,
    features: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    name: str,
):
    features = (features - mean.numpy()) / std.numpy()
    submit_loader = DataLoader(features, batch_size=1, shuffle=False)

    model.eval()
    submit_preds = [
        int(model(feature.float())[0].item() > th) for feature in submit_loader
    ]
    generate_submission_data(submit_preds, name)
