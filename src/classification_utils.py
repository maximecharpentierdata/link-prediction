from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
) -> List[float]:
    model.train()
    losses = []
    for batch, (X, y) in enumerate(dataloader):
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
    for batch, (X, y) in enumerate(dataloader):
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

    num_epochs = 30

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
    fpr, tpr, thresholds = roc_curve(test_labels, test_preds)
    roc_auc = roc_auc_score(test_labels, test_preds)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkred", label="ROC curve (area = %0.3f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="lightgray", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    plt.show()

    th = thresholds[np.argmax(tpr - fpr)]
    print("f1 score", f1_score(test_labels, [pred > th for pred in test_preds]))
    return model, th, (mean, std)


def generate_submission_data_for_neural_networks(
    model: nn.Module,
    th: float,
    features: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    name: str,
):
    features = (features - mean) / std

    submit_loader = DataLoader(features, batch_size=1, shuffle=False)

    model.eval()
    submit_preds = [
        int(model(feature.float())[0].item() > th) for feature in submit_loader
    ]
    submission_df = pd.DataFrame(dict(category=submit_preds))
    submission_df = submission_df.reset_index()
    submission_df.columns = ["id", "category"]
    submission_df.to_csv(f"../submission_{name}.csv", index=None)
