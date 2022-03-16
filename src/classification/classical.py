from typing import Tuple
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.classification.utils import _generate_submission_data, _show_roc_and_f1


def preprocess_data_classical(
    train_features: np.ndarray, test_features: np.ndarray
) -> Tuple[StandardScaler, np.ndarray, np.ndarray]:
    """Preprocess data for classical training

    Args:
        train_features (np.ndarray): Train features
        test_features (np.ndarray): Test features

    Returns:
        Tuple[StandardScaler, np.ndarray, np.ndarray]: Scaler, train features scaled and test features scaled
    """
    scaler = StandardScaler()
    full_features = np.concatenate([train_features, test_features], axis=0)
    full_features_scaled = scaler.fit_transform(full_features)
    return (
        scaler,
        full_features_scaled[: train_features.shape[0], :],
        full_features_scaled[train_features.shape[0] :, :],
    )


def make_classication_lr(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[LogisticRegression, float, StandardScaler]:
    """Runs the full classifcation pipeline with logistic regression

    Args:
        train_features (np.ndarray): Train features
        train_labels (np.ndarray): Train labels
        test_features (np.ndarray): Test features
        test_labels (np.ndarray): Test labels

    Returns:
        Tuple[LogisticRegression, float, StandardScaler]: Model, threshold and scaler
    """
    scaler, train_features_scaled, test_features_scaled = preprocess_data_classical(
        train_features, test_features
    )
    lr = LogisticRegression()
    lr.fit(train_features_scaled, train_labels)
    test_preds = lr.predict_proba(test_features_scaled)[:, 1]
    th = _show_roc_and_f1(test_labels, test_preds)
    return lr, th, scaler


def make_classication_svm(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[SVC, float, StandardScaler]:
    """Runs the full classifcation pipeline with SVM

    Args:
        train_features (np.ndarray): Train features
        train_labels (np.ndarray): Train labels
        test_features (np.ndarray): Test features
        test_labels (np.ndarray): Test labels

    Returns:
        Tuple[SVC, float, StandardScaler]: Model, threshold and scaler
    """
    scaler, train_features_scaled, test_features_scaled = preprocess_data_classical(
        train_features, test_features
    )
    svc = SVC(probability=True)
    svc.fit(train_features_scaled, train_labels)
    test_preds = svc.predict_proba(test_features_scaled)[:, 1]
    th = _show_roc_and_f1(test_labels, test_preds)
    return svc, th, scaler


def make_classication_rf(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[RandomForestClassifier, float, StandardScaler]:
    """Runs the full classifcation pipeline with random forest

    Args:
        train_features (np.ndarray): Train features
        train_labels (np.ndarray): Train labels
        test_features (np.ndarray): Test features
        test_labels (np.ndarray): Test labels

    Returns:
        Tuple[RandomForestClassifier, float, StandardScaler]: Model, threshold and scaler
    """
    scaler, train_features_scaled, test_features_scaled = preprocess_data_classical(
        train_features, test_features
    )
    rf = RandomForestClassifier()
    rf.fit(train_features_scaled, train_labels)
    test_preds = rf.predict_proba(test_features_scaled)[:, 1]
    th = _show_roc_and_f1(test_labels, test_preds)
    return rf, th, scaler


def make_classication_xgboost(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[XGBClassifier, float, StandardScaler]:
    """Runs the full classifcation pipeline with XGBoost

    Args:
        train_features (np.ndarray): Train features
        train_labels (np.ndarray): Train labels
        test_features (np.ndarray): Test features
        test_labels (np.ndarray): Test labels

    Returns:
        Tuple[XGBClassifier, float, StandardScaler]: Model, threshold and scaler
    """
    scaler, train_features_scaled, test_features_scaled = preprocess_data_classical(
        train_features, test_features
    )
    boosting = XGBClassifier()
    boosting.fit(train_features_scaled, train_labels)
    test_preds = boosting.predict_proba(test_features_scaled)[:, 1]
    th = _show_roc_and_f1(test_labels, test_preds)
    return boosting, th, scaler


def generate_submission_data_classical_model(
    model: ClassifierMixin,
    th: float,
    features: np.ndarray,
    scaler: StandardScaler,
    name: str,
):
    """Generate submissions csv for the Kaggle Data challenge

    Args:
        model (ClassifierMixin): Model
        th (float): Threshold
        features (np.ndarray): Features
        scaler (StandardScaler): Scaler for the standardization
        name (str): Name of the csv file
    """
    features_scaled = scaler.transform(features)

    submit_preds = [
        int(score > th) for score in model.predict_proba(features_scaled)[:, 1]
    ]
    _generate_submission_data(submit_preds, name)
