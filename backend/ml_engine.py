"""Training utilities for sklearn classifiers and regressors."""

from __future__ import annotations

import base64
import io
from typing import Any
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.datasets import load_iris, load_wine  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402

from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

CLASSIFICATION_ALGORITHMS = {
    "decision_tree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
    "naive_bayes": GaussianNB,
    "logistic_regression": LogisticRegression,
    "random_forest_classifier": RandomForestClassifier,
}

REGRESSION_ALGORITHMS = {
    "linear_regression": LinearRegression,
    "random_forest_regressor": RandomForestRegressor,
}

SAMPLE_DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _normalize_task_type(task_type: str | None) -> str:
    return "regression" if str(task_type or "classification").lower() == "regression" else "classification"


def _binary_roc(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any] | None:
    if len(np.unique(y_test)) != 2:
        return None
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(X_test)
        if proba.shape[1] < 2:
            return None
        y_score = proba[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(auc(fpr, tpr)),
        }
    except Exception:
        return None



def _evaluate_classification(
    model: Any,
    algorithm: str,
    feature_names: list[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    classes = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
    labels_for_report = classes
    target_labels_str = [str(c) for c in labels_for_report]
    report_raw = classification_report(
        y_test,
        y_pred,
        labels=labels_for_report,
        target_names=target_labels_str,
        output_dict=True,
        zero_division=0,
    )
    report = _json_safe(report_raw)
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    fi: list[dict[str, Any]] | None = None
    if algorithm in ("decision_tree", "random_forest_classifier") and hasattr(model, "feature_importances_"):
        names = feature_names or [f"f{i}" for i in range(len(model.feature_importances_))]
        fi = [
            {
                "feature": names[i] if i < len(names) else f"f{i}",
                "importance": float(model.feature_importances_[i]),
            }
            for i in range(len(model.feature_importances_))
        ]

    roc = _binary_roc(model, X_test, y_test)

    return {
        "task_type": "classification",
        "accuracy": acc,
        "confusion_matrix": cm,
        "classes": [str(c) for c in classes],
        "classification_report": report,
        "precision_score": prec,
        "recall_score": rec,
        "f1_score": f1,
        "feature_importance": fi,
        "roc_curve": roc,
        "predictions_sample": y_pred[:20].tolist(),
        "y_test_sample": y_test[:20].tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def _evaluate_regression(
    model: Any,
    algorithm: str,
    feature_names: list[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))

    fi: list[dict[str, Any]] | None = None
    if algorithm == "random_forest_regressor" and hasattr(model, "feature_importances_"):
        names = feature_names or [f"f{i}" for i in range(len(model.feature_importances_))]
        fi = [
            {
                "feature": names[i] if i < len(names) else f"f{i}",
                "importance": float(model.feature_importances_[i]),
            }
            for i in range(len(model.feature_importances_))
        ]

    return {
        "task_type": "regression",
        "r2_score": r2,
        "mae": mae,
        "mse": mse,
        "feature_importance": fi,
        "predictions_sample": y_pred[:20].tolist(),
        "y_test_sample": y_test[:20].tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


def load_data(
    dataset_name: str | None,
    csv_text: str | None,
    target_column: str | None,
    task_type: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return X, y, feature_names, target_names."""
    task_type = _normalize_task_type(task_type)

    if csv_text:
        df = pd.read_csv(io.StringIO(csv_text))
        if target_column and target_column in df.columns:
            y_raw = df[target_column].values
            X = df.drop(columns=[target_column]).select_dtypes(include=[np.number]).values
            feature_names = list(df.drop(columns=[target_column]).select_dtypes(include=[np.number]).columns)
        else:
            y_raw = df.iloc[:, -1].values
            X = df.iloc[:, :-1].select_dtypes(include=[np.number]).values
            num_cols = df.iloc[:, :-1].select_dtypes(include=[np.number]).columns
            feature_names = list(num_cols)

        if task_type == "classification":
            if y_raw.dtype == object:
                uniq = np.unique(y_raw.astype(str))
                mapping = {v: i for i, v in enumerate(uniq)}
                y = np.array([mapping[str(v)] for v in y_raw], dtype=np.int64)
                target_names = [str(v) for v in uniq]
            else:
                y = y_raw.astype(np.int64)
                classes = np.unique(y)
                target_names = [str(x) for x in classes]
        else:
            y = y_raw.astype(np.float64)
            target_names = ["target"]

        return X.astype(np.float64), y, feature_names, target_names

    if not dataset_name or dataset_name not in SAMPLE_DATASETS:
        dataset_name = "iris"

    bundle = SAMPLE_DATASETS[dataset_name]()
    X = np.asarray(bundle.data, dtype=np.float64)
    y_raw = np.asarray(bundle.target)
    feature_names = list(bundle.feature_names)

    if task_type == "classification":
        y = np.asarray(y_raw, dtype=np.int64)
        target_names = [str(t) for t in bundle.target_names]
    else:
        y = np.asarray(y_raw, dtype=np.float64)
        target_names = ["target"]

    return X, y, feature_names, target_names


def build_model(algorithm: str, params: dict[str, Any], task_type: str):
    task_type = _normalize_task_type(task_type)
    algo_map = CLASSIFICATION_ALGORITHMS if task_type == "classification" else REGRESSION_ALGORITHMS
    if algorithm not in algo_map:
        raise ValueError(f"Unknown {task_type} algorithm: {algorithm}")

    cls = algo_map[algorithm]

    if algorithm == "decision_tree":
        p = {
            "max_depth": int(params.get("max_depth") or 5),
            "min_samples_split": int(params.get("min_samples_split") or 2),
            "random_state": int(params.get("random_state", 42)),
        }
        return cls(**p)

    if algorithm == "knn":
        k = int(params.get("k") or params.get("n_neighbors") or 5)
        return cls(n_neighbors=max(1, k))

    if algorithm == "logistic_regression":
        c = float(params.get("c") or 1.0)
        max_iter = int(params.get("max_iter") or 500)
        return cls(C=max(c, 1e-6), max_iter=max_iter)

    if algorithm == "random_forest_classifier":
        return cls(
            n_estimators=int(params.get("n_estimators") or 100),
            max_depth=int(params.get("max_depth")) if params.get("max_depth") else None,
            random_state=int(params.get("random_state", 42)),
        )

    if algorithm == "random_forest_regressor":
        return cls(
            n_estimators=int(params.get("n_estimators") or 100),
            max_depth=int(params.get("max_depth")) if params.get("max_depth") else None,
            random_state=int(params.get("random_state", 42)),
        )

    return cls()


def split_train_test(X, y, test_size: float, random_state: int, task_type: str):
    task_type = _normalize_task_type(task_type)
    if task_type == "classification":
        uniq = np.unique(y)
        strat = y if len(uniq) > 1 and len(y) >= 10 else None
        try:
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)
        except ValueError:
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_one(
    algorithm: str,
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
    feature_names: list[str],
    task_type: str,
) -> tuple[dict[str, Any], Any]:
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state, task_type)
    model = build_model(algorithm, params, task_type)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if _normalize_task_type(task_type) == "classification":
        result = _evaluate_classification(model, algorithm, feature_names, X_test, y_test, y_pred, X_train, y_train)
    else:
        result = _evaluate_regression(model, algorithm, feature_names, X_test, y_test, y_pred, X_train, y_train)

    return result, model


def train_on_split(
    algorithm: str,
    params: dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    task_type: str,
) -> dict[str, Any]:
    model = build_model(algorithm, params, task_type)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if _normalize_task_type(task_type) == "classification":
        return _evaluate_classification(model, algorithm, feature_names, X_test, y_test, y_pred, X_train, y_train)
    return _evaluate_regression(model, algorithm, feature_names, X_test, y_test, y_pred, X_train, y_train)


def decision_boundary_b64(
    X: np.ndarray,
    y: np.ndarray,
    algorithm: str,
    params: dict[str, Any],
    test_size: float,
    random_state: int,
    task_type: str,
) -> str | None:
    if _normalize_task_type(task_type) != "classification":
        return None
    if X.shape[1] < 2:
        return None

    X2 = X[:, :2]
    X_train, X_test, y_train, y_test = split_train_test(X2, y, test_size, random_state, "classification")
    model = build_model(algorithm, params, "classification")
    model.fit(X_train, y_train)

    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 120),
        np.linspace(y_min, y_max, 120),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    try:
        Z = model.predict(grid).reshape(xx.shape)
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    ax.contourf(xx, yy, Z, alpha=0.35, cmap="tab10")
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=y, edgecolors="k", cmap="tab10", s=28)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision boundary (first 2 features)")
    fig.colorbar(scatter, ax=ax, shrink=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def available_algorithms() -> dict[str, list[str]]:
    return {
        "classification": list(CLASSIFICATION_ALGORITHMS.keys()),
        "regression": list(REGRESSION_ALGORITHMS.keys()),
    }
    
