from __future__ import annotations

import json

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from ml_engine import (
    available_algorithms,
    decision_boundary_b64,
    load_data,
    split_train_test,
    train_on_split,
    train_one,
)

app = Flask(__name__)
CORS(app)

MODEL_STATE: dict = {
    "model": None,
    "meta": None,
}


def _clear_model_state():
    MODEL_STATE["model"] = None
    MODEL_STATE["meta"] = None


def _task_type(raw: str | None) -> str:
    return "regression" if str(raw or "classification").lower() == "regression" else "classification"


def _parse_params(raw) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/algorithms", methods=["GET"])
def algorithms():
    return jsonify({"success": True, "algorithms": available_algorithms()})


@app.route("/model/status", methods=["GET"])
def model_status():
    meta = MODEL_STATE.get("meta") or {}
    return jsonify(
        {
            "trained": MODEL_STATE.get("model") is not None,
            "task_type": meta.get("task_type", "classification"),
            "algorithm": meta.get("algorithm"),
            "feature_names": meta.get("feature_names", []),
            "target_names": meta.get("target_names", []),
            "n_features": meta.get("n_features", 0),
        }
    )


@app.route("/reset", methods=["POST"])
def reset_model():
    _clear_model_state()
    return jsonify({"success": True, "message": "Model state cleared"})


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True, silent=True) or {}
    row = body.get("input")
    if MODEL_STATE.get("model") is None:
        return jsonify({"success": False, "error": "No trained model. Train a model first."}), 400
    meta = MODEL_STATE.get("meta") or {}
    n_expected = int(meta.get("n_features") or 0)
    task_type = _task_type(meta.get("task_type"))

    if not isinstance(row, list):
        return jsonify({"success": False, "error": "Body must include 'input' as a list of feature values"}), 400
    if len(row) != n_expected:
        return jsonify({"success": False, "error": f"Expected {n_expected} features, got {len(row)}"}), 400

    try:
        X_row = np.array([row], dtype=np.float64)
        pred = MODEL_STATE["model"].predict(X_row)[0]

        if task_type == "classification":
            idx = int(pred)
            tnames = meta.get("target_names") or []
            label = str(tnames[idx]) if 0 <= idx < len(tnames) else str(pred)
            return jsonify({"success": True, "prediction": label, "task_type": task_type})

        return jsonify({"success": True, "prediction": float(pred), "task_type": task_type})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/train", methods=["POST"])
def train():
    test_size = 0.2
    random_state = 42
    dataset_name = "iris"
    csv_text = None
    target_column = None
    algorithm = "decision_tree"
    params: dict = {}
    include_boundary = True
    task_type = "classification"

    if request.content_type and "multipart/form-data" in request.content_type:
        algorithm = request.form.get("algorithm", "decision_tree")
        task_type = _task_type(request.form.get("task_type"))
        params = _parse_params(request.form.get("parameters"))
        dataset_name = request.form.get("dataset") or "iris"
        target_column = request.form.get("target_column") or None
        test_size = float(request.form.get("test_size") or 0.2)
        random_state = int(request.form.get("random_state") or 42)
        include_boundary = request.form.get("include_boundary", "true").lower() in ("1", "true", "yes")
        f = request.files.get("file")
        if f and f.filename:
            csv_text = f.read().decode("utf-8", errors="replace")
    else:
        body = request.get_json(force=True, silent=True) or {}
        algorithm = body.get("algorithm", "decision_tree")
        task_type = _task_type(body.get("task_type"))
        params = body.get("parameters") or {}
        dataset_name = body.get("dataset", "iris")
        csv_text = body.get("csv_text")
        target_column = body.get("target_column")
        test_size = float(body.get("test_size", 0.2))
        random_state = int(body.get("random_state", 42))
        include_boundary = bool(body.get("include_boundary", True))

    try:
        X, y, feature_names, target_names = load_data(dataset_name, csv_text, target_column, task_type)
        result, fitted = train_one(algorithm, params, X, y, test_size, random_state, feature_names, task_type)
        result["algorithm"] = algorithm
        result["dataset"] = dataset_name or "custom"
        result["feature_names"] = feature_names
        result["target_names"] = target_names

        MODEL_STATE["model"] = fitted
        MODEL_STATE["meta"] = {
            "task_type": task_type,
            "algorithm": algorithm,
            "feature_names": feature_names,
            "target_names": target_names,
            "n_features": int(X.shape[1]),
            "parameters": params,
        }

        result["decision_boundary_png"] = (
            decision_boundary_b64(X, y, algorithm, params, test_size, random_state, task_type)
            if include_boundary
            else None
        )

        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/compare", methods=["POST"])
def compare():
    body = request.get_json(force=True, silent=True) or {}
    task_type = _task_type(body.get("task_type"))
    test_size = float(body.get("test_size", 0.2))
    random_state = int(body.get("random_state", 42))
    dataset_name = body.get("dataset", "iris")
    csv_text = body.get("csv_text")
    target_column = body.get("target_column")
    runs = body.get("runs") or []

    if len(runs) < 2:
        return jsonify({"success": False, "error": "Provide at least two runs in 'runs' array"}), 400

    try:
        X, y, feature_names, target_names = load_data(dataset_name, csv_text, target_column, task_type)
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state, task_type)
        results = []
        for i, run in enumerate(runs[:4]):
            algo = run.get("algorithm", "decision_tree")
            params = run.get("parameters") or {}
            out = train_on_split(algo, params, X_train, X_test, y_train, y_test, feature_names, task_type)
            out["algorithm"] = algo
            out["label"] = run.get("label") or f"Model {i + 1}: {algo}"
            out["parameters"] = params
            out["feature_names"] = feature_names
            out["target_names"] = target_names
            out["decision_boundary_png"] = decision_boundary_b64(X, y, algo, params, test_size, random_state, task_type)
            results.append(out)

        key = "accuracy" if task_type == "classification" else "r2_score"
        best = max(results, key=lambda r: r.get(key, float("-inf")))
        best_label = best.get("label", best["algorithm"])

        return jsonify(
            {
                "success": True,
                "task_type": task_type,
                "results": results,
                "best": {
                    "algorithm": best["algorithm"],
                    "score_key": key,
                    "score": best.get(key),
                    "label": best_label,
                },
                "target_names": target_names,
                "feature_names": feature_names,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/datasets", methods=["GET"])
def datasets():
    return jsonify({"sample_datasets": ["iris", "wine"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
