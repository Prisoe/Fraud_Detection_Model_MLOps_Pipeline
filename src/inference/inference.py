"""
Inference script — Fraud Detection
====================================
Loaded by SageMaker endpoint at startup via model_fn().
Handles CSV and JSON payloads, returns fraud predictions + probabilities.
Loads optimal threshold from threshold.json saved during training.
"""
import json
import os
import joblib
import numpy as np

_model     = None
_threshold = 0.5
EXPECTED_FEATURES = 29  # V1-V28 + Amount (Time dropped in preprocessing)


def model_fn(model_dir):
    global _model, _threshold
    _model = joblib.load(os.path.join(model_dir, "model.joblib"))

    # Load optimal threshold saved during training
    t_path = os.path.join(model_dir, "threshold.json")
    if os.path.exists(t_path):
        with open(t_path) as fh:
            data = json.load(fh)
        _threshold = float(data["threshold"])
        print(f"[inference] Loaded threshold={_threshold:.4f}")
    else:
        print(f"[inference] threshold.json not found, using default={_threshold}")
    return _model


def input_fn(body, content_type):
    if content_type == "text/csv":
        rows = [r.strip() for r in body.strip().splitlines() if r.strip()]
        data = [[float(x) for x in r.split(",")] for r in rows]
    elif content_type == "application/json":
        obj = json.loads(body)
        data = obj["instances"] if isinstance(obj, dict) and "instances" in obj else obj
    else:
        raise ValueError(f"Unsupported content-type: {content_type}")

    arr = np.array(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features (V1-V28 + Amount). "
            f"Got {arr.shape[1]}. Note: Time column should be dropped before sending."
        )
    return arr


def predict_fn(X, model):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= _threshold).astype(int)
    return {"probabilities": probs, "predictions": preds}


def output_fn(result, accept):
    payload = {
        "predictions":   result["predictions"].tolist(),
        "probabilities": [round(float(p), 6) for p in result["probabilities"]],
        "threshold":     _threshold,
        "n_flagged":     int(result["predictions"].sum()),
    }
    return json.dumps(payload), "application/json"
