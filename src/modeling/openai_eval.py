import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.util import get_records_list

MODEL_KEY = "gpt-4o-mini-2024-07-18"

# Define mappings for jurisdiction and insurance type
JURISDICTION_MAP = {"NY": 0, "CA": 1, "Unspecified": 2}
INSURANCE_TYPE_MAP = {"Commercial": 0, "Medicaid": 1, "Unspecified": 2}


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    # Check if we have enough unique classes for ROC AUC
    try:
        # Convert predictions to one-hot format for ROC AUC if they're not already
        if len(predictions.shape) == 1:  # If predictions are just class labels
            n_classes = len(np.unique(labels))
            one_hot_preds = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                if pred < n_classes:  # Ensure valid index
                    one_hot_preds[i, pred] = 1
            roc_auc = roc_auc_score(labels, one_hot_preds, multi_class="ovr")
        else:
            # Assuming predictions are already probabilities/scores
            roc_auc = roc_auc_score(labels, predictions, multi_class="ovr")
    except ValueError:
        # Handle case when there are classes with no predictions
        roc_auc = float("nan")

    acc = accuracy_score(labels, predictions)

    # Macro
    macro_prec = precision_score(labels, predictions, average="macro")
    macro_rec = recall_score(labels, predictions, average="macro")
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=np.nan)

    # Micro
    micro_prec = precision_score(labels, predictions, average="micro")
    micro_rec = recall_score(labels, predictions, average="micro")
    micro_f1 = f1_score(labels, predictions, average="micro", zero_division=np.nan)

    # Class-level
    prec = precision_score(labels, predictions, average=None)
    rec = recall_score(labels, predictions, average=None)
    f1 = f1_score(labels, predictions, average=None, zero_division=np.nan)

    return {
        "metric_fn_name": "compute_metrics",
        "accuracy": acc,
        "macro_precision": macro_prec.tolist(),
        "macro_recall": macro_rec.tolist(),
        "macro_f1": macro_f1.tolist(),
        "micro_precision": micro_prec.tolist(),
        "micro_recall": micro_rec.tolist(),
        "micro_f1": micro_f1.tolist(),
        "class_precision": prec.tolist(),
        "class_recall": rec.tolist(),
        "class_f1": f1.tolist(),
        "roc_auc": roc_auc,
    }


def compute_metrics_w_threshold(predictions: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    # Convert predictions to probabilities if they're not already
    if len(predictions.shape) == 1:  # If predictions are just class labels
        n_classes = max(3, np.max(predictions) + 1)  # Ensure we have at least 3 classes
        probs = np.zeros((len(predictions), n_classes))
        for i, pred in enumerate(predictions):
            if pred < n_classes:  # Ensure valid index
                probs[i, pred] = 1
    else:
        # Assuming predictions are already probabilities/scores
        probs = predictions

    try:
        roc_auc = roc_auc_score(labels, probs, multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")

    # Apply threshold logic as in original implementation
    pred_labels = np.full_like(labels, -1)  # Initialize with -1 for unassigned classes

    # Assign class 0 where the probability meets or exceeds the threshold
    class_0_mask = probs[:, 0] >= threshold
    pred_labels[class_0_mask] = 0

    # If probability does not exceed threshold, take argmax of upheld/overturned class options
    remaining_mask = pred_labels == -1
    remaining_preds = probs[remaining_mask, 1:]  # Exclude class 0
    pred_labels[remaining_mask] = np.argmax(remaining_preds, axis=1) + 1  # + 1 to adjust for class index shift

    # Compute metrics
    acc = accuracy_score(labels, pred_labels)

    # Macro
    macro_prec = precision_score(labels, pred_labels, average="macro")
    macro_rec = recall_score(labels, pred_labels, average="macro")
    macro_f1 = f1_score(labels, pred_labels, average="macro", zero_division=np.nan)

    # Micro
    micro_prec = precision_score(labels, pred_labels, average="micro")
    micro_rec = recall_score(labels, pred_labels, average="micro")
    micro_f1 = f1_score(labels, pred_labels, average="micro", zero_division=np.nan)

    # Class-level
    prec = precision_score(labels, pred_labels, average=None)
    rec = recall_score(labels, pred_labels, average=None)
    f1 = f1_score(labels, pred_labels, average=None, zero_division=np.nan)

    return {
        "metric_fn_name": "compute_metrics_w_threshold",
        "accuracy": acc,
        "macro_precision": macro_prec.tolist(),
        "macro_recall": macro_rec.tolist(),
        "macro_f1": macro_f1.tolist(),
        "micro_precision": micro_prec.tolist(),
        "micro_recall": micro_rec.tolist(),
        "micro_f1": micro_f1.tolist(),
        "class_precision": prec.tolist(),
        "class_recall": rec.tolist(),
        "class_f1": f1.tolist(),
        "roc_auc": roc_auc,
    }


def merge_pred_gt(response_path: str, answers_path: str) -> dict:
    merged = {}
    gt = get_records_list(answers_path)
    responses = get_records_list(response_path)

    # Extract ground truth and metadata
    for rec in gt:
        # Store decision and metadata in merged dict
        merged[rec["custom_id"]] = {
            "answer": rec["decision"],
            "jurisdiction": rec.get("jurisdiction", "Unspecified"),
            "insurance_type": rec.get("insurance_type", "Unspecified"),
            "sufficiency_id": rec.get("sufficiency_id", 1),  # Default to sufficient if not specified
        }

    # Extract predictions
    for rec in responses:
        pred = rec["parsed_response"]

        # Add prediction to the merged dict if the ID exists
        if rec["custom_id"] in merged:
            merged[rec["custom_id"]]["pred"] = pred

    return merged


def construct_label(outcome, sufficiency_id, label_map):
    """Construct 3-class label from outcome, if sufficient background."""
    if sufficiency_id == 0:
        return 0
    else:
        return label_map.get(outcome, 0)  # Default to 0 if outcome not in map


def eval_preds(threshold=None) -> dict:
    outcomes_dataset = "test_backgrounds_suff"
    answers_path = f"./data/provider_annotated_outcomes/openai/{outcomes_dataset}/hicric_eval_request_answers.jsonl"
    response_path = (
        f"./data/provider_annotated_outcomes/openai/{outcomes_dataset}/hicric_eval_response_{MODEL_KEY}.jsonl"
    )

    merged = merge_pred_gt(response_path, answers_path)
    out_map = {"Insufficient": 0, "Upheld": 1, "Overturned": 2}

    preds = []
    labels = []
    jurisdiction_ids = []
    insurance_type_ids = []

    # Favorable evaluation: don't count non-answers against GPT, to be generous
    for id, rec in merged.items():
        print(rec)
        if "pred" not in rec or rec["pred"] is None:
            continue
        if rec["pred"].get("decision") is None:
            continue
        if rec["pred"].get("decision") not in out_map.keys():
            continue
        else:
            # Extract prediction
            preds.append(out_map[rec["pred"]["decision"]])

            # Extract label with sufficiency consideration
            sufficiency_id = rec.get("sufficiency_id", 1)  # Default to sufficient if not specified
            labels.append(construct_label(rec["answer"], sufficiency_id, out_map))

            # Extract metadata
            j = rec.get("jurisdiction", "Unspecified")
            i = rec.get("insurance_type", "Unspecified")
            jurisdiction_ids.append(JURISDICTION_MAP.get(j, JURISDICTION_MAP["Unspecified"]))
            insurance_type_ids.append(INSURANCE_TYPE_MAP.get(i, INSURANCE_TYPE_MAP["Unspecified"]))

    if len(preds) < len(merged):
        print(f"Model failed to produce valid json for {len(merged) - len(preds)} values.")

    # Convert to numpy arrays for metrics computation
    preds_np = np.array(preds)
    labels_np = np.array(labels)

    # Compute standard metrics
    metrics = compute_metrics(preds_np, labels_np)

    # Add metadata distribution info to metrics
    metrics["jurisdiction_distribution"] = {j: jurisdiction_ids.count(JURISDICTION_MAP[j]) for j in JURISDICTION_MAP}
    metrics["insurance_type_distribution"] = {
        i: insurance_type_ids.count(INSURANCE_TYPE_MAP[i]) for i in INSURANCE_TYPE_MAP
    }

    # Compute threshold metrics if threshold is provided
    if threshold is not None:
        threshold_metrics = compute_metrics_w_threshold(preds_np, labels_np, threshold)

    # Print and save metrics
    print(metrics)
    output_dir = f"./data/provider_annotated_outcomes/openai/{outcomes_dataset}"
    with open(os.path.join(output_dir, f"{MODEL_KEY}_test_metrics.json"), "w") as f:
        json.dump(metrics, f)

    if threshold is not None:
        print(threshold_metrics)
        with open(os.path.join(output_dir, f"{MODEL_KEY}_test_metrics_w_threshold.json"), "w") as f:
            json.dump(threshold_metrics, f)

    return metrics


if __name__ == "__main__":
    # Optionally specify a threshold for the additional metrics
    threshold = 0.5  # Set to None if you don't want threshold-based metrics
    eval_preds(threshold)
