import argparse
import json
import os

import numpy as np
import scipy
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)

from src.modeling.train_outcome_predictor import TextClassificationWithMetadata
from src.modeling.util import load_config
from src.util import get_records_list

MODEL_DIR = "./models/overturn_predictor"
ID2LABEL = {0: "Insufficient", 1: "Upheld", 2: "Overturned"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Define mappings for jurisdiction and insurance type
JURISDICTION_MAP = {"NY": 0, "CA": 1, "Unspecified": 2}
INSURANCE_TYPE_MAP = {"Commercial": 0, "Medicaid": 1, "Unspecified": 2}


# TODO: centralize label construction, label consts
def construct_label(outcome, sufficiency_id, label2id):
    """Construct 3-class label from outcome, if sufficient background."""
    if sufficiency_id == 0:
        return 0
    else:
        return label2id[outcome]


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    roc_auc = roc_auc_score(labels, scipy.special.softmax(predictions, axis=-1), multi_class="ovr")

    predictions = torch.argmax(predictions, dim=-1).tolist()

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
    softmax_preds = scipy.special.softmax(predictions, axis=-1)

    roc_auc = roc_auc_score(labels, softmax_preds, multi_class="ovr")

    pred_labels = np.full_like(labels, -1)  # Initialize with -1 for unassigned classes

    # Assign class 0 where the probability meets or exceeds the threshold
    class_0_mask = softmax_preds[:, 0] >= threshold
    pred_labels[class_0_mask] = 0

    # If probability does not exceed threshold, take argmax of upheld/overturned class otpions
    remaining_mask = pred_labels == -1
    remaining_preds = softmax_preds[remaining_mask, 1:]  # Exclude class 0
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


class ClassificationPipeline(TextClassificationPipeline):
    """Pipeline instance that actually outputs raw logits."""

    def postprocess(self, model_outputs):
        out_logits = model_outputs["logits"]
        return out_logits


# Check if the model has a custom forward method that supports metadata
def is_metadata_model(model):
    """Check if model has metadata capabilities by inspecting its methods"""
    # Look for the key attributes in our custom model
    has_j_embeddings = hasattr(model, "jurisdiction_embeddings")
    has_i_embeddings = hasattr(model, "insurance_type_embeddings")

    # Check if model's forward method signature includes the metadata params
    forward_sig = model.forward.__code__.co_varnames
    accepts_metadata = "jurisdiction_id" in forward_sig and "insurance_type_id" in forward_sig

    return has_j_embeddings and has_i_embeddings and accepts_metadata


def count_parameters(model):
    """Count the number of trainable parameters in the model and calculate size in MB"""
    param_count = 0
    size_in_bytes = 0

    for p in model.parameters():
        if p.requires_grad:
            param_count += p.numel()
            size_in_bytes += p.numel() * p.element_size()

    size_in_mb = size_in_bytes / (1024 * 1024)
    return param_count, size_in_mb


def main(config_path: str):
    cfg = load_config(config_path)

    # Checkpoint config
    hicric_pretrained = cfg["hicric_pretrained"]
    base_model_name = cfg["base_model_name"]
    dataset_path = cfg["test_data_path"]
    checkpoint_name = cfg["checkpoint_name"]
    threshold = cfg.get("eval_threshold", None)

    if hicric_pretrained:
        base_model_name = f"{base_model_name}-hicric-pretrained"

    # Load raw dataset
    test_dataset = get_records_list(dataset_path)

    # Set up paths
    checkpoints_dir = os.path.join(MODEL_DIR, "train_backgrounds_suff_augmented", base_model_name)
    print("Evaluating from checkpoint: ", checkpoint_name)
    ckpt_path = os.path.join(checkpoints_dir, checkpoint_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    # Load model
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     ckpt_path, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    # )
    model = TextClassificationWithMetadata.from_pretrained(
        ckpt_path,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Count model parameters
    param_count, size_in_mb = count_parameters(model)
    print(f"Model size: {param_count:,} parameters ({size_in_mb:.2f} MB)")

    # Check if model supports metadata
    supports_metadata = is_metadata_model(model)

    # Prepare data
    text_records = [rec["text"] for rec in test_dataset]
    labels = [construct_label(rec["decision"], rec["sufficiency_id"], LABEL2ID) for rec in test_dataset]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # For models with metadata support, use a custom approach
    if supports_metadata:
        # Extract metadata
        jurisdictions = [rec.get("jurisdiction", "Unspecified") for rec in test_dataset]
        insurance_types = [rec.get("insurance_type", "Unspecified") for rec in test_dataset]

        # Convert metadata to tensors
        j_ids = torch.tensor([JURISDICTION_MAP.get(j, JURISDICTION_MAP["Unspecified"]) for j in jurisdictions])
        i_ids = torch.tensor([INSURANCE_TYPE_MAP.get(i, INSURANCE_TYPE_MAP["Unspecified"]) for i in insurance_types])

        model.to(device)
        model.eval()

        # Process in batches
        batch_size = 100
        all_logits = []

        for i in range(0, len(text_records), batch_size):
            end_idx = min(i + batch_size, len(text_records))
            batch_texts = text_records[i:end_idx]
            batch_j_ids = j_ids[i:end_idx].to(device)
            batch_i_ids = i_ids[i:end_idx].to(device)

            inputs = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs, jurisdiction_id=batch_j_ids, insurance_type_id=batch_i_ids)

            all_logits.append(outputs["logits"].cpu())

        predictions = torch.cat(all_logits)

        # Now create predictions with unspecified metadata for all examples
        print("Computing metrics with unspecified metadata for all examples...")
        unspecified_j_ids = torch.full((len(text_records),), JURISDICTION_MAP["Unspecified"], dtype=torch.long)
        unspecified_i_ids = torch.full((len(text_records),), INSURANCE_TYPE_MAP["Unspecified"], dtype=torch.long)

        all_unspecified_logits = []

        for i in range(0, len(text_records), batch_size):
            end_idx = min(i + batch_size, len(text_records))
            batch_texts = text_records[i:end_idx]
            batch_j_ids = unspecified_j_ids[i:end_idx].to(device)
            batch_i_ids = unspecified_i_ids[i:end_idx].to(device)

            inputs = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs, jurisdiction_id=batch_j_ids, insurance_type_id=batch_i_ids)

            all_unspecified_logits.append(outputs["logits"].cpu())

        unspecified_predictions = torch.cat(all_unspecified_logits)
    else:
        # For standard models, use the original pipeline
        pipeline = ClassificationPipeline(model=model, tokenizer=tokenizer, device=device, truncation=True)
        predictions = torch.cat(pipeline(text_records, batch_size=100))
        # No metadata to ignore for standard models, so unspecified predictions are the same
        unspecified_predictions = predictions

    # Compute standard metrics
    metrics = compute_metrics(predictions, labels)
    metrics["param_count"] = param_count
    metrics["model_size_mb"] = size_in_mb

    # Add model size to metrics
    metrics["param_count"] = param_count

    # Compute metrics with unspecified metadata
    unspecified_metrics = compute_metrics(unspecified_predictions, labels)
    # Add model size to unspecified metrics
    unspecified_metrics["param_count"] = param_count

    # Compute threshold metrics if specified
    if threshold is not None:
        threshold_metrics = compute_metrics_w_threshold(predictions, labels, threshold)
        threshold_metrics["param_count"] = param_count
        threshold_metrics["model_size_mb"] = size_in_mb

        unspecified_threshold_metrics = compute_metrics_w_threshold(unspecified_predictions, labels, threshold)
        unspecified_threshold_metrics["param_count"] = param_count
        unspecified_threshold_metrics["model_size_mb"] = size_in_mb

    # Print and write metrics
    print("Standard metrics:")
    print(metrics)
    with open(os.path.join(ckpt_path, "test_metrics.json"), "w") as f:
        json.dump(metrics, f)

    print("\nMetrics with unspecified metadata:")
    print(unspecified_metrics)
    with open(os.path.join(ckpt_path, "test_metrics_unspecified.json"), "w") as f:
        json.dump(unspecified_metrics, f)

    if threshold is not None:
        print("\nStandard threshold metrics:")
        print(threshold_metrics)
        with open(os.path.join(ckpt_path, "test_metrics_w_threshold.json"), "w") as f:
            json.dump(threshold_metrics, f)

        print("\nThreshold metrics with unspecified metadata:")
        print(unspecified_threshold_metrics)
        with open(os.path.join(ckpt_path, "test_metrics_w_threshold_unspecified.json"), "w") as f:
            json.dump(unspecified_threshold_metrics, f)

    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="src/modeling/config/outcome_prediction/legalbert_small.yaml",
        help="Path to configuration file.",
        dest="config_path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    main(**args_dict)
