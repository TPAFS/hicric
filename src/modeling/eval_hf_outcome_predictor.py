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

from src.modeling.util import load_config
from src.util import get_records_list

MODEL_DIR = "./models/overturn_predictor"
ID2LABEL = {0: "Insufficient", 1: "Upheld", 2: "Overturned"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


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

    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_key, model_max_length=512)
    checkpoints_dir = os.path.join(MODEL_DIR, "train_backgrounds_suff", base_model_name)
    print("Evaluating from checkpoint: ", checkpoint_name)
    ckpt_path = os.path.join(checkpoints_dir, checkpoint_name)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_path, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    # Isolate records
    text_records = [rec["text"] for rec in test_dataset]
    labels = [construct_label(rec["decision"], rec["sufficiency_id"], LABEL2ID) for rec in test_dataset]

    device = "cuda"
    # pipeline = ClassificationPipeline(model=model, tokenizer=tokenizer, device=device)
    pipeline = ClassificationPipeline(model=model, tokenizer=tokenizer, device=device, truncation=True)

    predictions = torch.cat(pipeline(text_records, batch_size=100))

    if threshold is not None:
        threshold_metrics = compute_metrics_w_threshold(predictions, labels, threshold)

    metrics = compute_metrics(predictions, labels)

    # Print and write metrics
    print(metrics)
    with open(os.path.join(ckpt_path, "test_metrics.json"), "w") as f:
        json.dump(metrics, f)
    if threshold is not None:
        print(threshold_metrics)
        with open(os.path.join(ckpt_path, "test_metrics_w_threshold.json"), "w") as f:
            json.dump(threshold_metrics, f)

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
