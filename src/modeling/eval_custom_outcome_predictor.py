import functools
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer

from src.modeling.train_custom_outcome_predictor import OverturnModel
from src.util import get_records_list

MODEL_DIR = "./models/overturn_predictor"
ID2LABEL = {0: "Upheld", 1: "Overturned"}
label2id = {value: key for key, value in ID2LABEL.items()}


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average="binary", pos_label=1)
    rec = recall_score(labels, predictions, average="binary", pos_label=1)
    f1 = f1_score(labels, predictions, average="binary", pos_label=1, zero_division=np.nan)
    return {
        "accuracy": acc,
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "f1": f1.tolist(),
    }


def batcher(records: list[dict], batch_size: int = 100):
    """Dataset batcher for list of json records."""
    for start_idx in range(0, len(records), batch_size):
        yield records[start_idx : start_idx + batch_size]


@functools.cache
def load_model(model_ckpt_path: str, device: str, quantized: bool = False):
    """Return model"""
    print("Loading model...")

    # Instantiate model
    checkpoint = torch.load(model_ckpt_path, map_location=torch.device(device))
    model = OverturnModel().to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Quantize for performance
    if quantized:
        model = torch.ao.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8,
        )
    print("Done loading model.\n")
    print(
        f"Model stats:\n\tCheckpoint size (MB): {round(os.path.getsize(model_ckpt_path) / (1024 * 1024))}\n\tNum Params: {sum([p.numel() for p in model.parameters()])}"
    )
    return model


if __name__ == "__main__":
    # Load model and tokenizer
    device = "cuda"
    MODEL_DIR = "./models/overturn_predictor/train_backgrounds-agg/intfloat/e5-base-v2/"
    model_ckpt_name = "model.ckpt"
    model_path = os.path.join(MODEL_DIR, model_ckpt_name)
    model = load_model(model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODEL_DIR, "tokenizer"),
        model_max_length=model.embedding_backbone.config.max_position_embeddings,
    )
    special_prompt_prepend = "query: "

    # Load dataset
    test_dataset = get_records_list("./data/outcomes/test_backgrounds-agg.jsonl")
    text_records = [special_prompt_prepend + rec["text"] for rec in test_dataset]
    labels = [label2id[rec["decision"]] for rec in test_dataset]

    device = "cuda"
    predictions = []
    for batch_texts in batcher(text_records):
        tokenized = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            result = model(
                tokenized["input_ids"].to(device),
                tokenized["attention_mask"].to(device),
            )
        probs = torch.softmax(result, dim=-1)
        prob, classes = torch.max(probs, dim=-1)
        predictions.extend(classes.cpu().tolist())

    metrics = compute_metrics(predictions, labels)

    # Print and write metrics
    print(metrics)
    with open(os.path.join(MODEL_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f)
