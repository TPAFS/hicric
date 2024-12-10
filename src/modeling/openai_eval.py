import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.util import get_records_list

MODEL_KEY = "gpt-4o-mini-2024-07-18"


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
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
    }


def merge_pred_gt(response_path: str, answers_path: str) -> dict:
    merged = {}
    gt = get_records_list(answers_path)
    responses = get_records_list(response_path)

    for rec in gt:
        merged[rec["custom_id"]] = {"answer": rec["decision"]}

    for rec in responses:
        resp = rec["response"]["body"]["choices"][0]["message"]["content"]
        try:
            pred = json.loads(resp.replace(".", "0."))
        except json.JSONDecodeError:
            print(rec["custom_id"])
            print(type(resp))
            print(resp)
            pred = None
        merged[rec["custom_id"]]["pred"] = pred

    return merged


def eval_preds() -> dict:
    pass


if __name__ == "__main__":
    outcomes_dataset = "test_backgrounds_suff"
    answers_path = f"./data/provider_annotated_outcomes/openai/{outcomes_dataset}/hicric_eval_request_answers.jsonl"
    response_path = (
        f"./data/provider_annotated_outcomes/openai/{outcomes_dataset}/hicric_eval_response_{MODEL_KEY}.jsonl"
    )
    merged = merge_pred_gt(response_path, answers_path)

    out_map = {"Insufficient": 0, "Upheld": 1, "Overturned": 2}

    preds = []
    labels = []
    # Favorable evaluation: don't count non-answers against GPT, to be generous
    for id, rec in merged.items():
        print(rec)
        if rec["pred"] is None:
            continue
        if rec["pred"].get("decision") is None:
            continue
        if rec["pred"].get("decision") not in out_map.keys():
            continue
        else:
            preds.append(out_map[rec["pred"]["decision"]])
            labels.append(out_map[rec["answer"]])
    if len(preds) < len(merged):
        print(f"Model failed to produce valid json for {len(merged) - len(preds)} values.")
    metrics = compute_metrics(preds, labels)
    print(metrics)
    with open(
        os.path.join(f"./data/provider_annotated_outcomes/openai/{outcomes_dataset}", f"{MODEL_KEY}_test_metrics.json"),
        "w",
    ) as f:
        json.dump(metrics, f)
