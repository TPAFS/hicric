# Token Classification Adapted for multi-span selection
import argparse
import datetime
import os
from functools import partial

import evaluate
import numpy as np
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from src.modeling.util import load_config

NO_CLASS_ENCODING = 0
BACKGROUND_CLASS_ENCODING = 1
# Label choice due to annoyance with seqeval
# TODO: compute metrics ourselves
LABEL_LIST = ["O", "B-ackground"]


def load_and_split(jsonl_path: str, test_size: float = 0.2, seed: int = 2):
    dataset = load_dataset("json", data_files=jsonl_path)["train"]
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    return dataset


def add_char_labels(
    examples: list,
    no_class_encoding: int = NO_CLASS_ENCODING,
    background_class_encoding: int = BACKGROUND_CLASS_ENCODING,
) -> list:
    """Construct char-level classification label from substring spans.

    0 for not background, 1 for background"""
    context = examples["context"]
    answers = examples["answers"]
    labels = [no_class_encoding] * len(context)
    for answer_idx, answer_start in enumerate(answers["answer_start"]):
        labels[answer_start : answer_start + len(answers["text"][answer_idx])] = [background_class_encoding] * len(
            answers["text"][answer_idx]
        )

    examples["char_labels"] = labels
    return examples


def tokenize_and_align(examples, tokenizer):
    tokenized = tokenizer(examples["context"], truncation=True, return_offsets_mapping=True)

    aligned_labels = []
    for idx, offset in enumerate(tokenized.offset_mapping):
        char_start = offset[0]
        char_end = offset[1]
        if char_start > len(examples["char_labels"]) - 1:
            continue
        if char_start == char_end == 0:
            aligned_labels.append(-100)  # Special label for special tokens, will tell pytorch to ignore in loss
        else:
            aligned_labels.append(examples["char_labels"][char_start])

    tokenized["labels"] = aligned_labels
    return tokenized


def compute_metrics(
    eval_preds: EvalPrediction, label_list: list[int], seqeval: evaluate.EvaluationModule
) -> dict[str, float]:
    logit_batch, label_batch = eval_preds
    prediction_batch = np.argmax(logit_batch, axis=-1)  # use argmax to start, tune based on threshold if insufficient

    true_predictions = [
        [label_list[p] for (p, token_label) in zip(prediction, label) if token_label != -100]
        for prediction, label in zip(prediction_batch, label_batch)
    ]
    true_labels = [
        [label_list[token_label] for (p, token_label) in zip(prediction, label) if token_label != -100]
        for prediction, label in zip(prediction_batch, label_batch)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    top_dir = os.getcwd()  # assuming script is being called from top level of repo
    pretrained_model_path = cfg["pretrained_hf_classifier"]
    dataset_name = cfg["dataset_name"]
    dataset_path = f"./data/annotated/{dataset_name}.jsonl"
    output_dir = os.path.join(top_dir, f"./models/background_span/{dataset_name}/{pretrained_model_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    id2label = {idx: LABEL_LIST[idx] for idx in range(2)}
    label2id = {value: key for key, value in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_path,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id,
    )

    # Load and process squad format data
    qa = load_and_split(dataset_path)
    qa_w_char_labels = qa.map(add_char_labels, batched=False)
    processed = qa_w_char_labels.map(partial(tokenize_and_align, tokenizer=tokenizer), batched=False)

    # Train config and loop
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")

    # Optional wandb config
    report_to = "none"
    if cfg["wandb_logging"]:
        wandb_project = cfg["wandb_project"]
        run_tag = cfg["wandb_run_tag"]
        wandb_experiment_name = "-".join([run_tag, datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%p")])

        wandb.init(project=wandb_project, name=wandb_experiment_name, config=cfg)
        report_to = "wandb"

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["num_epochs"],
        weight_decay=cfg["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=1,
        report_to=report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, label_list=LABEL_LIST, seqeval=seqeval),
    )

    trainer.train()

    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="src/modeling/config/background_token_classification/default.yaml",
        help="Path to configuration file.",
        dest="config_path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    main(**args_dict)
