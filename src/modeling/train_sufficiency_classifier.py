import argparse
import datetime
import json
import os
from functools import partial

import numpy as np
import scipy
import torch
import wandb
from datasets import Dataset
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
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.modeling.util import load_config
from src.util import get_records_list
from src.modeling.data_augmentation import (
    process_and_save_augmentations
)

ID2LABEL = {0: "Insufficient", 1: "Sufficient"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
OUTPUT_DIR = "./models/sufficiency_predictor"


def construct_sufficiency_label(raw_label: int) -> int:
    """Construct binary 0: insufficient, 1: sufficient label from 1-5 rating."""
    if raw_label < 3:
        return 0
    elif raw_label >= 3:
        return 1
    
    else:
        raise ValueError(f"Invalid sufficiency score: {raw_label}. Must be between 1 and 5.")


def load(jsonl_path: str):
    recs = get_records_list(jsonl_path)
    updated_recs = [{"text": rec["answers"]["text"][0], "sufficiency_score": rec["sufficiency_score"]} for rec in recs]
    dataset = Dataset.from_list(updated_recs)
    return dataset


def split(dataset: Dataset, test_size: float = 0.2, seed: int = 3):
    dataset = dataset.class_encode_column("label")
    dataset = dataset.train_test_split(test_size=test_size, seed=seed, stratify_by_column="label")
    return dataset


def tokenize_batch(examples, tokenizer):
    tokenized = tokenizer(examples["text"], truncation=True, padding=True)
    examples["input_ids"] = tokenized["input_ids"]
    examples["attention_mask"] = tokenized["attention_mask"]
    return examples


def add_integral_ids_batch(examples):
    scores = examples["sufficiency_score"]
    examples["label"] = [construct_sufficiency_label(score) for score in scores]
    return examples


def compute_metrics(eval_pred):
    """Compute metrics during training."""
    predictions, labels = eval_pred
    pred_probs = scipy.special.softmax(predictions, axis=1)

    roc_auc = roc_auc_score(labels, pred_probs[:, 1])

    # Compute metrics for many thresholds
    thresholds = np.linspace(0, 1, 500)
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    # Retain metrics that maximize macro average f1
    best_recall = None
    best_precision = None
    best_f1 = None
    best_accuracy = None
    for t in thresholds:
        predictions = np.where(pred_probs[:, 0] > t, 0, 1)
        prec = precision_score(labels, predictions, zero_division=np.nan, average="macro")
        acc = accuracy_score(labels, predictions)
        rec = recall_score(labels, predictions, average="macro")
        f1 = f1_score(labels, predictions, average="macro")
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        if not best_f1 or f1 > best_f1:
            best_f1 = f1
            best_recall = rec
            best_precision = prec
            best_accuracy = acc

    return {
        "accuracy": best_accuracy,
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
        "roc_auc": roc_auc,
    }


def eval(model: torch.nn.Module, test_set: Dataset):
    """Eval best model after training, and save results and best threshold config with checkpoint."""
    labels = test_set["label"]

    model = model.to("cuda")

    logits = []
    for input_ids, attention_mask in zip(test_set["input_ids"], test_set["attention_mask"]):
        with torch.no_grad():
            output = model(
                input_ids=torch.tensor(input_ids).unsqueeze(0).to("cuda"),
                attention_mask=torch.tensor(attention_mask).unsqueeze(0).to("cuda"),
            )
            logits.append(output.logits)
    logits = torch.stack(logits, dim=0).squeeze(1)
    pred_probs = torch.softmax(logits, dim=-1).to("cpu")

    roc_auc = roc_auc_score(labels, pred_probs[:, 1])

    # Compute metrics for many thresholds
    thresholds = np.linspace(0, 1, 500)
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    # Retain metrics that maximize macro average f1
    best_recall = None
    best_precision = None
    best_f1 = None
    best_threshold = None
    best_accuracy = None
    for t in thresholds:
        predictions = torch.where(pred_probs[:, 0] > t, 0, 1)
        prec = precision_score(labels, predictions, zero_division=np.nan, average="macro")
        acc = accuracy_score(labels, predictions)
        rec = recall_score(labels, predictions, average="macro")
        f1 = f1_score(labels, predictions, average="macro")
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        if not best_f1 or f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_recall = rec
            best_precision = prec
            best_accuracy = acc

    return {
        "f1": best_f1,
        "recall": best_recall,
        "precision": best_precision,
        "accuracy": best_accuracy,
        "roc_auc": roc_auc,
        "best_threshold": best_threshold,
    }


class TrainerWithClassWeights(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        logits = outputs["logits"]
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = criterion(logits, labels)

        return (loss, outputs) if return_outputs else loss


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    # Load raw dataset
    dataset_name = cfg["dataset_name"]
    pretrained_model_key = cfg["pretrained_hf_classifier"]
    dataset_path = f"./data/annotated/{dataset_name}.jsonl"
    
    # Check if we should apply data augmentation
    use_data_augmentation = cfg.get("use_data_augmentation", False)
    save_augmentations = cfg.get("save_augmentations", False)
    
    if use_data_augmentation:
        
        generic_rewrite_params = cfg.get("generic_rewrite_params", {
            "num_augmentations_per_example": 1,
            "seed": 1,
            "api_key": os.environ.get("OPENAI_API_KEY")
        })
        
        unrelated_params = cfg.get("unrelated_params", {
            "num_examples": 500,
            "seed": 1,
            "api_key": os.environ.get("OPENAI_API_KEY")
        })
        
        sufficient_augmentation_params = cfg.get("sufficient_augmentation_params", {
            "num_augmentations_per_example": 1,
            "api_type": "llamacpp",
            "model_name": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
            "seed": 1,
            "api_key": os.environ.get("OPENAI_API_KEY")
        })
        
        # Process and get augmented dataset
        augmented_output_path = f"./data/augmented/{dataset_name}_augmented.jsonl" if save_augmentations else None
        train_dataset, test_dataset = process_and_save_augmentations(
            original_dataset_path=dataset_path,
            output_dataset_path=augmented_output_path,
            generic_rewrite_params=generic_rewrite_params,
            unrelated_params=unrelated_params,
            sufficient_augmentation_params=sufficient_augmentation_params,
            train_test_split_ratio=0.2,
            append_to_original=True,  # Include original examples
            seed=42
        )
        
        # Use the datasets returned from the augmentation process
        dataset = {"train": train_dataset, "test": test_dataset}

    else:
        # Normal loading without augmentation
        dataset = load(dataset_path)
        dataset = split(dataset)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_key, model_max_length=512, truncation=True)

    # Prepare dataset for training
    dataset["train"] = dataset["train"].map(partial(tokenize_batch, tokenizer=tokenizer), batched=True)
    dataset["train"] = dataset["train"].map(add_integral_ids_batch, batched=True)
    
    dataset["test"] = dataset["test"].map(partial(tokenize_batch, tokenizer=tokenizer), batched=True)
    dataset["test"] = dataset["test"].map(add_integral_ids_batch, batched=True)
    
    train_classes = dataset["train"]["label"]
    class_weights = [
        len([c for c in train_classes if c == 1]) / len(train_classes),
        len([c for c in train_classes if c == 0]) / len(train_classes),
    ]

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_key, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    checkpoints_dir = os.path.join(OUTPUT_DIR, dataset_name, pretrained_model_key)

    # Optional wandb config
    report_to = "none"
    if cfg["wandb_logging"]:
        wandb_project = cfg["wandb_project"]
        run_tag = cfg["wandb_run_tag"]
        wandb_experiment_name = "-".join([run_tag, datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%p")])

        wandb.init(project=wandb_project, name=wandb_experiment_name, config=cfg)
        report_to = "wandb"

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["num_epochs"],
        weight_decay=cfg["weight_decay"],
        fp16=(cfg["dtype"] == "float16"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=report_to,
    )

    trainer = TrainerWithClassWeights(
        class_weights=torch.tensor(class_weights),
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Get best checkpoint dir
    checkpoint_dirs = sorted(os.listdir(checkpoints_dir))
    checkpoint_name = checkpoint_dirs[0]
    ckpt_dir = os.path.join(checkpoints_dir, checkpoint_name)

    # Save tokenizer
    tokenizer_path = os.path.join(ckpt_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)


    # Load best model
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)

    # Save eval results on test set for best model, and threshold maximizing macro F1 on test set.
    # Need to carefully tune here to get something of sufficient quality, because training data is tiny, and imperfect.
    eval_results = eval(model, dataset["test"])
    with open(os.path.join(ckpt_dir, "eval_results.json"), "w") as f:
        f.write(json.dumps(eval_results))
        
    # Print summary of augmentation if used
    if use_data_augmentation:
        print(f"Training data size before augmentation: {len(dataset['train']) - (len(dataset['test']) * 4)}")  # Approximation
        print(f"Training data size after augmentation: {len(dataset['train'])}")
        print(f"Test data size: {len(dataset['test'])}")
        
        # Print class distribution
        train_sufficiency = [construct_sufficiency_label(score) for score in dataset["train"]["sufficiency_score"]]
        test_sufficiency = [construct_sufficiency_label(score) for score in dataset["test"]["sufficiency_score"]]
        
        print("Training data class distribution:")
        print(f"  Insufficient (0): {train_sufficiency.count(0)}")
        print(f"  Sufficient (1): {train_sufficiency.count(1)}")
        
        print("Test data class distribution:")
        print(f"  Insufficient (0): {test_sufficiency.count(0)}")
        print(f"  Sufficient (1): {test_sufficiency.count(1)}")

    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="src/modeling/config/sufficiency_classification/default.yaml",
        help="Path to configuration file.",
        dest="config_path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    main(**args_dict)
