import os
import typing as t
from contextlib import nullcontext

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModel, AutoTokenizer

from src.modeling.util import export_onnx_model, quantize_onnx_model
from src.util import batcher, get_records_list

ID2LABEL = {0: "Upheld", 1: "Overturned"}
OUTPUT_DIR = "./models/overturn_predictor"
MODEL_PATH = "intfloat/e5-base-v2"


def deduce_label2id(labels: list[str]) -> dict:
    label2id = {}
    for label in labels:
        if "overturn" in label.lower():
            label2id[label] = 1
        else:
            label2id[label] = 0
    return label2id


def compute_metrics(eval_pred):
    acc = evaluate.load("accuracy")
    prec = evaluate.load("precision")
    rec = evaluate.load("recall")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = acc.compute(predictions=predictions, references=labels)
    prec = prec.compute(predictions=predictions, references=labels)
    rec = rec.compute(predictions=predictions, references=labels)
    return {"accuracy": acc, "precision": prec, "recall": rec}


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class OverturnModel(torch.nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.embedding_backbone = AutoModel.from_pretrained(MODEL_PATH)
        self.embedding_dim = self.embedding_backbone.config.hidden_size
        self.fc = torch.nn.Linear(self.embedding_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.embedding_backbone(input_ids, attention_mask)
        model_outputs = outputs.last_hidden_state
        embeddings = average_pool(model_outputs, attention_mask)

        # Pool and Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Class logits
        logits = self.fc(embeddings)

        return logits


def loss(logit_batch: torch.Tensor, targets: torch.Tensor):
    # TODO: Incorporate class weights to construct a stratified/balanced loss.
    # loss = F.binary_cross_entropy(F.softmax(logit_batch, dim=-1)[:, 1], targets)
    loss = F.binary_cross_entropy_with_logits(logit_batch[:, 1], targets)
    return loss


def train_epoch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    loss_fn: t.Callable,
    optimizer: torch.optim.Optimizer,
    records: list[dict],
    batch_size: int,
    device: str,
    fp16: bool,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    """Logic for a single epoch training pass of model, using records."""
    epoch_losses = []
    # Perform single epoch training loop
    forward_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16) if (device == "cuda" and fp16) else nullcontext()
    )

    batch_num = 1
    num_batches = len(records) // batch_size
    for batch in batcher(records, batch_size):
        if batch_num % 100 == 0:
            print(f"Batch {batch_num}/{num_batches}")

        # Prepare records
        text_inputs = ["query: " + rec["text"] for rec in batch]  # model annoyance with "query" prefix
        labels = (
            torch.tensor([rec["decision_label"] for rec in batch])
            .to(torch.float32)
            .pin_memory()
            .to(device, non_blocking=True)
        )

        # Tokenize batch
        tokenized_inputs = tokenizer(
            text_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.embedding_backbone.config.max_position_embeddings,
        )
        input_ids = tokenized_inputs["input_ids"].pin_memory().to(device, non_blocking=True)
        attention_mask = tokenized_inputs["attention_mask"].pin_memory().to(device, non_blocking=True)

        # Clear optimizer grad state
        optimizer.zero_grad()

        # Forward
        with forward_context:
            logit_batch = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Loss
            batch_loss = loss_fn(logit_batch, labels)

        # Backward
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Log loss to stdout
        float_loss = batch_loss.item()
        epoch_losses.append(float_loss)

        batch_num += 1

    return sum(epoch_losses) / len(epoch_losses)


def batch_forward(
    records: list[dict],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: str = "cpu",
    batch_size: int = 100,
    output_probs: bool = True,
) -> torch.Tensor:
    """Forward a model (without storing gradient graph) on a collection of records, via batched inference calls. Return single matrix of normalized logits."""
    model = model.to(device)

    # Calculate model outputs on records via batched inference
    out_batches = []
    for _idx, batch in enumerate(batcher(records, batch_size=batch_size)):
        # Prepare batch of preprocessed records
        text_inputs = ["query: " + rec["text"] for rec in batch]  # model annoyance with "query" prefix

        # Tokenize batch
        tokenized_inputs = tokenizer(
            text_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.embedding_backbone.config.max_position_embeddings,
        )
        input_ids = tokenized_inputs["input_ids"].to(device)
        attention_mask = tokenized_inputs["attention_mask"].to(device)

        with torch.no_grad():
            # Forward
            logit_batch = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        if output_probs:
            prob_batch = F.softmax(logit_batch, dim=-1).cpu()
            out_batches.append(prob_batch)
        else:
            out_batches.append(logit_batch)

    outs = torch.cat(out_batches, dim=0)

    return outs


def get_binary_labels(probs: torch.Tensor) -> np.ndarray:
    """Given a tensor of class_logits of shape (batch_size, 2), return the
    associated class labels of shape (batch_size, ) corresponding to argmax.
    """
    # TODO: allow for selection via threshold on prob, rather than argmax (e.g. .5 threshold).
    return probs.argmax(1).cpu().numpy()


def evaluate_model_probs(probs: torch.Tensor, gt: np.ndarray, verbose: bool = False) -> dict:
    """Compute evaluation metric suite given tensor of class probs (0-1) array of labels."""

    # Metrics computed for best thresholds
    preds = get_binary_labels(probs)

    # Overall accuracy (exactly correct multi-label assignment)
    total_acc = accuracy_score(gt, preds)

    # Class precisions
    class_precisions = precision_score(gt, preds, average=None, zero_division=0)

    # Class recalls
    class_recalls = recall_score(gt, preds, average=None)

    # F1 on a per-class basis
    class_f1s = f1_score(gt, preds, average=None, zero_division=np.nan)

    # Confusion Matrix
    conf_mat = confusion_matrix(gt, preds)

    if verbose:
        print("Eval Metrics:")
        print(f"\tTotal accuracy: {total_acc}")
        print(f"\tClass Precisions: {class_precisions}")
        print(f"\tClass Recalls: {class_recalls}")
        print(f"\tClass F1s: {class_f1s}")
        print(f"\tConf mat: {conf_mat}")

    return {
        "accuracy": total_acc,
        "precisions": class_precisions,
        "recalls": class_recalls,
        "f1s": class_f1s,
        "confusion_matrix": conf_mat,
    }


def eval_epoch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    records: list[dict],
    device: str,
    loss_fn: t.Callable,
    verbose: bool = False,
):
    logits = batch_forward(
        records,
        model,
        tokenizer,
        device=device,
        batch_size=16,
        output_probs=False,
    )

    # Get true labels for records
    labels = torch.tensor([rec["decision_label"] for rec in records]).to(torch.float32).to(device)
    labels_numpy = np.array([rec["decision_label"] for rec in records])

    # Compute loss
    with torch.no_grad():
        total_loss = loss_fn(logits, labels).item()
        probs = F.softmax(logits, dim=1)
        eval_metrics = evaluate_model_probs(probs, labels_numpy, verbose)

    return total_loss, eval_metrics


def train(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: str,
    epochs: int,
    batch_size: int,
    train_records: list[dict],
    val_records: list[dict],
    loss_fn: t.Callable,
    output_dir: str,
    fp16: bool = False,
):
    """A complete but minimal training loop."""

    # Instantiate model
    model = model.to(device)

    # Instantiate optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=2e-6)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=6)
    loss_fn = loss_fn

    # Config
    print_freq = 1  # in epochs
    compiled_model = torch.compile(model)

    # Train loop
    best_eval_loss = None
    for epoch in range(epochs):
        print(f"Starting epoch: {epoch}")
        compiled_model.train()

        scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device == "cuda"))

        _avg_loss = train_epoch(
            compiled_model,
            tokenizer,
            loss_fn,
            optimizer,
            train_records,
            batch_size,
            device,
            fp16,
            scaler,
        )
        if epoch % print_freq == 0:
            print(f"Train loss: {_avg_loss}")

        # Eval on entire val dataset once per epoch
        if epoch % print_freq == 0:
            verbose_eval = True
        else:
            verbose_eval = False
        compiled_model.eval()
        avg_eval_loss, eval_metrics = eval_epoch(
            compiled_model,
            tokenizer,
            val_records,
            device,
            loss_fn,
            verbose=verbose_eval,
        )
        if epoch % print_freq == 0:
            print(f"Eval loss: {avg_eval_loss}\n\n")

        # Save model + config + metrics, if model is new 'best' per eval loss
        if not best_eval_loss or avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            scheduler.step(best_eval_loss)
            ckpt_path = os.path.join(output_dir, "model.ckpt")
            model_cfg = {"context_length": model.embedding_backbone.config.max_position_embeddings}
            out_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "eval_loss": best_eval_loss,
                "model_config": model_cfg,
            }
            out_dict.update(eval_metrics)
            torch.save(
                out_dict,
                ckpt_path,
            )

            # Save tokenizer
            tokenizer_path = os.path.join(output_dir, "tokenizer")
            tokenizer.save_pretrained(tokenizer_path)

            # Save ONNX binaries
            model.to("cpu")
            export_onnx_model(os.path.join(output_dir, "model.onnx"), model, tokenizer)
            quantize_onnx_model(
                os.path.join(output_dir, "model.onnx"),
                os.path.join(output_dir, "quant-model.onnx"),
            )
            model.to(device)

    return None


if __name__ == "__main__":
    # Load raw dataset
    device = "cuda"
    dataset_name = "train_backgrounds-agg"
    pretrained_model_key = MODEL_PATH
    dataset_path = f"./data/outcomes/{dataset_name}.jsonl"
    output_dir = os.path.join(OUTPUT_DIR, dataset_name, pretrained_model_key)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    full_dataset = get_records_list(dataset_path)

    # Instantiate label mapping
    labels = set([rec["decision"] for rec in full_dataset])
    label2id = deduce_label2id(labels)

    # Prepare dataset for training w/ integral IDs
    for rec in full_dataset:
        rec["decision_label"] = label2id[rec["decision"]]

    # Split train/test
    train_data, val_data = train_test_split(
        full_dataset,
        test_size=0.2,
        random_state=1,
        shuffle=True,
        stratify=[rec["decision_label"] for rec in full_dataset],
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = OverturnModel()

    # Implement minimal training loop
    batch_size = 8
    num_epochs = 14
    fp16 = True

    train(
        model,
        tokenizer,
        device,
        num_epochs,
        batch_size,
        train_data,
        val_data,
        loss,
        output_dir,
        fp16,
    )
