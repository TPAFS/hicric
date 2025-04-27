import argparse
import datetime
import os
from functools import partial

import numpy as np
import scipy
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

import wandb
from src.modeling.util import export_onnx_model, load_config, quantize_onnx_model
from src.util import get_records_list

ID2LABEL = {0: "Insufficient", 1: "Upheld", 2: "Overturned"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Define mappings for jurisdiction and insurance type
JURISDICTION_MAP = {"NY": 0, "CA": 1, "Unspecified": 2}
INSURANCE_TYPE_MAP = {"Commercial": 0, "Medicaid": 1, "Unspecified": 2}

OUTPUT_DIR = "./models/overturn_predictor"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_and_split(
    jsonl_path: str,
    test_size: float = 0.2,
    filter_keys: list = ["decision", "text", "sufficiency_id", "jurisdiction", "insurance_type"],
    seed: int = 2,
) -> Dataset:
    if len(filter_keys) > 0:
        recs = get_records_list(jsonl_path)
        recs = [{key: rec.get(key, "Unspecified") for key in filter_keys} for rec in recs]
        dataset = Dataset.from_list(recs)

    else:
        dataset = load_dataset("json", data_files=jsonl_path)["train"]

    # Use scikit-learn train_test_split for stratified sampling
    train_indices, test_indices = train_test_split(
        list(range(len(dataset))), test_size=test_size, random_state=seed, shuffle=True, stratify=dataset["decision"]
    )

    # Create train and test datasets using the indices
    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)

    # Return as a DatasetDict
    return DatasetDict({"train": train_dataset, "test": test_dataset})


def tokenize_batch(examples, tokenizer):
    tokenized = tokenizer(examples["text"], truncation=True, padding=True, return_tensors="pt")
    examples["input_ids"] = tokenized["input_ids"]
    examples["attention_mask"] = tokenized["attention_mask"]
    return examples


def construct_label(outcome: str, sufficiency_id: int, label2id: dict) -> int:
    """Construct 3-class label from outcome, if sufficient background."""
    if sufficiency_id == 0:
        return 0
    else:
        return label2id[outcome]


def add_integral_ids_batch(examples, label2id: dict, jurisdiction_map: dict, insurance_type_map: dict):
    outcomes = examples["decision"]
    sufficiency_ids = examples["sufficiency_id"]

    # Map jurisdiction and insurance_type to their respective IDs
    jurisdictions = examples.get("jurisdiction", ["Unspecified"] * len(outcomes))
    insurance_types = examples.get("insurance_type", ["Unspecified"] * len(outcomes))

    examples["label"] = [construct_label(outcome, id, label2id) for (outcome, id) in zip(outcomes, sufficiency_ids)]
    examples["jurisdiction_id"] = [jurisdiction_map.get(j, jurisdiction_map["Unspecified"]) for j in jurisdictions]
    examples["insurance_type_id"] = [
        insurance_type_map.get(i, insurance_type_map["Unspecified"]) for i in insurance_types
    ]

    return examples


def compute_metrics(eval_pred) -> dict:
    predictions, labels = eval_pred

    softmax_preds = scipy.special.softmax(predictions, axis=-1)

    roc_auc = roc_auc_score(labels, softmax_preds, multi_class="ovr")

    predictions = np.argmax(predictions, axis=1)

    acc = accuracy_score(labels, predictions)

    # Macro
    macro_prec = precision_score(labels, predictions, average="macro")
    macro_rec = recall_score(labels, predictions, average="macro")
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=np.nan)

    # Class-level
    prec = precision_score(labels, predictions, average=None)
    rec = recall_score(labels, predictions, average=None)
    f1 = f1_score(labels, predictions, average=None, zero_division=np.nan)

    return {
        "accuracy": acc,
        "macro_precision": macro_prec.tolist(),
        "macro_recall": macro_rec.tolist(),
        "macro_f1": macro_f1.tolist(),
        "class_precision": prec.tolist(),
        "class_recall": rec.tolist(),
        "class_f1": f1.tolist(),
        "roc_auc": roc_auc,
    }


def compute_metrics2(eval_pred) -> dict:
    predictions, labels = eval_pred

    softmax_preds = scipy.special.softmax(predictions, axis=-1)

    roc_auc = roc_auc_score(labels, softmax_preds, multi_class="ovr", average=None)
    roc_auc_average = roc_auc_score(labels, softmax_preds, multi_class="ovr")

    c0_thresholds = np.arange(0.05, 1.0, 0.05)

    best_macro_f1 = 0
    best_metrics = {}

    # Iterate over each threshold to evaluate performance
    for c0_threshold in c0_thresholds:
        pred_labels = np.full_like(labels, -1)  # Initialize with -1 for unassigned classes

        # Assign class 0 where the probability meets or exceeds the threshold
        class_0_mask = softmax_preds[:, 0] >= c0_threshold
        pred_labels[class_0_mask] = 0

        # If probability does not exceed threshold, take argmax of upheld/overturned class otpions
        remaining_mask = pred_labels == -1
        remaining_preds = softmax_preds[remaining_mask, 1:]  # Exclude class 0
        pred_labels[remaining_mask] = np.argmax(remaining_preds, axis=1) + 1  # + 1 to adjust for class index shift

        # Compute metrics
        acc = accuracy_score(labels, pred_labels)
        macro_prec = precision_score(labels, pred_labels, average="macro", zero_division=0)
        macro_rec = recall_score(labels, pred_labels, average="macro", zero_division=0)
        macro_f1 = f1_score(labels, pred_labels, average="macro", zero_division=0)

        # If this threshold results in a better macro F1 score, store its results
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1

            prec = precision_score(labels, pred_labels, average=None, zero_division=0)
            rec = recall_score(labels, pred_labels, average=None, zero_division=0)
            f1 = f1_score(labels, pred_labels, average=None, zero_division=0)

            best_metrics = {
                "accuracy": acc,
                "macro_precision": macro_prec,
                "macro_recall": macro_rec,
                "macro_f1": macro_f1,
                "class_precision": prec.tolist(),
                "class_recall": rec.tolist(),
                "class_f1": f1.tolist(),
                "roc_aucs_ovr": roc_auc.tolist(),
                "roc_auc_average": roc_auc_average,
                "best_c0_threshold": c0_threshold,
            }

    return best_metrics


class TextClassificationWithMetadata(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size

        # Metadata embeddings
        self.jurisdiction_embeddings = nn.Embedding(2, self.hidden_size // 4)
        self.insurance_type_embeddings = nn.Embedding(2, self.hidden_size // 4)

        # Fusion projection
        self.metadata_projection = nn.Sequential(
            nn.Linear(self.hidden_size // 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Manual cross-attention components
        self.num_heads = 8
        self.head_dim = self.hidden_size // self.num_heads

        # Query, Key, Value projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(0.1)

        self.metadata_norm = nn.LayerNorm(self.hidden_size)
        self.sequence_norm = nn.LayerNorm(self.hidden_size)

        # Save the original num_labels
        self.num_labels = (
            self.classifier.out_features if hasattr(self.classifier, "out_features") else config.num_labels
        )

        # Now replace the classifier
        self.enhanced_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.num_labels),
        )

    def manual_cross_attention(self, query, key, value, key_padding_mask=None):
        # Same implementation, but with explicit cleanup
        batch_size = query.size(1)
        seq_length = key.size(0)

        # Project query, key, value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(1, batch_size, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        k = k.view(seq_length, batch_size, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        v = v.view(seq_length, batch_size, self.num_heads, self.head_dim).permute(2, 1, 0, 3)

        # Calculate attention scores
        scaling = float(self.head_dim) ** -0.5
        q = q * scaling
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # Apply attention mask if provided
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.transpose(0, 1)
            attn_mask = attn_mask.expand(self.num_heads, -1, 1, -1)
            attn_scores = attn_scores.masked_fill(attn_mask, -10000.0)

        # Get max values for numerical stability
        attn_scores_max, _ = torch.max(attn_scores, dim=-1, keepdim=True)
        attn_scores = attn_scores - attn_scores_max

        # Apply softmax
        attn_weights = torch.exp(torch.clamp(attn_scores, min=-10000.0, max=100.0))
        attn_sum = torch.sum(attn_weights, dim=-1, keepdim=True) + 1e-6
        attn_weights = attn_weights / attn_sum

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Clean up intermediates
        del attn_scores, attn_scores_max, attn_sum
        if key_padding_mask is not None:
            del attn_mask

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)

        # Clean up more intermediates
        del attn_weights

        # Reshape back
        context = context.permute(2, 1, 0, 3)
        context = context.reshape(1, batch_size, self.hidden_size)

        # Clean up
        del q, k, v

        # Apply output projection
        result = self.out_proj(context)

        # Final cleanup
        del context

        return result

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        jurisdiction_id=None,
        insurance_type_id=None,
        labels=None,
        return_dict=None,
        token_type_ids=None,
    ):
        # Validate that required parameters are provided, use index 2 for optional
        if jurisdiction_id is None or insurance_type_id is None:
            raise ValueError("jurisdiction_id and insurance_type_id must be provided")

        # Process metadata
        j_unspecified_mask = jurisdiction_id == 2
        i_unspecified_mask = insurance_type_id == 2

        # Temporarily assign unspecified metadata classes -> 1
        j_ids_valid = torch.clamp(jurisdiction_id, 0, 1)
        i_ids_valid = torch.clamp(insurance_type_id, 0, 1)

        j_embeddings = self.jurisdiction_embeddings(j_ids_valid)
        i_embeddings = self.insurance_type_embeddings(i_ids_valid)

        avg_j_embedding = self.jurisdiction_embeddings.weight.mean(dim=0)
        avg_i_embedding = self.insurance_type_embeddings.weight.mean(dim=0)

        # Assign unspecified classes the average embeddings
        j_embeddings = torch.where(
            j_unspecified_mask.unsqueeze(-1).expand_as(j_embeddings),
            avg_j_embedding.expand_as(j_embeddings),
            j_embeddings,
        )
        i_embeddings = torch.where(
            i_unspecified_mask.unsqueeze(-1).expand_as(i_embeddings),
            avg_i_embedding.expand_as(i_embeddings),
            i_embeddings,
        )

        # Combine metadata and project
        combined_metadata = torch.cat([j_embeddings, i_embeddings], dim=1)
        metadata_features = self.metadata_projection(combined_metadata)

        # Call the parent model but bypass its classification head
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        base_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # We need the hidden states
            return_dict=True,
            token_type_ids=token_type_ids,
            labels=None,  # Don't pass labels yet
        )

        sequence_output = base_outputs.hidden_states[-1]

        # Use cross-attention for fusion
        metadata_features = self.metadata_norm(metadata_features)
        sequence_output = self.sequence_norm(sequence_output)

        # Reshape for attention
        metadata_features = metadata_features.unsqueeze(0)  # [1, batch, hidden_dim]
        sequence_output_t = sequence_output.transpose(0, 1)  # [seq_len, batch, hidden_dim]

        key_padding_mask = attention_mask == 0

        fused_features = self.manual_cross_attention(
            query=metadata_features, key=sequence_output_t, value=sequence_output_t, key_padding_mask=key_padding_mask
        )

        # Combine with CLS token
        fused_features = fused_features.squeeze(0) + sequence_output[:, 0]

        logits = self.enhanced_classifier(fused_features)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if return_dict:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=base_outputs.hidden_states,
                attentions=base_outputs.attentions,
            )
        else:
            output = (logits,) + base_outputs[2:]
            return ((loss,) + output) if loss is not None else output


class DataCollatorWithMetadata(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)

        # Add the jurisdiction and insurance type IDs to the batch
        if "jurisdiction_id" in features[0]:
            batch["jurisdiction_id"] = torch.tensor([f["jurisdiction_id"] for f in features])

        if "insurance_type_id" in features[0]:
            batch["insurance_type_id"] = torch.tensor([f["insurance_type_id"] for f in features])

        return batch


class CPUEvalTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # Move logits to CPU immediately to save GPU memory
            logits = logits.detach().cpu()

            labels = None
            if has_labels:
                labels = tuple(inputs.get(name).detach().cpu() for name in self.label_names)
                if len(labels) == 1:
                    labels = labels[0]

            loss = None
            if has_labels and outputs.loss is not None:
                loss = outputs.loss.detach().cpu()

        return (loss, logits, labels)


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    # Are we finetuning a model pretrained on hicric, or off the shelf-pretrained
    hicric_pretrained = cfg["hicric_pretrained"]

    # Define key for HF AutoModel, define output checkpoint naming accordingly
    base_model_name = cfg["base_model_name"]
    if hicric_pretrained:
        pretrained_model_key = cfg["pretrained_model_dir"]
        outdir_name = f"{base_model_name}-hicric-pretrained"
    else:
        pretrained_model_key = cfg["pretrained_hf_model_key"]
        outdir_name = base_model_name

    # Load raw dataset
    dataset_path = cfg["train_data_path"]
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # Optional wandb config
    report_to = "none"
    if cfg["wandb_logging"]:
        wandb_project = cfg["wandb_project"]
        run_tag = cfg["wandb_run_tag"] + f"_{outdir_name}"
        wandb_experiment_name = "-".join([run_tag, datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%p")])

        wandb.init(project=wandb_project, name=wandb_experiment_name, config=cfg)
        report_to = "wandb"

    # Load raw training dataset, split into train/train-time-eval
    dataset = load_and_split(dataset_path)

    # Load tokenizer
    # TODO: don't hardcode 512
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_key, model_max_length=512, truncation=True)

    # Prepare dataset for training
    dataset = dataset.map(partial(tokenize_batch, tokenizer=tokenizer), batched=True)
    dataset = dataset.map(
        partial(
            add_integral_ids_batch,
            label2id=LABEL2ID,
            jurisdiction_map=JURISDICTION_MAP,
            insurance_type_map=INSURANCE_TYPE_MAP,
        ),
        batched=True,
    )

    # Use our custom data collator that handles the additional features
    data_collator = DataCollatorWithMetadata(tokenizer=tokenizer)

    # Now instantiate your custom model correctly
    model = TextClassificationWithMetadata.from_pretrained(
        pretrained_model_key,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Handle annoyance with HF / pretrained legalbert bug/user error
    # HF trainer complains of param data not being contiguous when loading checkpoints
    if base_model_name == "legal-bert-small-uncased":
        for param in model.parameters():
            param.data = param.data.contiguous()

    checkpoints_dir = os.path.join(OUTPUT_DIR, dataset_name, outdir_name)
    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        run_name=cfg["wandb_run_tag"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        eval_accumulation_steps=16,
        num_train_epochs=cfg["num_epochs"],
        weight_decay=cfg["weight_decay"],
        fp16=(cfg["dtype"] == "float16"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        torch_compile=(cfg["compile"]),
        report_to=report_to,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = CPUEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics2,
    )

    trainer.train()

    # Get best checkpoint dir
    checkpoint_dirs = sorted(os.listdir(checkpoints_dir))
    checkpoint_name = checkpoint_dirs[0]
    ckpt_dir = os.path.join(checkpoints_dir, checkpoint_name)

    # Save tokenizer
    tokenizer_path = os.path.join(ckpt_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

    # Save ONNX binaries for best model
    export_onnx_model(os.path.join(ckpt_dir, "model.onnx"), model.to("cpu"), tokenizer)
    quantize_onnx_model(
        os.path.join(ckpt_dir, "model.onnx"),
        os.path.join(ckpt_dir, "quant-model.onnx"),
    )
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
