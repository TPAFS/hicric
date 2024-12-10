import argparse
import collections
import datetime
import itertools
import json
import math
import os
from functools import partial

import datasets
import numpy as np
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from src.modeling.util import load_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

OUTPUT_DIR = "./models/fill_mask"


# Load all of the processed sources into
def corpus_generator(processed_meta_path: str, exclude_list: list[str], train_case_paths: list[str]):
    paths = []
    with open(processed_meta_path, "r") as meta_f:
        for line in meta_f:
            # tags = json.loads(line)["tags"]
            path = json.loads(line)["local_processed_path"]
            if path not in exclude_list:  # Exclude data sources used to construct test set
                paths.append(path)

    # Add back in case descriptions not appearing in test set.
    for path in train_case_paths:
        with open(path, "r") as f:
            for line in f:
                yield {
                    "text": json.loads(line)["full_text"]
                }  # text key in this dataset is a model output, not the case description.

    for path in paths:
        with open(path, "r") as f:
            for line in f:
                yield {"text": json.loads(line)["text"]}


def tokenize_function(examples, tokenizer):
    result = tokenizer(examples["text"], truncation=None, padding=False)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# TODO: don't group this way that allows unrelated texts to be concatenated. Instead
# pad short examples, and chop long examples. No concat of unrelated examples.
def group_texts(examples, chunk_size: int):
    concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}

    # Compute length of concatenated texts
    total_length = len(concatenated_examples["input_ids"])

    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size

    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)] for k, t in concatenated_examples.items()
    }

    # Create a new labels column
    result["labels"] = result["input_ids"].copy()

    return result


def pad_and_group_texts(examples: dict, tokenizer: AutoTokenizer, chunk_size: int):
    new_examples = {k: [] for k in examples.keys()}

    for i in range(len(examples["input_ids"])):
        for k in examples.keys():
            tokens = examples[k][i]
            if k == "word_ids":
                # Handle word_ids separately
                word_ids = examples[k][i]
                if len(word_ids) < chunk_size:
                    word_ids += [-1] * (chunk_size - len(word_ids))
                    new_examples[k].append(word_ids)
                else:
                    for j in range(0, len(word_ids), chunk_size):
                        chunk = word_ids[j : j + chunk_size]
                        if len(chunk) < chunk_size:
                            chunk += [-1] * (chunk_size - len(chunk))
                        new_examples[k].append(chunk)
            else:
                if len(tokens) < chunk_size:
                    tokens = tokenizer.pad({"input_ids": tokens}, padding="max_length", max_length=chunk_size)[
                        "input_ids"
                    ]
                    new_examples[k].append(tokens)
                else:
                    for j in range(0, len(tokens), chunk_size):
                        chunk = tokens[j : j + chunk_size]
                        if len(chunk) < chunk_size:
                            chunk = tokenizer.pad({"input_ids": chunk}, padding="max_length", max_length=chunk_size)[
                                "input_ids"
                            ]
                        new_examples[k].append(chunk)

    new_examples["labels"] = new_examples["input_ids"].copy()

    return new_examples


def whole_word_masking_data_collator(features: dict, tokenizer: AutoTokenizer, mask_prob: float):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, mask_prob, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    base_model_name = cfg["base_model_name"]
    pretrained_model_key = cfg["pretrained_hf_model_key"]

    # Optional wandb config
    report_to = "none"
    if cfg["wandb_logging"]:
        wandb_project = cfg["wandb_project"]
        run_tag = cfg["wandb_run_tag"] + f"_{base_model_name}"
        wandb_experiment_name = "-".join([run_tag, datetime.datetime.now().strftime("%B-%d-%Y-%I-%M-%p")])

        wandb.init(project=wandb_project, name=wandb_experiment_name, config=cfg)
        report_to = "wandb"

    model = AutoModelForMaskedLM.from_pretrained(pretrained_model_key)

    # Annoyance with HF/ legalbert small bug
    if base_model_name == "legal-bert-small-uncased":
        for param in model.parameters():
            param.data = param.data.contiguous()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_key, clean_up_tokenization_spaces=False)
    chunk_size = tokenizer.model_max_length
    chunk_size = 512

    # Store and load preprocessed from file, since HF cache not working for some reason.
    processed_pretrain_path = f"./data/hf_cache/{base_model_name}/processed_pretrain.hf"
    if not os.path.exists(processed_pretrain_path):
        print("No cached processed dataset, processing.")
        processed_meta_path = "processed_sources.jsonl"
        dataset = Dataset.from_generator(
            partial(
                corpus_generator,
                processed_meta_path=processed_meta_path,
                exclude_list=[
                    "./data/processed/ca_dmhc/independent-medical-review-imr-determinations-trend.jsonl",
                    "./data/processed/ny_dfs/nydfs.jsonl",
                    "./data/processed/ca_cdi/summaries/aggregate.jsonl",
                ],
                train_case_paths=["./data/outcomes/train_backgrounds_suff.jsonl"],
            )
        ).train_test_split(test_size=0.1)
        tokenized_dataset = dataset.map(
            partial(tokenize_function, tokenizer=tokenizer),
            batched=True,
            remove_columns=["text"],
            load_from_cache_file=True,
        )
        grouped_ds = tokenized_dataset.map(
            partial(pad_and_group_texts, chunk_size=chunk_size, tokenizer=tokenizer),
            batched=True,
            load_from_cache_file=True,
        )
        grouped_ds.save_to_disk(processed_pretrain_path)
    else:
        print("Loading cached processed dataset.")
        grouped_ds = datasets.load_from_disk(processed_pretrain_path)

    mask_prob = cfg["mask_prob"]
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mask_prob)
    data_collator = partial(
        whole_word_masking_data_collator, tokenizer=tokenizer, mask_prob=mask_prob
    )  # Mask whole words not arbitrary tokens

    batch_size = cfg["batch_size"]

    # Show the training loss with every epoch
    logging_steps = len(grouped_ds["train"]) // batch_size

    checkpoints_dir = os.path.join(OUTPUT_DIR, base_model_name)
    resume_from_checkpoint = cfg.get("resume_from_checkpoint", False)

    # resume_from_checkpoint = None
    training_args = TrainingArguments(
        output_dir=f"./models/fill_mask/{base_model_name}",
        evaluation_strategy="epoch",
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        push_to_hub=False,
        fp16=(cfg["dtype"] == "float16"),
        logging_steps=logging_steps,
        remove_unused_columns=False,
        save_strategy="epoch",
        save_total_limit=2,  # Want to save last checkpoint to be able to resume
        load_best_model_at_end=True,
        report_to=report_to,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        torch_compile=(cfg["compile"]),
        resume_from_checkpoint=resume_from_checkpoint,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=grouped_ds["train"],
        eval_dataset=grouped_ds["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save tokenizer
    tokenizer_path = os.path.join(checkpoints_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="src/modeling/config/pretrain/legalbert_small.yaml",
        help="Path to configuration file.",
        dest="config_path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    main(**args_dict)
