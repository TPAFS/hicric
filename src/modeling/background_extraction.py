import json
import os
from multiprocessing import Pool, cpu_count

import torch
from rapidfuzz import fuzz
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from src.util import add_jsonl_lines, batcher, get_records_list


def find_contiguous_ones_indices(lst: list[int]) -> tuple[list[int], list[int]]:
    """Given a list of 0s and 1s, provide list of start and list of end indices for
    sequences of 1s.

    E.g. lst = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]

    yields start_indices = [2, 8], end_indices = [4, 10]
    """
    start_indices = []
    end_indices = []
    start = None

    for i, num in enumerate(lst):
        if num == 1:
            if start is None:
                start = i
        elif start is not None:
            # End of a contiguous sequence of 1s
            start_indices.append(start)
            end_indices.append(i - 1)
            start = None

    # Check if the last element is 1
    if start is not None:
        start_indices.append(start)
        end_indices.append(len(lst) - 1)

    return start_indices, end_indices


def get_background_spans(
    context: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification,
    device: str = "cpu",
) -> tuple[list[str], list[tuple[int, int]]]:
    """Use model to predict backround spans in context string."""
    model = model.to(device)
    inputs = tokenizer(context, truncation=True, return_tensors="pt", return_offsets_mapping=True).to(device)
    model = model.to(device)
    offset_mapping = inputs.pop("offset_mapping")[0]
    with torch.no_grad():
        outputs = model(**inputs)

    token_labels = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    start_indices, end_indices = find_contiguous_ones_indices(token_labels)

    start_chars = [offset_mapping[idx][0].item() for idx in start_indices]
    end_chars = [offset_mapping[idx][1].item() for idx in end_indices]
    span_boundaries = list(zip(start_chars, end_chars))

    spans = []
    for start, end in span_boundaries:
        span = context[start : end + 1]
        spans.append(span)
    return spans, span_boundaries


def get_background_spans_batch(
    context_batch, tokenizer, model, device="cpu"
) -> tuple[list[list[str]], list[list[tuple[int, int]]]]:
    inputs = tokenizer(
        context_batch,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=True,
    ).to(device)
    model = model.to(device)

    offset_mapping_batch = inputs.pop("offset_mapping")

    with torch.no_grad():
        batch_outputs = model(**inputs)

    token_label_batch = torch.argmax(batch_outputs.logits, dim=-1).tolist()

    span_boundary_batch = []
    for idx, token_labels in enumerate(token_label_batch):
        start_indices, end_indices = find_contiguous_ones_indices(token_labels)
        start_chars = [offset_mapping_batch[idx][start_idx][0].item() for start_idx in start_indices]
        end_chars = [offset_mapping_batch[idx][end_idx][1].item() for end_idx in end_indices]
        span_boundaries = list(zip(start_chars, end_chars))
        span_boundaries = [span for span in span_boundaries if not (span[0] == 0 and span[1] == 0)]
        span_boundary_batch.append(span_boundaries)

    spans_batch = []
    for idx, span_boundaries in enumerate(span_boundary_batch):
        spans = []
        for start, end in span_boundaries:
            span = context_batch[idx][start : end + 1]
            spans.append(span)
        spans_batch.append(spans)

    return spans_batch, span_boundary_batch


def filter_spans(spans: list[str]) -> list[str]:
    filtered = []
    for span in spans:
        if "uph" in span or "overturn" in span or "reversed" in span or len(span) < 60:
            continue
        else:
            filtered.append(span)
    return filtered


def combine_adjacent_tuples(tuples, k):
    if not tuples:
        return []

    result = [tuples[0]]

    for i in range(1, len(tuples)):
        prev_tuple = result[-1]
        curr_tuple = tuples[i]

        if curr_tuple[0] - prev_tuple[1] <= k:
            result[-1] = (prev_tuple[0], curr_tuple[1])
        else:
            result.append(curr_tuple)

    return result


def get_background_text(context: str, tokenizer: AutoTokenizer, model: torch.nn.Module, device: str):
    _spans, span_boundaries = get_background_spans(context, tokenizer, model, device)
    merged_span_boundaries = combine_adjacent_tuples(span_boundaries, k=10)
    merged_spans = []
    for start, end in merged_span_boundaries:
        span = context[start : end + 1]
        merged_spans.append(span)
    filtered_spans = filter_spans(merged_spans)
    return "\n".join(filtered_spans)


def get_background_text_batch(
    context_batch: list[str],
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    device: str,
):
    span_batch, span_boundary_batch = get_background_spans_batch(context_batch, tokenizer, model, device)

    idx = 0
    text_batch = []
    for _spans, span_boundaries in zip(span_batch, span_boundary_batch):
        merged_span_boundaries = combine_adjacent_tuples(span_boundaries, k=10)
        merged_spans = []
        for start, end in merged_span_boundaries:
            span = context_batch[idx][start : end + 1]
            merged_spans.append(span)
        filtered_spans = filter_spans(merged_spans)
        text_batch.append("\n".join(filtered_spans))
        idx += 1
    return text_batch


def get_sufficiency_batch(
    background_batch: list[str],
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    device: str,
):
    inputs = tokenizer(
        background_batch,
        truncation=True,
        padding=True,
        return_tensors="pt",
        return_offsets_mapping=False,
    ).to(device)
    model = model.to(device)

    with torch.no_grad():
        batch_outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    pred_probs = torch.softmax(batch_outputs.logits, dim=-1)

    sufficiency_ids = torch.where(pred_probs[:, 0] > model.best_threshold, 0, 1).tolist()

    return sufficiency_ids


def construct_recs(raw_recs: list[dict], tokenizer: AutoTokenizer, model: torch.nn.Module, device: str) -> list[dict]:
    backgrounds = []
    idx = 0
    for rec in raw_recs:
        if idx % 1000 == 0:
            print(f"Processed: {idx}/{len(raw_recs)}")
        background = get_background_text(rec["text"], tokenizer, model, device)
        # TODO: add sufficiency to unbatched func variants
        updated_record = {
            "text": background,
            "decision": rec["decision"],
            "appeal_type": rec["appeal_type"],
            "full_text": rec["text"],
            "jurisdiction": rec.get("jurisdiction", "Unspecified"),
            "insurance_type": rec.get("insurance_type", "Unspecified"),
        }
        backgrounds.append(updated_record)
        idx += 1
    return backgrounds


def construct_recs_batch(
    raw_recs: list[dict],
    background_tokenizer: AutoTokenizer,
    background_model: torch.nn.Module,
    sufficiency_tokenizer: AutoTokenizer,
    sufficiency_model: torch.nn.Module,
    device: str,
    batch_size: int = 16,
) -> list[dict]:
    backgrounds = []
    num_batches = len(raw_recs) // batch_size
    batch_idx = 0
    for batch in batcher(raw_recs, batch_size):
        if batch_idx % 100 == 0:
            print(f"Processed: {batch_idx}/{num_batches} batches.")
        context_batch = [rec["text"] for rec in batch]
        background_batch = get_background_text_batch(context_batch, background_tokenizer, background_model, device)
        sufficiency_batch = get_sufficiency_batch(background_batch, sufficiency_tokenizer, sufficiency_model, device)
        batch_records = [
            {
                "text": background,
                "decision": rec["decision"],
                "appeal_type": rec["appeal_type"],
                "full_text": rec["text"],
                "sufficiency_id": sufficiency_id,
                "jurisdiction": rec.get("jurisdiction", "Unspecified"),
                "insurance_type": rec.get("insurance_type", "Unspecified"),
            }
            for (rec, background, sufficiency_id) in zip(batch, background_batch, sufficiency_batch)
        ]
        backgrounds.extend(batch_records)
        batch_idx += 1

    return backgrounds


def filter_recs(
    recs: list[dict],
    min_length: int = 50,
    disallowed_substrings: list[str] = [
        "overturn",
        "uph",
        # "was not medically necessary",
        # "was medically necessary",
    ],
) -> list[dict]:
    """Postprocessing to clean up some obviously poor spans"""
    filtered_recs = []
    for rec in recs:
        for substring in disallowed_substrings:
            if substring in rec["text"]:
                break  # return to outer loop
        if len(rec["text"]) < min_length:
            continue
        else:
            filtered_recs.append(rec)

    return filtered_recs


def is_significant_overlap(line1, line2, threshold=0.95):
    # return SequenceMatcher(None, line1, line2).ratio() > threshold # Sequencematcher too slow
    return fuzz.ratio(line1, line2) > (threshold * 100)


def check_overlap(rec_b, rec_list_a, threshold):
    for rec_a in rec_list_a:
        if is_significant_overlap(rec_a["text"], rec_b["text"], threshold):
            return rec_b, True, rec_a
    return rec_b, False, rec_a


def move_overlapping(rec_list_a, rec_list_b, threshold=0.95):
    new_recs_a = rec_list_a.copy()
    new_recs_b = []
    total_swaps = 0

    with Pool(cpu_count()) as pool:
        results = pool.starmap(check_overlap, [(rec_b, rec_list_a, threshold) for rec_b in rec_list_b])

    for rec_b, overlap_found, rec_a in results:
        if overlap_found:
            total_swaps += 1
            new_recs_a.append(rec_b)
            print(f"Match with overlap found.\n Line A: {rec_a['text']}\n Line B: {rec_b['text']}.")  # debug
        else:
            new_recs_b.append(rec_b)
    print(f"Made {total_swaps} total swaps.")

    return new_recs_a, new_recs_b


def combine_jsonl(directory) -> None:
    output_file = os.path.join(directory, "aggregate.jsonl")
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as infile:
                for line in infile:
                    json_obj = json.loads(line)
                    with open(output_file, "a+", encoding="utf-8") as outfile:
                        outfile.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
    return None


if __name__ == "__main__":
    # Config applied to both models
    device = "cuda"
    quantize = True if device == "cpu" else False
    batch_size = 16

    # Load pretrained background model
    pretrained_model_path = "distilbert/distilbert-base-cased"
    background_dataset = "case-backgrounds"
    checkpoints_dir = f"./models/background_span/{background_dataset}/{pretrained_model_path}"
    trained_model_path = [f.path for f in os.scandir(checkpoints_dir) if f.is_dir()][
        0
    ]  # Only saved best checkpoint for now
    background_tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
    background_model = AutoModelForTokenClassification.from_pretrained(trained_model_path)

    # Load pretrained sufficiency model
    pretrained_model_path = "distilbert/distilbert-base-uncased"
    background_dataset = "case-backgrounds"
    checkpoints_dir = f"./models/sufficiency_predictor/{background_dataset}/{pretrained_model_path}"
    trained_model_path = [f.path for f in os.scandir(checkpoints_dir) if f.is_dir()][
        0
    ]  # Only saved best checkpoint for now
    sufficiency_tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
    sufficiency_model = AutoModelForSequenceClassification.from_pretrained(trained_model_path)
    # Load tuned threshold:
    with open(os.path.join(trained_model_path, "eval_results.json")) as f:
        eval_results = json.loads(f.read())
    sufficiency_model.best_threshold = eval_results["best_threshold"]

    if quantize:
        background_model = torch.ao.quantization.quantize_dynamic(
            background_model,  # the original model
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8,
        ).to(device)  # the target dtype for quantized weights
    else:
        model = background_model.to(device)

    # Combine CA CDI files into single jsonl
    combine_jsonl("./data/processed/ca_cdi/summaries")

    # Define Outcome Map / preprocessing and jurisdiction mapping
    # TODO: decide if this standardization belongs here, or in raw dataset storage (i.e in records we read here)
    extraction_targets = [
        (
            "./data/processed/ca_cdi/summaries/aggregate.jsonl",
            {
                "Insurer Denial Overturned": "Overturned",
                "Insurer Denial Upheld": "Upheld",
            },
        ),
        (
            "./data/processed/ca_dmhc/independent-medical-review-imr-determinations-trend.jsonl",
            {
                "Overturned Decision of Health Plan": "Overturned",
                "Upheld Decision of Health Plan": "Upheld",
            },
        ),
        (
            "./data/processed/ny_dfs/nydfs.jsonl",
            {
                "Overturned": "Overturned",
                "Overturned in Part": "Overturned",
                "Upheld": "Upheld",
            },
        ),
    ]

    # Out path for aggregate dataset
    out_path = "./data/outcomes/backgrounds_suff.jsonl"

    # We will also split a train and test set for consistent experiments
    train_out_path = "./data/outcomes/train_backgrounds_suff.jsonl"
    test_out_path = "./data/outcomes/test_backgrounds_suff.jsonl"
    train_subset = []
    test_subset = []
    for path, outcome_map, jurisdiction, insurance_type in extraction_targets:
        print(f"Processing dataset at {path}")

        # Get records and standardize outcome labels
        recs = get_records_list(path)
        for rec in recs:
            rec["decision"] = outcome_map[rec["decision"]]

        # Extract background spans
        # extracted = construct_recs(recs, tokenizer, model, device) # unbatched variant
        extracted = construct_recs_batch(
            recs, background_tokenizer, background_model, sufficiency_tokenizer, sufficiency_model, device, batch_size
        )

        # Filter results
        filtered_recs = filter_recs(extracted)

        # Write backgrounds and labels to output loc
        add_jsonl_lines(out_path, filtered_recs)

        stratification_keys = [rec["decision"] for rec in filtered_recs]  # stratify splits by outcomes, and dataset
        train_recs, val_recs = train_test_split(
            filtered_recs,
            test_size=0.15,
            random_state=1,
            shuffle=True,
            stratify=stratification_keys,
        )

        # Fix heavy overlaps
        # E.g. Certain situations, such as "An enrollee has requested Harvoni..." occurs many times in both splits
        # This code ensures that highly similar examples appear in exactly one split
        train_recs, val_recs = move_overlapping(train_recs, val_recs)

        train_subset.extend(train_recs)
        test_subset.extend(val_recs)

    add_jsonl_lines(train_out_path, train_subset)
    add_jsonl_lines(test_out_path, test_subset)
