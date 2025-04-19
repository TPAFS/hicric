import json
import os
from multiprocessing import Pool, cpu_count

import torch
from datasets import Dataset
from rapidfuzz import fuzz
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from src.modeling.data_augmentation import (
    augment_sufficient_examples,
    generate_unrelated_content,
    rewrite_to_generic,
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


def create_augmented_examples(
    records: list[dict],
    api_type: str = "openai",
    api_url: str = "https://api.openai.com/v1/chat/completions",
    api_key: str | None = None,
    model_name: str = "gpt-4o",
    generic_rewrites_per_example: int = 1,
    sufficient_augmentations_per_example: int = 1,
    num_unrelated_examples: int = 100,
    api_call_limit: int | None = 700,
    seed: int = 42,
) -> list[dict]:
    """
    Create augmented examples from background texts with optional limit on API calls.

    Args:
        records: List of record dictionaries with text, decision, etc.
        api_type: Type of API to use ("openai" or "llamacpp")
        api_url: URL for the API endpoint
        model_name: Name of the model to use
        generic_rewrites_per_example: Number of generic rewrites per sufficient example
        sufficient_augmentations_per_example: Number of sufficient augmentations per sufficient example
        num_unrelated_examples: Number of unrelated content examples to generate
        api_call_limit: Maximum number of API calls to make (if None, no limit)
        seed: Random seed for reproducibility

    Returns:
        List of augmented record dictionaries
    """
    import random

    random.seed(seed)

    print(f"Creating augmented examples using {api_type} API at {api_url}...")
    augmented_records = []

    # Track API calls
    api_calls_made = 0

    # 1. Create dataset for augmentation functions
    dataset_records = []
    for rec in records:
        # Check if record has sufficiency_id
        sufficiency_score = 4  # Default to sufficient for original records
        if "sufficiency_id" in rec:
            sufficiency_score = 4 if rec["sufficiency_id"] == 1 else 2

        dataset_records.append({"text": rec["text"], "sufficiency_score": sufficiency_score})

    dataset = Dataset.from_list(dataset_records)

    # If we have a limit, we need to randomly select which examples to augment
    sufficient_examples = [ex for ex in dataset_records if ex["sufficiency_score"] >= 3]
    num_sufficient = len(sufficient_examples)

    # Calculate how many API calls each technique would require
    total_generic_calls = num_sufficient * generic_rewrites_per_example
    total_sufficient_calls = num_sufficient * sufficient_augmentations_per_example
    total_unrelated_calls = num_unrelated_examples

    # Calculate total potential API calls
    total_potential_calls = total_generic_calls + total_sufficient_calls + total_unrelated_calls

    # If we have a limit and it's less than the potential total, adjust
    if api_call_limit is not None and api_call_limit < total_potential_calls:
        print(f"API call limit ({api_call_limit}) is less than potential total ({total_potential_calls})")

        # Distribute the limit evenly across the three techniques
        calls_per_technique = api_call_limit // 3

        # Calculate adjusted calls for each technique
        adjusted_generic_calls = calls_per_technique
        adjusted_sufficient_calls = calls_per_technique
        adjusted_unrelated_calls = (
            api_call_limit - adjusted_generic_calls - adjusted_sufficient_calls
        )  # Use remainder for unrelated

        # Calculate how many examples we can augment for each technique
        examples_for_generic = adjusted_generic_calls // generic_rewrites_per_example
        examples_for_sufficient = adjusted_sufficient_calls // sufficient_augmentations_per_example

        # Randomly select examples to augment
        if examples_for_generic < num_sufficient:
            examples_generic = random.sample(sufficient_examples, examples_for_generic)
        else:
            examples_generic = sufficient_examples

        if examples_for_sufficient < num_sufficient:
            examples_sufficient = random.sample(sufficient_examples, examples_for_sufficient)
        else:
            examples_sufficient = sufficient_examples

        # Adjust unrelated examples count
        adjusted_num_unrelated = adjusted_unrelated_calls

        print("Adjusted numbers based on API call limit:")
        print(f"  Generic rewrites: {examples_for_generic} examples ({adjusted_generic_calls} calls)")
        print(f"  Sufficient augmentations: {examples_for_sufficient} examples ({adjusted_sufficient_calls} calls)")
        print(f"  Unrelated content: {adjusted_num_unrelated} examples ({adjusted_unrelated_calls} calls)")

        # Create filtered datasets for limited augmentation
        generic_dataset = Dataset.from_list(examples_generic)
        sufficient_dataset = Dataset.from_list(examples_sufficient)

        # Update parameters
        generic_params_count = adjusted_generic_calls
        sufficient_params_count = adjusted_sufficient_calls
        unrelated_params_count = adjusted_unrelated_calls
    else:
        # No limit or limit is high enough, use all examples
        generic_dataset = dataset
        sufficient_dataset = dataset
        generic_params_count = total_generic_calls
        sufficient_params_count = total_sufficient_calls
        unrelated_params_count = num_unrelated_examples

    # 2. Apply generic rewrite augmentation (make sufficient examples insufficient)
    if generic_params_count > 0:
        print("Generating generic rewrites...")
        generic_rewrite_params = {
            "num_augmentations_per_example": generic_rewrites_per_example,
            "api_type": api_type,
            "api_url": api_url,
            "api_key": api_key,
            "model_name": model_name,
            "seed": seed,
        }

        generic_examples = rewrite_to_generic(generic_dataset, **generic_rewrite_params)
        api_calls_made += len(generic_examples)
        print(f"Generated {len(generic_examples)} examples through generic rewriting")
    else:
        generic_examples = []
        print("Skipping generic rewrites due to API call limit")

    # 3. Apply sufficient example augmentation (keep sufficient examples sufficient)
    if sufficient_params_count > 0:
        print("Generating sufficient augmentations...")
        sufficient_augmentation_params = {
            "num_augmentations_per_example": sufficient_augmentations_per_example,
            "api_type": api_type,
            "api_url": api_url,
            "api_key": api_key,
            "model_name": model_name,
            "seed": seed,
        }

        sufficient_examples = augment_sufficient_examples(sufficient_dataset, **sufficient_augmentation_params)
        api_calls_made += len(sufficient_examples)
        print(f"Generated {len(sufficient_examples)} augmented sufficient examples")
    else:
        sufficient_examples = []
        print("Skipping sufficient augmentations due to API call limit")

    # 4. Generate unrelated content
    if unrelated_params_count > 0:
        print("Generating unrelated content...")
        unrelated_params = {
            "num_examples": unrelated_params_count,
            "api_type": api_type,
            "api_url": api_url,
            "api_key": api_key,
            "model_name": model_name,
            "seed": seed,
        }

        unrelated_examples = generate_unrelated_content(**unrelated_params)
        api_calls_made += len(unrelated_examples)
        print(f"Generated {len(unrelated_examples)} examples with unrelated content")
    else:
        unrelated_examples = []
        print("Skipping unrelated content due to API call limit")

    # 5. Convert augmented texts back to record format
    # For generic rewrites and sufficient augmentations, we need to find the original record
    all_records_dict = {rec["text"]: rec for rec in records}

    for aug_example in generic_examples + sufficient_examples:
        source_text = aug_example["source_text"]
        # Find the original record
        original_rec = all_records_dict.get(source_text)
        if original_rec:
            augmented_rec = original_rec.copy()
            augmented_rec["text"] = aug_example["text"]
            augmented_rec["sufficiency_id"] = 1 if aug_example["sufficiency_score"] >= 3 else 0
            augmented_rec["augmentation_type"] = aug_example["augmentation_type"]
            augmented_records.append(augmented_rec)

    # For unrelated content, create new records with random decision label
    decisions = ["Upheld", "Overturned"]
    appeal_types = ["IMR", "DMHC", "CDI"]  # Example appeal types

    for i, aug_example in enumerate(unrelated_examples):
        augmented_rec = {
            "text": aug_example["text"],
            "decision": random.choice(decisions),  # Random decision since unrelated to actual cases
            "appeal_type": random.choice(appeal_types),
            "full_text": aug_example["text"],  # No original full text
            "sufficiency_id": 0,  # Always insufficient
            "jurisdiction": "Unspecified",
            "insurance_type": "Unspecified",
            "augmentation_type": aug_example["augmentation_type"],
            "id": f"unrelated_{i}",
        }
        augmented_records.append(augmented_rec)

    print(f"Total API calls made: {api_calls_made}")
    print(f"Total augmented records created: {len(augmented_records)}")
    return augmented_records


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
    pretrained_model_path = "distilbert/distilbert-base-cased"
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

    # New paths for augmented datasets
    train_augmented_out_path = "./data/outcomes/train_backgrounds_suff_augmented.jsonl"
    test_augmented_out_path = "./data/outcomes/test_backgrounds_suff_augmented.jsonl"

    train_subset = []
    test_subset = []
    for path, outcome_map in extraction_targets:
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

    # Write original train and test sets
    add_jsonl_lines(train_out_path, train_subset)
    add_jsonl_lines(test_out_path, test_subset)

    print(f"Original train set: {len(train_subset)} examples")
    print(f"Original test set: {len(test_subset)} examples")

    # Check for OpenAI API key in environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    api_type = "openai" if api_key else "llamacpp"
    api_url = "https://api.openai.com/v1/chat/completions" if api_key else "http://localhost:8080/completion"
    model_name = "gpt-4o" if api_key else "llama-3.1"

    print(f"Using {api_type} API for augmentation")

    train_subset = get_records_list(train_out_path)
    test_subset = get_records_list(test_out_path)

    # Create augmented examples for train set
    print("\n=== Creating Augmented Examples for Train Set ===")
    train_augmented = create_augmented_examples(
        train_subset,
        api_type=api_type,
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        generic_rewrites_per_example=2,
        sufficient_augmentations_per_example=2,
        num_unrelated_examples=100,
        seed=42,
    )

    # Write augmented train and test sets
    # Combine original and augmented examples
    train_combined = train_subset + train_augmented
    add_jsonl_lines(train_augmented_out_path, train_combined)

    # Create augmented examples for test set
    print("\n=== Creating Augmented Examples for Test Set ===")
    test_augmented = create_augmented_examples(
        test_subset,
        api_type=api_type,
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        generic_rewrites_per_example=1,  # Fewer augmentations for test set
        sufficient_augmentations_per_example=1,
        num_unrelated_examples=10,
        seed=43,  # Different seed for test set
    )

    # Combine original and augmented examples
    test_combined = test_subset + test_augmented

    # Write augmented test set
    add_jsonl_lines(test_augmented_out_path, test_combined)

    # print(f"\nFinal augmented train set: {len(train_combined)} examples")
    print(f"Final augmented test set: {len(test_combined)} examples")

    # Print augmentation statistics
    train_augmented_types = {}
    for rec in train_augmented:
        aug_type = rec.get("augmentation_type", "unknown")
        if aug_type not in train_augmented_types:
            train_augmented_types[aug_type] = 0
        train_augmented_types[aug_type] += 1

    print("\nTrain set augmentation breakdown:")
    for aug_type, count in train_augmented_types.items():
        print(f"  {aug_type}: {count}")
