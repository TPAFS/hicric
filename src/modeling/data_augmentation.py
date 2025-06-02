import json
import os
import random
import uuid
from typing import Any, Tuple

import requests
from datasets import Dataset


def rewrite_to_generic(
    dataset: Dataset,
    num_augmentations_per_example: int = 1,
    api_type: str = "openai",  # "openai" or "llamacpp"
    api_url: str = "https://api.openai.com/v1/chat/completions",
    api_key: str | None = None,
    model_name: str = "gpt-4o",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Rewrite sufficient examples to be too generic to sufficiently describe the medical situation.

    Args:
        dataset: The dataset containing examples to rewrite
        num_augmentations_per_example: Number of rewrites to create per example
        api_type: Type of API to use ("openai" or "llamacpp")
        api_url: URL for the API endpoint
        api_key: API key for authentication (may not be needed for LLaMa.cpp)
        model_name: Name of the model to use
        seed: Random seed for reproducibility

    Returns:
        List of augmented examples with "text", "sufficiency_score", "source_text", "source_score" keys
    """
    random.seed(seed)
    augmented_examples = []

    sufficient_examples = [ex for ex in dataset if ex["sufficiency_score"] >= 3]

    if not sufficient_examples:
        return []

    # Print the total number of examples to process
    total_examples = len(sufficient_examples) * num_augmentations_per_example
    print(
        f"Starting to rewrite {len(sufficient_examples)} examples with {num_augmentations_per_example} augmentations each ({total_examples} total)"
    )

    headers = {"Content-Type": "application/json"}

    if api_key and api_type == "openai":
        headers["Authorization"] = f"Bearer {api_key}"

    system_instruction = """Your task is to rewrite healthcare denial descriptions to be too generic or vague to sufficiently describe the medical situation.

Make the text insufficient by:
1. Removing specific diagnoses, treatments, or medical services
2. Making statements too general (e.g., "my care was denied" instead of "my chemotherapy for stage 3 breast cancer was denied")
3. Avoiding medical terminology that would clarify the situation

Important: Keep the basic subject matter the same, just make it too vague or generic to properly evaluate."""

    user_prompt = """Please rewrite the following healthcare denial description to be too generic or vague to properly evaluate whether the denial was justified. Remove specific details while keeping the general subject matter:

Original: "{}"

Rewrite the text to be insufficient for evaluation by removing key details or making it too generic."""

    completed_count = 0
    progress_interval = 10  # Report progress every 10 examples

    for example in sufficient_examples:
        text = example["text"]
        sufficiency_score = example["sufficiency_score"]

        for _ in range(num_augmentations_per_example):
            prompt = user_prompt.format(text)

            try:
                if api_type == "openai":
                    request_data = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 500,
                        "temperature": 1.2,
                    }
                else:  # llama
                    request_data = {"prompt": f"{system_instruction}\n\n{prompt}", "temperature": 0.7, "stream": False}

                response = requests.post(api_url, headers=headers, data=json.dumps(request_data), timeout=60)

                if response.status_code == 200:
                    response_json = response.json()

                    # Extract the generated text based on API type
                    if api_type == "openai":
                        # Standard OpenAI API response format
                        new_text = response_json["choices"][0]["message"]["content"].strip()

                    else:  # LLaMa.cpp
                        # Print the response structure for debugging
                        print(f"Full LLaMa.cpp response: {response_json}")

                        # Handle various possible response formats from LLaMa.cpp servers
                        if "content" in response_json:
                            new_text = response_json["content"].strip()

                    new_score = 1

                    # Add the augmented example with source tracking
                    augmented_examples.append(
                        {
                            "text": new_text,
                            "sufficiency_score": new_score,
                            "source_text": text,
                            "source_score": sufficiency_score,
                            "augmentation_type": "generic_rewrite",
                        }
                    )

                    # Update progress counter
                    completed_count += 1
                    if completed_count % progress_interval == 0:
                        print(f"{completed_count} / {total_examples} examples completed in rewrite_to_generic")
                else:
                    print(f"Error from API: {response.status_code}, {response.text}")
                    continue

            except Exception as e:
                print(f"Error using API: {e}")
                continue

    print(f"Completed rewrite_to_generic: {completed_count} examples processed")
    return augmented_examples


def generate_unrelated_content(
    num_examples: int = 100,
    api_type: str = "openai",  # "openai" or "llamacpp"
    api_url: str = "https://api.openai.com/v1/chat/completions",
    api_key: str | None = None,
    model_name: str = "gpt-3.5-turbo",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate examples of unrelated or random content that would be insufficient using an LLM.

    Args:
        num_examples: Number of examples to generate
        api_type: Type of API to use ("openai" or "llamacpp")
        api_url: URL for the API endpoint
        api_key: API key for authentication (may not be needed for LLaMa.cpp)
        model_name: Name of the model to use
        seed: Random seed for reproducibility
        fallback_to_static: Whether to fall back to static examples if API fails

    Returns:
        List of generated examples with "text", "sufficiency_score", and metadata
    """
    random.seed(seed)
    augmented_examples = []

    # Print start message
    print(f"Starting to generate {num_examples} unrelated content examples")

    # Set up headers for API requests
    headers = {"Content-Type": "application/json"}

    if api_key and api_type == "openai":
        headers["Authorization"] = f"Bearer {api_key}"

    # Simple system instruction for clarity
    system_instruction = """In what follows I will ask you to generate short text examples that do not describe specific situations pertaining to a single health insurance denial. 
    They should instead be about other topics. The generated texts will be used as negatives to train a model. It will be helpful for them to be diverse, realistic,
    and sometimes similar to the target topic despite being distinct. You can use this category as a starting point:"""

    # Different categories to request
    categories = [
        "Ask a question about hospital charity care.",
        "Ask a question about U.S. health insurance, aside from a direct question about an individual denial.",
        "Ask about Medicare.",
        "Make a general statement about U.S. healthcare.",
        "Make a general statement about a medical topic.",
        "Make a general statement about insurance denials",
        "Make a statement about insurance denials for a particular type of care, but that does not describe a specific situation.",
        "Write some generic unrelated user input (hello, how are you, etc.). Nothing inappropriate.",
        "Write some gibberish that might be input accidentally by a user, or sent inadvertently, cut off sentences, etc.",
    ]

    # Calculate how many examples to generate per category
    examples_per_category = max(1, num_examples // len(categories))
    remaining_examples = num_examples - (examples_per_category * len(categories))

    successful_generations = 0
    all_generations = []

    progress_interval = 10  # Report progress every 10 examples

    # Generate examples for each category
    for category in categories:
        # Calculate how many examples for this category
        num_for_category = examples_per_category
        if remaining_examples > 0:
            num_for_category += 1
            remaining_examples -= 1

        print(f"Generating {num_for_category} examples for category: {category}")

        for _ in range(num_for_category):
            try:
                if api_type == "openai":
                    request_data = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": category},
                        ],
                        "max_tokens": 100,
                        "temperature": 1.2,
                    }
                else:  # LLaMa.cpp
                    request_data = {
                        "prompt": f"{system_instruction}\n\n{category}",
                        "temperature": 1.2,
                        "max_tokens": 100,
                        "stream": False,
                    }

                # Make the API request
                response = requests.post(api_url, headers=headers, data=json.dumps(request_data), timeout=30)

                if response.status_code == 200:
                    response_json = response.json()

                    # Extract the generated text based on API type
                    if api_type == "openai":
                        new_text = response_json["choices"][0]["message"]["content"].strip()
                    else:  # LLaMa.cpp
                        if "content" in response_json:
                            new_text = response_json["content"].strip()

                    # Clean up and limit length
                    if len(new_text) > 512:
                        new_text = new_text[:512]

                    # Add the example with metadata
                    all_generations.append(
                        {
                            "text": new_text,
                            "sufficiency_score": 1,
                            "source_text": None,
                            "source_score": None,
                            "augmentation_type": "unrelated_llm",
                            "category": category,
                        }
                    )

                    successful_generations += 1

                    # Print progress
                    if successful_generations % progress_interval == 0:
                        print(f"{successful_generations} / {num_examples} examples generated")
                else:
                    print(f"Error from API: {response.status_code}, {response.text}")
            except Exception as e:
                print(f"Error generating unrelated content: {e}")

    if len(all_generations) > num_examples:
        augmented_examples = random.sample(all_generations, num_examples)
    else:
        augmented_examples = all_generations

    print(f"Completed generate_unrelated_content: {len(augmented_examples)} examples generated")
    return augmented_examples


def augment_sufficient_examples(
    dataset: Dataset,
    num_augmentations_per_example: int = 1,
    api_type: str = "openai",  # "openai" or "llamacpp"
    api_url: str = "https://api.openai.com/v1/chat/completions",
    api_key: str | None = None,
    model_name: str = "localllama",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Augment sufficient examples using OpenAI-compatible APIs or LLaMa.cpp server.

    Args:
        dataset: The dataset containing examples to augment
        num_augmentations_per_example: Number of augmented versions to create per example
        api_type: Type of API to use ("openai" or "llamacpp")
        api_url: URL for the API endpoint
        api_key: API key for authentication (may not be needed for LLaMa.cpp)
        model_name: Name of the model to use
        seed: Random seed for reproducibility

    Returns:
        List of augmented examples with "text", "sufficiency_score", "source_text", "source_score", and "augmentation_type" keys
    """

    random.seed(seed)
    augmented_examples = []

    # Filter for sufficient examples only (score >= 3)
    sufficient_examples = [ex for ex in dataset if ex["sufficiency_score"] >= 3]

    if not sufficient_examples:
        return []

    # Print start message
    total_examples = len(sufficient_examples) * num_augmentations_per_example
    print(
        f"Starting to augment {len(sufficient_examples)} examples with {num_augmentations_per_example} augmentations each ({total_examples} total)"
    )

    headers = {"Content-Type": "application/json"}

    if api_key and api_type == "openai":
        headers["Authorization"] = f"Bearer {api_key}"

    # Augmentation techniques to apply
    augmentation_techniques = ["patient_perspective", "clinical_language", "word_deletion", "paraphrase", "add_details"]

    # Prompts for different techniques
    perspective_prompt = (
        "Rewrite the following description from a patient's perspective, maintaining all the key details. {}"
    )
    clinical_prompt = "Rewrite the following description using more clinical and technical medical terminology, maintaining all the key details. {}"
    paraphrase_prompt = "Rewrite the following description in different words while preserving the exact same meaning and all key details. Make minimal changes: {}"
    details_prompt = "Rewrite the following description, adding a few more specific details about the condition and treatment, but removing none. Make minimal changes: {}"

    # System instruction for models
    system_instruction = "You are a helpful assistant that rewrites healthcare denial descriptions while preserving their key information."

    completed_count = 0
    progress_interval = 10  # Report progress every 10 examples

    for example in sufficient_examples:
        text = example["text"]
        sufficiency_score = example["sufficiency_score"]

        for _ in range(num_augmentations_per_example):
            technique = random.choice(augmentation_techniques)

            if technique == "word_deletion" or api_type == "openai" and not api_key:
                # Simple word deletion (no LLM needed)
                words = text.split()
                if len(words) <= 3:  # Skip if too few words
                    continue

                # Delete 1-3 random words (but not too many)
                num_to_delete = 1
                indices_to_delete = random.sample(range(len(words)), num_to_delete)

                new_text = " ".join([w for i, w in enumerate(words) if i not in indices_to_delete])

                # Add the augmented example with metadata
                augmented_examples.append(
                    {
                        "text": new_text,
                        "sufficiency_score": sufficiency_score,
                        "source_text": text,
                        "augmentation_type": "sufficient_word_deletion",
                    }
                )

                # Update progress counter
                completed_count += 1
                if completed_count % progress_interval == 0:
                    print(f"{completed_count} / {total_examples} examples completed in augment_sufficient_examples")

            else:
                # Use API for more advanced augmentations
                if technique == "patient_perspective":
                    prompt = perspective_prompt.format(text)
                elif technique == "clinical_language":
                    prompt = clinical_prompt.format(text)
                elif technique == "paraphrase":
                    prompt = paraphrase_prompt.format(text)
                else:  # add_details
                    prompt = details_prompt.format(text)

                try:
                    if api_type == "openai":
                        # OpenAI API format
                        request_data = {
                            "model": model_name,
                            "messages": [
                                {"role": "system", "content": system_instruction},
                                {"role": "user", "content": prompt},
                            ],
                            "max_tokens": 500,
                            "temperature": 1.2,
                        }
                    else:  # LLaMa.cpp
                        # LLaMa.cpp format (depends on your server implementation)
                        request_data = {
                            "prompt": f"{system_instruction}\n\n{prompt}",
                            "temperature": 1.2,
                            "max_tokens": 500,
                            "stop": ["\n\n", "###"],  # Common stop sequences
                            "stream": False,
                        }

                    # Make the API request
                    response = requests.post(
                        api_url,
                        headers=headers,
                        data=json.dumps(request_data),
                        timeout=30,  # Add a timeout to prevent hanging
                    )

                    if response.status_code == 200:
                        response_json = response.json()

                        # Extract the generated text based on API type
                        if api_type == "openai":
                            # Standard OpenAI API response format
                            new_text = response_json["choices"][0]["message"]["content"].strip()
                        else:  # LLaMa.cpp
                            if "content" in response_json:
                                new_text = response_json["content"].strip()

                        # Keep same sufficiency score or slightly increase for add_details
                        new_score = min(5, sufficiency_score + 1) if technique == "add_details" else sufficiency_score

                        # Add the augmented example with metadata
                        augmented_examples.append(
                            {
                                "text": new_text,
                                "sufficiency_score": new_score,
                                "source_text": text,
                                "source_score": sufficiency_score,
                                "augmentation_type": f"sufficient_{technique}",
                            }
                        )

                        # Update progress counter
                        completed_count += 1
                        if completed_count % progress_interval == 0:
                            print(
                                f"{completed_count} / {total_examples} examples completed in augment_sufficient_examples"
                            )
                    else:
                        print(f"Error from API: {response.status_code}, {response.text}")
                        continue

                except Exception as e:
                    print(f"Error using API: {e}")
                    continue

    print(f"Completed augment_sufficient_examples: {completed_count} examples processed")
    return augmented_examples


def load_augmented_dataset(train_path: str, test_path: str) -> dict:
    """
    Load saved augmented datasets from train and test paths.

    Args:
        train_path: Path to the saved augmented training dataset JSONL file
        test_path: Path to the saved augmented test dataset JSONL file

    Returns:
        Dictionary with train and test datasets
    """
    # Load train records
    with open(train_path, "r") as f:
        train_records = [json.loads(line) for line in f]

    # Load test records
    with open(test_path, "r") as f:
        test_records = [json.loads(line) for line in f]

    # Extract answers text from records for train
    train_dataset_records = [
        {"text": rec["answers"]["text"][0], "sufficiency_score": rec["sufficiency_score"]} for rec in train_records
    ]

    # Extract answers text from records for test
    test_dataset_records = [
        {"text": rec["answers"]["text"][0], "sufficiency_score": rec["sufficiency_score"]} for rec in test_records
    ]

    # Create datasets
    train_dataset = Dataset.from_list(train_dataset_records)
    test_dataset = Dataset.from_list(test_dataset_records)

    return {"train": train_dataset, "test": test_dataset}


def process_and_save_augmentations(
    original_dataset_path: str,
    output_train_path: str | None = None,
    output_test_path: str | None = None,
    generic_rewrite_params: dict | None = None,
    unrelated_params: dict | None = None,
    sufficient_augmentation_params: dict | None = None,
    train_test_split_ratio: float = 0.2,
    append_to_original: bool = False,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Apply all augmentation techniques and save the result.

    Args:
        original_dataset_path: Path to the original dataset JSONL file
        output_train_path: Path to save the augmented training dataset (if None, won't save)
        output_test_path: Path to save the augmented test dataset (if None, won't save)
        generic_rewrite_params: Parameters for generic rewrite augmentation (if None, won't apply)
        unrelated_params: Parameters for unrelated content generation (if None, won't apply)
        sufficient_augmentation_params: Parameters for sufficient example augmentation (if None, won't apply)
        train_test_split_ratio: Ratio for the test set split
        append_to_original: Whether to include original examples in output
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Load original dataset
    with open(original_dataset_path, "r") as f:
        records = [json.loads(line) for line in f]

    # Extract answers text from records
    dataset_records = [
        {"text": rec["answers"]["text"][0], "sufficiency_score": rec["sufficiency_score"]} for rec in records
    ]
    dataset = Dataset.from_list(dataset_records)

    # Store original train/test indices to maintain the same split
    random.seed(seed)
    test_indices = set(random.sample(range(len(dataset)), int(len(dataset) * train_test_split_ratio)))
    train_indices = set(range(len(dataset))) - test_indices

    train_records = [dataset_records[i] for i in train_indices]
    test_records = [dataset_records[i] for i in test_indices]

    # Apply augmentations to training data only
    train_dataset = Dataset.from_list(train_records)
    test_dataset = Dataset.from_list(test_records)
    augmented_train_records = []
    augmented_test_records = []

    # 1. Apply generic rewrite augmentation
    if generic_rewrite_params:
        # Training
        generic_train_examples = rewrite_to_generic(train_dataset, **generic_rewrite_params)
        augmented_train_records.extend(generic_train_examples)
        print(f"Generated {len(generic_train_examples)} training set examples through generic rewriting")

        # Testing
        generic_test_examples = rewrite_to_generic(test_dataset, **generic_rewrite_params)
        augmented_test_records.extend(generic_test_examples)
        print(f"Generated {len(generic_test_examples)} test set examples through generic rewriting")

    # 2. Generate unrelated content
    if unrelated_params:
        # Training
        # Set the number of examples to generate to be .1 of the data set size
        num_examples = int(len(train_records) * 0.1)
        unrelated_params.update({"num_examples": num_examples})
        unrelated_train_examples = generate_unrelated_content(**unrelated_params)
        augmented_train_records.extend(unrelated_train_examples)
        print(f"Generated {len(unrelated_train_examples)} training set examples with unrelated content")

        # Testing
        num_examples = int(len(test_records) * 0.1)
        unrelated_params.update({"num_examples": num_examples})
        unrelated_test_examples = generate_unrelated_content(**unrelated_params)
        augmented_test_records.extend(unrelated_test_examples)
        print(f"Generated {len(unrelated_test_examples)} test set examples with unrelated content")

    # 3. Augment sufficient examples
    if sufficient_augmentation_params:
        # Training
        sufficient_examples = augment_sufficient_examples(train_dataset, **sufficient_augmentation_params)
        augmented_train_records.extend(sufficient_examples)
        print(f"Generated {len(sufficient_examples)} training set augmented sufficient examples")

        # Testing
        sufficient_test_examples = augment_sufficient_examples(test_dataset, **sufficient_augmentation_params)
        augmented_test_records.extend(sufficient_test_examples)
        print(f"Generated {len(sufficient_test_examples)} test set augmented sufficient examples")

    # Combine with original examples if requested
    if append_to_original:
        all_train_records = train_records + augmented_train_records
        all_test_records = test_records + augmented_test_records
    else:
        all_train_records = augmented_train_records
        all_test_records = augmented_test_records

    # Create the final datasets
    final_train_dataset = Dataset.from_list(all_train_records)
    final_test_dataset = Dataset.from_list(all_test_records)

    # Save train split to file if requested
    if output_train_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_train_path), exist_ok=True)

        # Format each record with the original dataset structure
        with open(output_train_path, "w") as f:
            for record in all_train_records:
                # Create a full record with the same structure as the original
                record_key = uuid.uuid4().hex
                full_record = {
                    "answers": {"text": [record["text"]], "answer_start": [0]},
                    "id": f"augmented_{record_key}",
                    "question": "What is the background context in this case summary?",
                    "title": f"augmented_{record_key}",
                    "sufficiency_score": record["sufficiency_score"],
                }

                # Add metadata if available
                if "source_text" in record or "source_score" in record or "augmentation_type" in record:
                    full_record["metadata"] = {
                        "source_text": record.get("source_text"),
                        "source_score": record.get("source_score"),
                        "augmentation_type": record.get("augmentation_type"),
                    }

                f.write(json.dumps(full_record) + "\n")

    # Save test split to file if requested
    if output_test_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_test_path), exist_ok=True)

        # Format each record with the original dataset structure
        with open(output_test_path, "w") as f:
            for record in all_test_records:
                # Create a full record with the same structure as the original
                record_key = uuid.uuid4().hex
                full_record = {
                    "answers": {"text": [record["text"]], "answer_start": [0]},
                    "id": f"augmented_{record_key}",
                    "question": "What is the background context in this case summary?",
                    "title": f"augmented_{record_key}",
                    "sufficiency_score": record["sufficiency_score"],
                }

                # Add metadata if available
                if "source_text" in record or "source_score" in record or "augmentation_type" in record:
                    full_record["metadata"] = {
                        "source_text": record.get("source_text"),
                        "source_score": record.get("source_score"),
                        "augmentation_type": record.get("augmentation_type"),
                    }

                f.write(json.dumps(full_record) + "\n")

    return final_train_dataset, final_test_dataset
