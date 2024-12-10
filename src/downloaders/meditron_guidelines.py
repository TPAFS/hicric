import os

from datasets import load_dataset

from src.util import (
    add_jsonl_line,
    add_jsonl_lines,
    gen_src_metadata,
    generate_file_md5,
)


def download(output_dir: str, source_meta_path: str):
    # Create output folder for downloaded PDFs if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("Downloading Meditron Guidelines dataset.")
    dataset = load_dataset("epfl-llm/guidelines")
    dataset = dataset["train"]
    local_path = os.path.join(output_dir, "guidelines.jsonl")
    add_jsonl_lines(local_path, dataset)

    # Construct single "source" for epfl guidelines dataset
    tags = ["epfl-guidelines", "clinical-guidelines"]

    # Construct raw source metadata
    dataset_url = "https://huggingface.co/datasets/epfl-llm/guidelines"
    proc = "epfl-guidelines"

    md5 = generate_file_md5(local_path)
    meta = gen_src_metadata(dataset_url, local_path, tags, proc, md5)

    # Add source metadata to sources jsonl
    add_jsonl_line(source_meta_path, meta)
