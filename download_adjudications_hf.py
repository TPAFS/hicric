import os

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def download_data(repo_name: str = "Persius/imr-appeals", output_dir="./data"):
    """Download adjudication data from HF hub, and save it locally in the format expected by HICRIC codebase."""
    train = load_dataset(repo_name, split="train")
    test = load_dataset(repo_name, split="test")
    train.to_json(os.path.join(output_dir, "outcomes", "train_backgrounds_suff.jsonl"))
    test.to_json(os.path.join(output_dir, "outcomes", "test_backgrounds_suff.jsonl"))

    _path = hf_hub_download(
        repo_id="Persius/imr-appeals",
        filename="case-backgrounds.jsonl",
        local_dir=output_dir,
        subfolder="annotated",
        repo_type="dataset",
    )
    return None


if __name__ == "__main__":
    download_data()
