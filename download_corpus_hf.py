import json
import os

from datasets import DatasetDict, load_dataset, load_from_disk

PARTITIONING_CATS = [
    "legal",
    "regulatory-guidance",
    "contract-coverage-rule-medical-policy",
    "opinion-policy-summary",
    "case-description",
    "clinical-guidelines",
]


def download_dir(repo_name: str = "persius/hicric", output_dir="./arrow_data"):
    """Download the dir from HF hub without cloning, if you like, and save locally."""
    ds_dict = DatasetDict()
    for split in PARTITIONING_CATS:
        ds = load_dataset(repo_name, name=split)
        ds_dict[split] = ds
    ds_dict.save_to_disk(output_dir)
    return None


def repopulate_dir(hf_data_dir: str = "./arrow_data", rehydrate_target_dir: str = "."):
    """Rehydrate the HICRIC processed data dir from the HF Dataset.

    This hydrates the data in the same format in which it was/is originally produced in
    the HICRIC repository's code.
    """

    for split in PARTITIONING_CATS:
        dataset = load_from_disk(os.path.join(hf_data_dir, split, "train"))
        # Get individual lines
        for instance in dataset:
            # Extract the output file/directory associated with line
            rel_path = instance["relative_path"]
            output_file_path = os.path.join(rehydrate_target_dir, rel_path)
            output_directory = os.path.join(rehydrate_target_dir, os.path.dirname(rel_path))
            os.makedirs(output_directory, exist_ok=True)

            with open(output_file_path, "a") as writer:
                writer.write(json.dumps(instance) + "\n")

    print(f"Repopulated data saved to {rehydrate_target_dir}")
    return None


if __name__ == "__main__":
    download_dir()
    repopulate_dir()
