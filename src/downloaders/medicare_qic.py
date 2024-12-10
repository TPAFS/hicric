import os

from src.util import add_jsonl_line, download_file, gen_src_metadata


def download(output_dir: str, source_meta_path: str):
    # Create output folder for downloaded PDFs if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("Downloading Part C QIC Appeals")

    dataset_url = "https://qic.cms.gov/api/1/datastore/query/e1fc1663-7675-4d83-a8dd-5f709f440bbb/0/download?redirect=false&ACA=fgCeroIxj9&format=csv"

    # Download to dir
    path, hash = download_file(dataset_url, output_dir, hash=True)

    updated_path = os.path.join(output_dir, "part_c.csv")
    os.rename(path, updated_path)

    # Metadata to retain in jsonl record
    tags = ["medicare", "part-c", "independent-medical-review"]

    # Construct raw source metadata
    proc = "medicare_qic"
    meta = gen_src_metadata(dataset_url, updated_path, tags, proc, hash)

    # Add source metadata to sources jsonl
    add_jsonl_line(source_meta_path, meta)

    print("Downloading Part D QIC Appeals")

    dataset_url = "https://qic.cms.gov/api/1/datastore/query/8152455d-179d-4455-9d09-e5dfc516be10/0/download?redirect=false&ACA=fgCeroIxj9&format=csv"

    # Download to dir
    path, hash = download_file(dataset_url, output_dir, hash=True)

    updated_path = os.path.join(output_dir, "part_d.csv")
    os.rename(path, updated_path)

    # Metadata to retain in jsonl record
    tags = ["medicare", "part-d", "independent-medical-review", "case-description"]

    # Construct raw source metadata
    proc = "medicare_qic"
    meta = gen_src_metadata(dataset_url, updated_path, tags, proc, hash)

    # Add source metadata to sources jsonl
    add_jsonl_line(source_meta_path, meta)
