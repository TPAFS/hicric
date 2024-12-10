import os

from src.util import (
    add_jsonl_line,
    decompress_zip,
    download_file,
    gen_src_metadata,
    generate_file_md5,
    is_already_downloaded,
)


def download(output_dir: str, source_meta_path: str):
    # Create output folder for downloaded PDFs if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("Downloading CA DMHC IMR data.")

    dataset_url = "https://data.chhs.ca.gov/dataset/b79b3447-4c10-4ae6-84e2-1076f83bb24e/resource/9ab6e381-bef2-43dd-b096-efa85d93a804/download/independent-medical-review-imr-determinations-trend-hiiiaw.zip"

    # Download to dir
    if not is_already_downloaded(dataset_url, output_dir):
        path, _hash = download_file(dataset_url, output_dir, hash=True)
        decompress_zip(path, output_dir)

    data_path = os.path.join(output_dir, "independent-medical-review-imr-determinations-trend.csv")
    hash = generate_file_md5(data_path)

    # Add source meta

    # Metadata to retain in jsonl record
    tags = ["california", "dmhc", "independent-medical-review", "case-description"]

    # Construct raw source metadata
    proc = "ca_dmhc"
    meta = gen_src_metadata(dataset_url, data_path, tags, proc, hash)

    # Add source metadata to sources jsonl
    add_jsonl_line(source_meta_path, meta)
