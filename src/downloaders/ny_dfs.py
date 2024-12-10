import os

from src.util import add_jsonl_line, download_file, gen_src_metadata


def download(output_dir: str, source_meta_path: str):
    # Create output folder for downloaded PDFs if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("Downloading NY DFS data.")

    url_of_record = "https://www.dfs.ny.gov/public-appeal/search"
    dataset_url = "https://myportal.dfs.ny.gov/peasa-dataextract-portlet/rest/dfsservices/peasaserviceexcel"

    path, hash = download_file(dataset_url, output_dir, hash=True)
    updated_path = os.path.join(output_dir, "nydfs.xlsx")
    os.rename(path, updated_path)

    # Add source meta

    # Metadata to retain in jsonl record
    tags = ["new-york", "dfs", "independent-medical-review", "case-description"]

    # Construct raw source metadata
    proc = "ny_dfs"
    meta = gen_src_metadata(url_of_record, updated_path, tags, proc, hash)

    # Add source metadata to sources jsonl
    add_jsonl_line(source_meta_path, meta)
