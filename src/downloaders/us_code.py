import glob
import os
import zipfile

from src.util import (
    add_jsonl_line,
    download_file,
    gen_src_metadata,
    generate_file_md5,
    is_already_downloaded,
)


def download(output_dir: str, source_meta_path: str) -> None:
    """Download raw US code release point, extract it, and"""
    code_url = "https://uscode.house.gov/download/releasepoints/us/pl/118/34not31/xml_uscAll@118-34not31.zip"
    proc = "usc-xml"

    # Create output folder for downloaded PDFs if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Download US code archive locally
    if not is_already_downloaded(code_url, output_dir):
        print("Downloading US code release point...")
        code_filepath, _hash = download_file(code_url, output_dir, hash=True)
        print("Done.")

        # Extract zip
        with zipfile.ZipFile(code_filepath, "r") as file:
            file.extractall(output_dir)

    # Refs to all xml files
    files = glob.glob(os.path.join(output_dir, "*.xml"))
    files = sorted(files)

    #  xml title
    for filepath in files:
        # Compute individual xml hash
        hash = generate_file_md5(filepath)

        # Metadata to retain in source jsonl record
        tags = ["legal", "us-code", "kb"]

        # Construct raw source metadata
        meta = gen_src_metadata(code_url, filepath, tags, proc, hash)

        # Add source metadata to sources jsonl
        add_jsonl_line(source_meta_path, meta)

    return None
