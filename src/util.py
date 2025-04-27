import hashlib
import json
import os
import re
import typing as t
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import html2text
import pdfplumber
import polars as pl
import requests
from bs4 import BeautifulSoup


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a pdf."""
    with pdfplumber.open(pdf_path) as pdf:
        pdf_page_texts = []
        for page in pdf.pages:
            page_text = page.extract_text_simple()
            pdf_page_texts.append(page_text)
    full_text = " ".join(pdf_page_texts)

    return full_text


def dump_pdf_text_to_file(pdf_path: str, text_dir: str) -> Path:
    """Extract text from a pdf and output to text file of the same name in text_dir."""
    with pdfplumber.open(pdf_path) as pdf:
        pdf_page_texts = []
        for page in pdf.pages:
            page_text = page.extract_text_simple()
            pdf_page_texts.append(page_text)
    full_text = " ".join(pdf_page_texts)

    # Dump to file
    filename = Path(pdf_path).stem
    text_path = Path(text_dir) / f"{filename}.txt"
    with open(text_path, "w") as text_file:
        text_file.write(full_text)

    return text_path


def decompress_zip(compressed_path: str, extract_path: str):
    """
    Decompress a zip file and return the path to the decompressed folder.

    Parameters
    ----------
    compressed_path: str
        Path to the zip file.
    extract_path: str
        Path where the contents will be extracted.

    Returns
    -------
    : str
        Path to the decompressed folder.
    """
    with zipfile.ZipFile(compressed_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    # Assuming the zip file contains a single folder, return its path
    decompressed_folder = os.path.join(extract_path, os.path.splitext(os.path.basename(compressed_path))[0])

    return decompressed_folder


def medicare_cd_to_jsonl(csv_path: str, columns: list, output_jsonl_path: str) -> None:
    """Convert Medicare coverage determination csv file to jsonl"""
    df = pl.read_csv(csv_path, encoding="latin-1")

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for row in df.iter_rows(named=True):
            text_pieces = [row[col] for col in columns if "N/A" not in row[col]]
            json_data = {"text": html2text.html2text("\n".join(text_pieces))}
            jsonl_file.write(json.dumps(json_data) + "\n")

    return None


def add_jsonl_line(file_path: str, line_data: dict) -> None:
    # If necessary, make a dir
    if os.path.split(file_path)[0]:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Open the file in append mode, creating it if necessary
    with open(file_path, "a+") as file:
        # Add the new JSON Lines element
        file.write(json.dumps(line_data, ensure_ascii=False))
        file.write("\n")

    return None


def add_jsonl_lines(file_path: str, lines_data: list[dict]) -> None:
    # If necessary, make a dir
    if os.path.split(file_path)[0]:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Open the file in append mode, creating it if necessary
    with open(file_path, "a+") as file:
        for line in lines_data:
            # Add the new JSON Lines element
            file.write(json.dumps(line, ensure_ascii=False))
            file.write("\n")

    return None


def update_jsonl_key(file_path: str, key: str, line_updater: t.Callable) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(file_path, "w") as f:
        for idx, line in enumerate(lines):
            json_obj = json.loads(line)
            if key in json_obj:
                json_obj[key] = line_updater(json_obj[key])
            else:
                raise ValueError(
                    f"Trying to update every occurrence of key={key} in jsonl path, but does not occur on line {idx}"
                )
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
    return None


def generate_file_md5(filepath: str):
    with open(filepath, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def is_already_downloaded(url: str, download_folder: str, filename: t.Optional[str] = None) -> bool:
    if not filename:
        filepath = os.path.join(download_folder, os.path.basename(url))
    else:
        filepath = os.path.join(download_folder, filename)
    return os.path.isfile(filepath)


def download_file(
    url: str,
    download_folder: str,
    hash: bool = False,
    verbose: bool = False,
    extension: t.Optional[str] = None,
    filename: t.Optional[str] = None,
) -> tuple[str, t.Optional[str]]:
    if verbose:
        print(f"Attempting file download from {url}")

    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the filename from the URL
        filepath = os.path.join(
            download_folder,
            os.path.basename(url),
        )
        if filename:
            filepath = os.path.join(download_folder, filename)
        elif extension:
            filepath = filepath + f".{extension}"

        # Save the to the specified folder
        with open(filepath, "wb") as file:
            file.write(response.content)

        # Hash
        if hash:
            filehash = generate_file_md5(filepath)

            return filepath, filehash

        else:
            return filepath, None

    else:
        raise ValueError(f"Failed to fetch file content from {url}")


def gen_src_metadata(
    url: str,
    local_path: str,
    tags: list[str],
    processor: str,
    hash: str,
) -> dict:
    return {
        "url": url,
        "date_accessed": datetime.now().strftime("%Y-%m-%d"),
        "local_path": local_path,
        "tags": tags,
        "preprocessor": processor,
        "md5": hash,
    }


def gen_processed_metadata(
    url: str,
    local_path: str,
    tags: list[str],
    processor: str,
    processed_path: str,
    hash: str,
) -> dict:
    return {
        "url": url,
        "date_accessed": datetime.now().strftime("%Y-%m-%d"),
        "local_path": local_path,
        "tags": tags,
        "preprocessor": processor,
        "local_processed_path": processed_path,
        "md5": hash,
    }


def get_page_links(
    url: str,
    contained_in: t.Optional[str] = None,
    id: t.Optional[str] = None,
    classname: t.Optional[str] = None,
    with_ending: t.Optional[tuple[str]] = None,
) -> list[str]:
    try:
        # Fetch the HTML content of the page
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        html_content = response.text

        # Parse the HTML content
        soup = BeautifulSoup(html_content, "html.parser")

        if contained_in and id:
            target_div = soup.find(contained_in, {"id": id})

        elif contained_in and classname:
            target_div = soup.find(contained_in, class_=classname)
        elif contained_in:
            target_div = soup.find(contained_in)
        else:
            target_div = soup

        # Find all links with 'pdf' in their href attribute
        pdf_links = [a.get("href") for a in target_div.find_all("a") if a.get("href") is not None]

        if with_ending:
            pdf_links = [link for link in pdf_links if link.endswith(with_ending)]

        # Make absolute URLs
        pdf_links = [urljoin(url, link) for link in pdf_links]

        return pdf_links

    except Exception as e:
        raise e


def read_jsonl_batched(jsonl_path: str, batch_size: int):
    with open(jsonl_path, "r") as file:
        batch = []
        for line in file:
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:  # yield non-divisible remainder batch at end of file
            yield batch


def replace_path_component(path: str, component_idx: int, new_text: str) -> str:
    """Replace the kth component of a path with new text."""
    components = path.split("/")
    if component_idx < 0 or component_idx >= len(components):
        raise ValueError("Invalid value for k.")

    components[component_idx] = new_text

    return "/".join(components)


def get_records_list(path: str) -> list[dict]:
    recs = []
    with open(path, "r") as file:
        for line in file:
            recs.append(json.loads(line))
    return recs


def batcher(records: list[dict], batch_size: int = 5):
    """Simple dataset batcher for list of records."""
    for start_idx in range(0, len(records), batch_size):
        yield records[start_idx : start_idx + batch_size]


def normalize_unicode(text: str) -> str:
    normalized_text = unicodedata.normalize("NFKD", text).encode("utf-8").decode("utf8")
    # Return ascii only
    return re.sub(r"[^\x00-\x7F]+", "", normalized_text)
