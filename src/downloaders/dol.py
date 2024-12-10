import os
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.util import (
    add_jsonl_line,
    download_file,
    gen_src_metadata,
    is_already_downloaded,
)


def get_pdf_links(url: str) -> list[str]:
    try:
        # Fetch the HTML content of the page
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        html_content = response.text

        # Parse the HTML content
        soup = BeautifulSoup(html_content, "html.parser")

        # Find all links with 'pdf' in their href attribute
        pdf_links = [
            a.get("href", None)
            for a in soup.find_all("a")
            if (a.get("href", None) is not None and a.get("href").endswith(".pdf"))
        ]

        # TODO: Also get newer links which have pdfs as additional ref links

        # Make absolute URLs
        pdf_links = [urljoin(url, link) for link in pdf_links]

        return pdf_links

    except Exception as e:
        raise e


def download(output_dir: str, source_meta_path: str):
    # Create output folder for downloaded PDFs if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("Downloading DOL FAQS.")

    init_url = "https://www.dol.gov/agencies/ebsa/laws-and-regulations/laws/affordable-care-act/for-employers-and-advisers/aca-implementation-faqs"
    pdf_links = get_pdf_links(init_url)

    # Download to dir
    for idx, link in enumerate(pdf_links):
        if not is_already_downloaded(link, output_dir):
            try:
                path, hash = download_file(link, output_dir, hash=True)

                # Construct raw source metadata
                tags = ["dol", "aca", "kb", "faq", "regulatory-guidance"]
                proc = "pdf"
                meta = gen_src_metadata(link, path, tags, proc, hash)

                # Add source metadata to sources jsonl
                add_jsonl_line(source_meta_path, meta)
            except requests.exceptions.ConnectionError:
                pass

    # TODO: Manual add ons w/ individual tags
    # pdf_links = [
    #     "https://www.dol.gov/sites/dolgov/files/EBSA/about-ebsa/our-activities/resource-center/faqs/cobra-model-notices.pdf",
    #     "https://www.dol.gov/sites/dolgov/files/EBSA/about-ebsa/our-activities/resource-center/faqs/gina.pdf",
    # ]

    return None
