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


def get_page_pdf_links(url: str) -> list[str]:
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
            if (a.get("href", None) is not None and a.get("href").startswith("/media"))
        ]

        # Make absolute URLs
        pdf_links = [urljoin(url, link) for link in pdf_links]

        return pdf_links

    except Exception as e:
        raise e


def get_next_url(url: str):
    try:
        # Fetch the HTML content of the page
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        html_content = response.text

        # Parse the HTML content
        soup = BeautifulSoup(html_content, "html.parser")

        # Find the anchor tag with the specified title
        title = "Go to next page"
        link_element = soup.find("a", {"title": title})

        if link_element:
            # Extract the link associated with the specified title
            link = link_element.get("href")

            # Make the link absolute
            link = urljoin(url, link)

            return link

        else:
            print(f"No link with title '{title}' found on the page.")
            return None

    except Exception as e:
        return f"Error: {e}"


def get_pdf_links(init_url: str, end_page: int = 10) -> list[str]:
    # Pre-specify pages
    page_urls = [init_url] + [f"{init_url}?page={pagenum}" for pagenum in range(1, end_page)]

    all_links = set([])
    for url in page_urls:
        try:
            links = get_page_pdf_links(url)
            all_links.update(links)
            print(f"Processed page: {url}. {len(all_links)} links compiled total.")
        except Exception as e:
            print(f"Failed to process page: {url}.\n Error: {e}")

    return all_links


def download(output_dir: str, source_meta_path: str):
    # Create output folder for downloaded PDFs if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("Downloading medicaid guidance.")
    # Choose whether to use archive.org snapshot:
    init_url = "https://www.medicaid.gov/federal-policy-guidance/index.html"
    # init_url = "https://web.archive.org/web/20240101171822/https://www.medicaid.gov/federal-policy-guidance/index.html"
    pdf_links = get_pdf_links(init_url, end_page=102)

    # Download to dir
    num_links = len(pdf_links)
    for idx, link in enumerate(pdf_links):
        if idx % 10 == 0:
            print(f"Dled {idx} / {num_links} links.")
        if not is_already_downloaded(link, output_dir):
            # print(f"Downloading link: {link}")
            # Download
            try:
                path, hash = download_file(link, output_dir, hash=True, extension="pdf")

                # Construct raw source metadata
                tags = ["medicaid", "kb", "regulatory-guidance"]
                proc = "pdf"
                meta = gen_src_metadata(link, path, tags, proc, hash)

                # Add source metadata to sources jsonl
                add_jsonl_line(source_meta_path, meta)
            except requests.exceptions.ConnectionError:
                pass

    return None
