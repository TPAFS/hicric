import os
import string
import typing as t
from functools import partial
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.util import (
    add_jsonl_line,
    download_file,
    gen_src_metadata,
    get_page_links,
    is_already_downloaded,
)


def get_next_url(url: str):
    try:
        # Fetch the HTML content of the page
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        html_content = response.text

        # Parse the HTML content
        soup = BeautifulSoup(html_content, "html.parser")

        # Find the anchor tag with the specified title
        link_element = soup.find("a", class_="pagination-next")

        if link_element:
            # Extract the link associated with the specified title
            link = link_element.get("href")

            # Make the link absolute
            link = urljoin(url, link)

            return link

        else:
            # print(f"No link with title '{title}' found on the page.")
            return None

    except Exception as e:
        return f"Error: {e}"


def isolate_report_pdfs(links: list[str]) -> list[str]:
    final_links = []
    for link in links:
        if link.endswith(".pdf"):
            final_links.append(link)
        else:
            next_res = requests.get(link)
            html_content = next_res.text
            soup = BeautifulSoup(html_content, "html.parser")
            report_link = soup.find("a", string="Complete Report")
            if report_link:
                final_link = urljoin(link, report_link.get("href"))
                final_links.append(final_link)
    return final_links


def get_port_report_links(url: str) -> list[str]:
    first_links = get_page_links(url, contained_in="main", with_ending=(".pdf", ".asp"))
    return isolate_report_pdfs(first_links)


def get_testimony_links(url: str) -> list[str]:
    cur_url = url
    all_pdf_urls = []
    while cur_url:
        # Scrape next metadata pages
        metadata_pages = get_page_links(cur_url, contained_in="ul", classname="usa-card-group usa-section")

        # Scrape pdf urls from those
        for page_url in metadata_pages:
            pdf_urls = get_page_links(page_url, with_ending=".pdf")
            all_pdf_urls.extend(pdf_urls)

        # Walk to next page
        cur_url = get_next_url(cur_url)

    return all_pdf_urls


# Attempt to get "Complete Reports" link from pages that aren't themselves
# reports
def find_pdf(url: str) -> t.Optional[str]:
    if url.endswith(".pdf"):
        return url
    elif url.endswith(".asp"):
        try:
            report_link = get_page_links(url, contained_in="p", classname="report-metadata", with_ending=".pdf")[0]
            return report_link
        except Exception as _e:
            return None


def get_eoi_report_links(base_url: str) -> list[str]:
    letters = list(string.ascii_lowercase)
    all_links = []
    for letter in letters:
        url = base_url + f"{letter}.asp"
        # print(url)
        try:  # Some letters don't have corresponding reports
            links = get_page_links(
                url,
                contained_in="div",
                classname="usa-width-three-fourths usa-layout-docs-main_content",
            )
            links = [link for link in links if "#" not in link]
        except AttributeError:
            links = get_page_links(url, contained_in="div", id="oei_subjects")
            links = [link for link in links if "#" not in link]
        except requests.exceptions.HTTPError:
            # print(err)
            continue
        all_links.extend(links)

    pruned_links = [find_pdf(link) for link in all_links]
    pruned_links = [link for link in pruned_links if link is not None]

    return pruned_links


SCRAPE_TARGETS = [
    {
        "tags": ["hhs-oig", "regulatory-guidance"],
        "url": "https://oig.hhs.gov/reports-and-publications/portfolio/reports/index.asp",
        "crawler": get_port_report_links,
    },
    {
        "tags": ["hhs-oig", "testimony", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/newsroom/testimony/",
        "crawler": get_testimony_links,
    },
    {
        "tags": ["hhs-oig", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/archives/compendium/redbook.asp",
        "crawler": partial(get_page_links, id="leftContentInterior", with_ending=".pdf"),
    },
    {
        "tags": ["hhs-oig", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/top-challenges/2013/2013-tmc.pdf",
        "crawler": None,
    },
    {
        "tags": ["hhs-oig", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/top-challenges/2014/2014-tmc.pdf",
        "crawler": None,
    },
    {
        "tags": ["hhs-oig", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/top-challenges/2015/2015-tmc.pdf",
        "crawler": None,
    },
    {
        "tags": ["hhs-oig", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/top-challenges/2017/2017-tmc.pdf",
        "crawler": None,
    },
    {
        "tags": ["hhs-oig", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/top-challenges/2018/2018-tmc.pdf",
        "crawler": None,
    },
    {
        "tags": ["hhs-oig", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/top-challenges/2019/2019-tmc.pdf",
        "crawler": None,
    },
    {
        "tags": ["hhs-oig", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/top-challenges/2020/2020-tmc.pdf",
        "crawler": None,
    },
    {
        "tags": ["hhs-oig", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/top-challenges/2021/2021-tmc.pdf",
        "crawler": None,
    },
    {
        "tags": ["hhs-oig", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/top-challenges/2022/2022-tmc.pdf",
        "crawler": None,
    },
    {
        "tags": ["hhs-oig", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/top-challenges/2023/2023-tmc.pdf",
        "crawler": None,
    },
    {
        "tags": ["hhs-oig", "medicaid", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/medicaid-integrity/index.asp",
        "crawler": partial(get_page_links, id="leftContentInterior", with_ending=".pdf"),
    },
    {
        "tags": ["hhs-oig", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/archives/semiannual/index.asp",
        "crawler": partial(
            get_page_links,
            contained_in="ul",
            classname="usa-accordion-bordered",
            with_ending=".pdf",
        ),
    },
    {
        "tags": ["hhs-oig", "doj", "hcfac", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/hcfac/index.asp",
        "crawler": partial(get_page_links, contained_in="main", with_ending=".pdf"),
    },
    {
        "tags": ["hhs-oig", "oei", "opinion-policy-summary"],
        "url": "https://oig.hhs.gov/reports-and-publications/oei/",
        "crawler": get_eoi_report_links,
    },
    # {
    #     "tags" : ["hhs-oig", "oas", "cms"],
    #     "url": "https://oig.hhs.gov/reports-and-publications/archives/oas/cms_archive.asp",
    #     "crawler": None,
    # },
]


def download(output_dir: str, source_meta_path: str):
    # Create output folder for downloaded PDFs if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    proc = "pdf"

    for target in SCRAPE_TARGETS:
        tags = target["tags"]
        url = target["url"]
        crawler = target["crawler"]

        if crawler:
            pdf_links = crawler(url)
        else:
            pdf_links = [url]

        # Download to dir
        num_links = len(pdf_links)
        print(f"Preparing to download {num_links} links traversed from: {url}")
        for idx, link in enumerate(pdf_links):
            if idx % 10 == 0:
                print(f"Dled {idx} / {num_links} links.")
            if not is_already_downloaded(link, output_dir):
                # print(f"Downloading link: {link}")
                # Download
                try:
                    path, hash = download_file(link, output_dir, hash=True)

                    # Construct raw source metadata
                    meta = gen_src_metadata(link, path, tags, proc, hash)

                    # Add source metadata to sources jsonl
                    add_jsonl_line(source_meta_path, meta)
                except requests.exceptions.ConnectionError:
                    pass

    return None
