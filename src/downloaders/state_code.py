import json
import os
from urllib.parse import urljoin

import aiohttp
import requests
from bs4 import BeautifulSoup

from src.util import (
    add_jsonl_line,
    gen_src_metadata,
    generate_file_md5,
    get_page_links,
    is_already_downloaded,
)


def get_latest_state_code(code_years_url: str) -> tuple[int, str]:
    """Get the latest state code url (and associated year) from a justia state code page."""
    page = requests.get(code_years_url)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="main-content")
    latest_code_li = results.find("ul")
    year = int(latest_code_li.text[:4])
    code_url = urljoin(code_years_url, latest_code_li.find("a").get("href"))
    return year, code_url


def get_page_content(url: str) -> str:
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="codes-content")
    if results:
        return results.text
    else:
        return ""


async def fetch_page(session, url: str):
    async with session.get(url) as response:
        return await response.text()


def get_relevant_code_titles(
    url: str,
    keywords: list[str] = [
        "insurance",
        "health",
        "healthcare",
        "hospital",
        "disability",
    ],
):
    page_content = requests.get(url).text
    soup = BeautifulSoup(page_content, "html.parser")
    title_section_div = soup.find("div", class_="codes-listing")
    title_sections = title_section_div.find_all("li")
    ret = []
    for sec in title_sections:
        if any(key in sec.text.lower() for key in keywords):
            link = sec.find("a").get("href")
            title_name = link.split("/")[-2]
            url = urljoin(url, link)
            ret.append((title_name, url))
    return ret


async def create_nested_doc(url, base_url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            page_content = await response.text()
            soup = BeautifulSoup(page_content, "html.parser")
            title_section_div = soup.find("div", class_="codes-listing")
            if title_section_div is None:
                return [(url, get_page_content(url))]  # Return url so can re-sort
            else:
                text_pieces = []
                title_sections = title_section_div.find_all("li")
                for sec in title_sections:
                    final_link = None
                    further_sec = sec.find_all("a", href=True)
                    for link in further_sec:
                        final_link = link["href"]
                    if final_link is not None:
                        sec_url = base_url + final_link
                        nested_text = await create_nested_doc(sec_url, base_url)
                        text_pieces.extend(nested_text)
                return text_pieces


async def get_title_text(url: str, base_url: str) -> str:
    # Construct title text as single doc
    text_pieces = await create_nested_doc(url, base_url)

    # Sort the text pieces based on the section title (lexicographic order)
    sorted_text_pieces = sorted(text_pieces, key=lambda x: x[0])

    # Extract the text from the sorted text pieces
    final_text = "\n".join(piece[1] for piece in sorted_text_pieces)

    return final_text


# TODO: Make this faster, incredibly slow. Might be rate limited by server, not sure.
async def download(output_dir: str, source_meta_path: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    base_url = "https://law.justia.com"
    proc = None  # Processing coupled to download

    # Get State/region urls
    classname = "panel-pane pane-block pane-justia-laws-listings-2"
    state_code_urls = get_page_links(base_url, contained_in="div", classname=classname)

    # Get refs to latest state codes, and years
    yearly_code_refs = []
    for state_url in state_code_urls:
        year, url = get_latest_state_code(state_url)
        yearly_code_refs.append((year, url))

    for year, state_url in yearly_code_refs:
        # Get relevant titles from entire state code
        ret = get_relevant_code_titles(state_url)

        state = state_url.split("/")[-3]

        for title_name, title_url in ret:
            # Download title text to file
            filename = f"{state}-{title_name}.jsonl"
            skip_list = [
                "california-code-hsc.jsonl",
                "indiana-title-27.jsonl",
                "new-jersey-title-17.jsonl",
                "new-jersey-title-26.jsonl",
            ]  # Issue with these files, to debug
            if not is_already_downloaded(title_url, output_dir, filename=filename) and filename not in skip_list:
                print(f"Downloading {title_name} text from {state}'s {year} state code.")
                try:
                    # Actual title code as text
                    title_text = await get_title_text(title_url, base_url)

                    # Save text to file
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, "w") as file:
                        file.write(json.dumps({"text": title_text}) + "\n")

                    # Compute its hash
                    hash = generate_file_md5(filepath)

                    # Metadata to retain in jsonl record
                    tags = [
                        f"{state}",
                        f"{year}",
                        f"{title_name}",
                        "legal",
                        "state-code",
                        "kb",
                    ]

                    # Construct raw source metadata
                    meta = gen_src_metadata(title_url, filepath, tags, proc, hash)

                    # Add source metadata to sources jsonl
                    add_jsonl_line(source_meta_path, meta)

                except requests.exceptions.ConnectionError as e:
                    print(e)
                    pass

        else:
            print(f"Already downloaded {title_name} text from {state}'s {year} state code.")
