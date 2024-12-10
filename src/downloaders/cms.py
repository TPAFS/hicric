import json
import os
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

from src.util import download_file, gen_src_metadata, is_already_downloaded


def download_cms_pdfs(
    url,
    table_header,
    secondary_page_heading,
    output_jsonl,
    download_folder="pdfs",
    req_delay=0.5,
):
    # Create a session with retry handling
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=req_delay,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    response = session.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")

        target_header = soup.find("th", string=table_header)

        if target_header:
            # Determine the index of the column based on the position of the header
            column_index = target_header.parent.find_all("th").index(target_header)

            # Create a folder for downloaded PDFs if it doesn't exist
            if not os.path.exists(download_folder):
                os.makedirs(download_folder, exist_ok=True)

            # Iterate through all rows in the table
            for row in soup.find_all("tr"):
                # Find the cells in the current row
                cells = row.find_all("td")

                # Check if the row has enough cells and the current cell is in the target column
                if len(cells) > column_index:
                    # Get the link in the target column
                    link = cells[column_index].find("a", href=True)

                    if link:
                        # Construct the absolute URL
                        link_url = urljoin(url, link["href"])
                        # print(f"Found link to downloads page: {link_url}")

                        # Follow the link to the next webpage
                        link_response = requests.get(link_url)

                        time.sleep(req_delay)

                        # Check if the request was successful
                        if link_response.status_code == 200:
                            # Parse the HTML content of the linked page
                            link_soup = BeautifulSoup(link_response.content, "html.parser")
                            heading = link_soup.find(
                                lambda tag: tag.name == "h2" and secondary_page_heading in tag.text
                            )

                            if heading:
                                # print("Found downloads heading.")
                                # Find the unordered list under the heading
                                unordered_list = heading.find_next("ul")

                                if unordered_list:
                                    # Iterate through all list items in the unordered list
                                    for li in unordered_list.find_all("li"):
                                        # Get the PDF link in the list item
                                        pdf_link = li.find("a", href=True)

                                        if pdf_link and pdf_link["href"].endswith(".pdf"):
                                            # Construct the absolute PDF URL
                                            pdf_url = urljoin(link_url, pdf_link["href"])

                                            if not is_already_downloaded(pdf_url, download_folder):
                                                print(f"Attempting download of {pdf_url}")

                                                # Make a request to the PDF URL
                                                pdf_response = session.get(pdf_url)

                                                # Check if the request was successful
                                                if pdf_response.status_code == 200:
                                                    filename, filehash = download_file(
                                                        pdf_url,
                                                        download_folder,
                                                        hash=True,
                                                    )

                                                    # Create a dictionary for the PDF record
                                                    local_path = (
                                                        f"./data/raw/regulatory_guidance/{os.path.basename(pdf_url)}"
                                                    )
                                                    tags = [
                                                        "cms",
                                                        "kb",
                                                        "regulatory-guidance",
                                                    ]
                                                    proc = "pdf"
                                                    rec = gen_src_metadata(
                                                        pdf_url,
                                                        local_path,
                                                        tags,
                                                        proc,
                                                        filehash,
                                                    )

                                                    # Write the PDF record to the JSONL file
                                                    with open(output_jsonl, "a") as jsonl_file:
                                                        jsonl_file.write(json.dumps(rec) + "\n")

                                                else:
                                                    print(f"Failed to download PDF from {pdf_url}")
                                                time.sleep(req_delay)

                                            else:
                                                print(f"Already downloaded links from {link_url}")
                                                break

                                else:
                                    print(f"No unordered list found under heading '{secondary_page_heading}'.")
                            else:
                                print("No heading found with text in secondary_page_headings.")
                        else:
                            print(f"Failed to fetch linked page. Status code: {link_response.status_code}")
        else:
            print(f"Table header '{table_header}' not found.")
    else:
        print(f"Failed to fetch the main webpage. Status code: {response.status_code}")
