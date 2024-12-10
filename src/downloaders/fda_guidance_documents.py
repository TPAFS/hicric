import os
import time

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.firefox.options import Options

from src.util import (
    add_jsonl_line,
    download_file,
    gen_src_metadata,
    is_already_downloaded,
)


def download(out_dir, outfile):
    options = Options()
    # Set download directory
    download_dir = out_dir
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)
    # Configure Firefox profile accordingly
    firefox_profile = FirefoxProfile()
    firefox_profile.set_preference("browser.download.folderList", 2)
    firefox_profile.set_preference("browser.download.dir", download_dir)
    firefox_profile.set_preference("browser.download.useDownloadDir", True)
    firefox_profile.set_preference("browser.download.manager.showWhenStarting", False)
    firefox_profile.set_preference("pdfjs.disabled", True)
    firefox_profile.set_preference(
        "browser.helperApps.neverAsk.saveToDisk",
        "application/pdf",
    )
    options.profile = firefox_profile
    # options.add_argument("--headless")

    driver = webdriver.Firefox(options=options)

    # Navigate to the website
    url = "https://www.fda.gov/regulatory-information/search-fda-guidance-documents#guidancesearch"
    driver.get(url)

    # Wait for the page to load (adjust the sleep time according to your page loading time)
    time.sleep(3)

    more_pages = True
    pages_seen = 0
    while more_pages:
        print(f"Processsing page {pages_seen + 1}")
        # Get relevant table
        table_element = driver.find_element(By.XPATH, "//table[@id='DataTables_Table_0']")

        # Columns with links + metadata of interest
        summaries = table_element.find_elements(By.XPATH, "//td[1]/a")
        links = table_element.find_elements(By.XPATH, "//td[2]/a")
        dates = table_element.find_elements(By.XPATH, "//td[3]")
        topics = table_element.find_elements(By.XPATH, "//td[5]")

        # Extract links from the column
        for sum, link, date, topic in zip(summaries, links, dates, topics):
            link = link.get_attribute("href")
            sum = sum.text  # unused for now
            topic = topic.text
            date = date.text

            additional_tags = []
            topic_tags = topic.lower().split(",")
            topic_tags = [tag.strip() for tag in topic_tags]
            topic_tags = [tag for tag in topic_tags if tag != ""]
            date_tag = date.split("/")[-1]
            if len(topic_tags) > 0:
                additional_tags.extend(topic_tags)
            additional_tags.append(date_tag)

            filename = link.split("/")[-2]
            filename = f"{filename}.pdf"

            if not is_already_downloaded(link, download_dir, filename):
                try:
                    path, hash = download_file(link, download_dir, hash=True, filename=filename)

                    # Construct raw source metadata
                    tags = ["fda", "clinical-guidelines"] + additional_tags
                    proc = "pdf"
                    meta = gen_src_metadata(link, path, tags, proc, hash)

                    # Add source metadata to sources jsonl
                    add_jsonl_line(outfile, meta)
                except requests.exceptions.ConnectionError:
                    print(f"Connection error encountered while attempting to download file from {link}")
                    pass

                except ValueError:
                    print(f"Value error encountered in attempting to download file from {link}")
                    pass

        # Find the button by text
        try:
            next_page_li = driver.find_element(By.XPATH, "//li[@class='paginate_button next']")

            next_page_button = next_page_li.find_element(By.XPATH, "//a[contains(text(), 'Next') and @href='#']")

            # Scroll to the element
            driver.execute_script("arguments[0].scrollIntoView(true);", next_page_button)

            # Click the button using JavaScript
            driver.execute_script("arguments[0].click();", next_page_button)

            # Wait for the next page to load
            time.sleep(1)
            pages_seen += 1

        except Exception as _e:
            more_pages = False

    # Close the browser window
    driver.quit()

    return None
