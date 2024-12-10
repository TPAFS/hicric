import json
import os
import time

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.util import add_jsonl_line


def scrape_policies(homepage_url: str) -> list[dict]:
    options = Options()

    # Configure Firefox profile accordingly
    firefox_profile = FirefoxProfile()

    options.profile = firefox_profile
    # options.add_argument("--headless")

    driver = webdriver.Firefox(options=options)

    wait = WebDriverWait(driver, 10)

    policies = []

    try:
        driver.get(homepage_url)

        # Wait for the modal to appear and click the link that starts with "Accept and go to"
        try:
            accept_link = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//a[starts-with(text(), 'Accept and go to')]"))
            )
            accept_link.click()
            time.sleep(2)  # Wait for the modal to close
        except Exception as e:
            print("No modal dialog found or unable to click accept link:", e)

        while True:
            # Wait for the "right-wrap" div to load
            right_wrap_div = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "right-wrap")))
            table_rows = right_wrap_div.find_elements(By.CLASS_NAME, "row")

            for row in table_rows:
                cols = row.find_elements(By.CLASS_NAME, "col-md-2")
                if cols:
                    document_number = cols[0].text.strip()
                    if document_number == "Policy #":
                        continue
                else:
                    continue

                row_elem = row.find_element(By.CLASS_NAME, "col-md-10")
                document_title = row_elem.text.strip()
                link_elem = row_elem.find_elements(By.XPATH, ".//a")[0]
                document_url = link_elem.get_attribute("href")

                if document_title != "" and document_number != "":
                    policies.append({"title": document_title, "policy_number": document_number, "url": document_url})

            # Check for 'Next' button and click if exists
            try:
                next_button = driver.find_element(By.XPATH, "//span[contains(text(), 'Next')]")
                next_button.click()
                time.sleep(2)  # Wait for the next page to load
            except Exception:
                print("No more pages left to scrape")
                break

    finally:
        driver.quit()

    return policies


if __name__ == "__main__":
    # Set download directory
    download_dir = "./data/raw/policy_bulletins/ibx"
    download_dir = os.path.abspath(download_dir)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)

    # Scrape target links and compile jsonl (Commercial)
    homepage = "https://medpolicy.ibx.com/ibc/Commercial/Pages/Policy-Bulletin-View.aspx"
    targets_list = scrape_policies(homepage)
    targets_meta_path = os.path.join(download_dir, "commercial_ibx_targets.jsonl")
    for target in targets_list:
        add_jsonl_line(targets_meta_path, target)

    # Download commercial targets
    with open(targets_meta_path, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            print(f"Downloading {idx + 1}/{len(lines)} docs")
            rec = json.loads(line)
            html = requests.get(rec["url"]).text
            with open(os.path.join(download_dir, rec["policy_number"] + ".html"), "w") as f:
                f.write(html)

    # Scrape target links and compile in jsonl (Medicare Advantage)
    homepage = "https://medpolicy.ibx.com/ibc/ma/Pages/Policy-Bulletin-View.aspx"
    targets_list = scrape_policies(homepage)
    targets_meta_path = os.path.join(download_dir, "ma_ibx_targets.jsonl")
    for target in targets_list:
        add_jsonl_line(targets_meta_path, target)

    with open(targets_meta_path, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            print(f"Downloading {idx + 1}/{len(lines)} docs")
            rec = json.loads(line)
            html = requests.get(rec["url"]).text
            with open(os.path.join(download_dir, rec["policy_number"] + ".html"), "w") as f:
                f.write(html)

    # Note: no standard downloader here, as this data is not redistributable, or incorporated in our corpus.
