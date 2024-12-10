import os
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.util import add_jsonl_line, gen_src_metadata, generate_file_md5


def download_case_summaries(download_dir: str):
    options = Options()
    # Set download directory
    download_dir = "./data/raw/ca_cdi/summaries"
    download_dir = os.path.abspath(download_dir)
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
    options.add_argument("--headless")

    driver = webdriver.Firefox(options=options)

    # Navigate to the website
    url = "https://interactive.web.insurance.ca.gov/apex_extprd/f?p=192:1:15394748574802::NO:RP,1::"
    driver.get(url)

    # Wait for the page to load (adjust the sleep time according to your page loading time)
    time.sleep(5)

    # Find the button by text
    search_button = driver.find_element(By.XPATH, "//button/span[contains(text(), 'Search')]")

    # Scroll to the element
    driver.execute_script("arguments[0].scrollIntoView(true);", search_button)

    # Click the button using JavaScript
    driver.execute_script("arguments[0].click();", search_button)

    more_pages = True
    while more_pages:
        # Wait for the "details" header element to be present, then select it
        details_header = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(text(),'Details')]"))
        )

        # Get the index of the column containing "Details"
        details_column_index = details_header.get_attribute("data-column")

        # Wait for the "Reference Number" header element to be present, then select it
        reference_number_header = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(text(),'Reference Number')]"))
        )

        # Get the index of the column containing "Reference Number"
        reference_number_column_index = reference_number_header.get_attribute("data-column")
        time.sleep(3)  # Wait for ref numbers to populate

        # Find all the strings in the "Reference Number" column
        reference_numbers = driver.find_elements(By.XPATH, f"//td[@headers='C{reference_number_column_index}']")

        # Extract the text from the reference number elements
        reference_numbers_texts = [reference_number.text for reference_number in reference_numbers]

        # # Find all the links in the "Details" column
        detail_links = driver.find_elements(By.XPATH, f"//td[@headers='C{details_column_index}']/a")

        # Zip detail links and reference numbers together
        details_and_references = zip(detail_links, reference_numbers_texts)

        # Iterate over each link in the "Details" column
        for link, ref in details_and_references:
            new_filename_candidates = [f"{ref}.txt", f"{ref}.pdf"]
            new_filepath_candidates = [
                os.path.join(download_dir, new_filename) for new_filename in new_filename_candidates
            ]

            # In case no download exists for case ref
            existing_files = [os.path.exists(filepath) for filepath in new_filepath_candidates]
            if not any(existing_files):
                link.click()

                # Wait for the popup to appear
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[@role='dialog']")))

                # Enter iframe context
                iframe = driver.find_element(By.TAG_NAME, "iframe")
                driver.switch_to.frame(iframe)
                time.sleep(1)

                popup_links = driver.find_elements(By.CSS_SELECTOR, "[alt='Download']")
                if len(popup_links) != 2:  # Unknown context
                    # Exit iframe context
                    driver.switch_to.default_content()
                    # Close the popup and move on regardless
                    close_button = driver.find_element(By.XPATH, "//button[@title='Close']")
                    close_button.click()
                    continue

                # Second link is summary, click triggers download
                popup_links[1].click()

                # Attempt to wait for slow download
                time.sleep(2)
                expected_filenames = [
                    filename for filename in os.listdir(download_dir) if "Summary" in filename or "summary" in filename
                ]
                if len(expected_filenames) != 1:
                    print("No download present with expected name patterns.")
                    # Exit iframe context
                    driver.switch_to.default_content()
                    # Close the popup and move on regardless
                    close_button = driver.find_element(By.XPATH, "//button[@title='Close']")
                    close_button.click()

                    continue
                else:
                    # Rename file
                    original_filename = expected_filenames[0]
                    original_filepath = os.path.join(download_dir, original_filename)
                    new_filename = f"{ref}{os.path.splitext(original_filename)[1]}"
                    new_file_path = os.path.join(download_dir, new_filename)
                    os.rename(original_filepath, new_file_path)

                # Exit iframe context
                driver.switch_to.default_content()
                # Close the popup and move on regardless
                close_button = driver.find_element(By.XPATH, "//button[@title='Close']")
                close_button.click()

            else:
                print(f"Skipping previously downloaded file at: {new_filepath_candidates[existing_files.index(True)]}")

        # Find the next page button and click it
        next_page_button = driver.find_element(By.XPATH, "//button[@title='Next']")
        if next_page_button:
            next_page_button.click()

            # Wait for the next page to load
            time.sleep(1)
        else:
            more_pages = False

    # Close the browser window
    driver.quit()


def download(output_dir: str, source_meta_path: str) -> None:
    print("Downloading CA-DOI IMR Cases.")
    download_dir = os.path.abspath(output_dir)

    # Do the actual downloads
    download_case_summaries(download_dir)

    # Produce source metadata for each file

    # Collect all the pdfs in the download dir
    downloaded_files = [os.path.join(output_dir, filename) for filename in os.listdir(download_dir)]
    pdf_paths = [file for file in downloaded_files if os.path.splitext(file)[1] == ".pdf"]
    txt_paths = [file for file in downloaded_files if os.path.splitext(file)[1] == ".txt"]

    for path in txt_paths + pdf_paths:
        # Hash file
        hash = generate_file_md5(path)

        # Produce source metadata
        url = "https://interactive.web.insurance.ca.gov/apex_extprd/f?p=192:1:15394748574802::NO:RP,1::"
        processor = "ca_cdi"
        tags = [
            "california",
            "independent-medical-review",
            "cdi",
            "case-description",
        ]
        source_meta = gen_src_metadata(url, path, tags, processor, hash)

        # Log meta
        add_jsonl_line(source_meta_path, source_meta)

    return None
