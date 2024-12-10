# Minimally Modified from Pile of Law: https://github.com/Breakend/PileOfLaw/blob/main/dataset_creation/federal_register/process_federal_register.py

# Processes federal register proposed rules (2016 - 2023)
# Bulk data pulled from: https://www.govinfo.gov/bulkdata/xml/FR

import datetime
import json
import os
from urllib.request import Request, urlopen

from dateutil import parser

# xpath only available in lxml etree, not ElementTree
from lxml import etree
from tqdm import tqdm

from src.util import add_jsonl_line, gen_src_metadata, generate_file_md5

# Request variables
headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/xml"}


def save_to_file(data, out_dir, fname):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w") as out_file:
        for x in data:
            out_file.write(json.dumps(x) + "\n")
    print(f"Written {len(data)} to {fpath}")


def request_raw_data(init_url):
    urls = [init_url]
    xmls = []
    while len(urls) > 0:
        next_urls = []
        for url in urls:
            print(url)
            request = Request(url, headers=headers)
            with urlopen(request) as response:
                root = etree.fromstring(response.read())
            elems = root.xpath("*/file[folder='true' and name!='resources']")
            if len(elems) > 0:
                for e in elems:
                    next_url = e.find("link").text
                    next_urls.append(next_url)
            else:
                elems = root.xpath("*/file[mimeType='application/xml']")
                for e in elems:
                    xml_url = e.find("link").text
                    request = Request(xml_url, headers=headers)
                    with urlopen(request) as response:
                        xml = etree.fromstring(response.read())
                    # Add tuple of xml_url, xml Element instance
                    xmls.append((xml_url, xml))
        urls = next_urls

    return xmls


def extract_rule_docs(xmls):
    docs = []
    for xml_url, xml in tqdm(xmls):
        print(xml_url)
        date = xml.find("DATE").text
        creation_date = ""
        try:
            creation_date = parser.parse(date).strftime("%m-%d-%Y")
        except Exception as e:
            print(f"Issue parsing date from federal regulation XML: {e}")
        proposed_rules = xml.xpath("PRORULES/PRORULE")
        for rule in proposed_rules:
            all_text = etree.tostring(rule, encoding="unicode", method="text")

        doc = {
            "url": xml_url,
            "created_timestamp": creation_date,
            "downloaded_timestamp": datetime.date.today().strftime("%m-%d-%Y"),
            "text": all_text,
        }
        docs.append(doc)

    return docs


def download(out_dir, outfile):
    # Request raw data directly using bulk data API

    years = [2024, 2023, 2022, 2021, 2020]
    for year in years:
        init_url = f"https://www.govinfo.gov/bulkdata/xml/FR/{year}"
        xmls = request_raw_data(init_url)
        docs = extract_rule_docs(xmls)
        fname = f"federal_register_{year}.jsonl"

        save_to_file(docs, out_dir, fname)

        # Document metadata
        tags = ["fr", "kb", "legal"]
        proc = None
        filepath = os.path.join(out_dir, fname)
        hash = generate_file_md5(filepath)
        lineitem = gen_src_metadata(init_url, filepath, tags, proc, hash)
        add_jsonl_line(outfile, lineitem)

    return None
