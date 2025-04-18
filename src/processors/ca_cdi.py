import os

import pandas as pd

from src.util import add_jsonl_line, extract_pdf_text, replace_path_component


def extract_summary_substrings(input_string):
    """
    Extracts substrings starting with "Summary" from the input string.

    Args:
        input_string (str): The input string containing multiple occurrences of "Summary".

    Returns:
        list: A list of strings corresponding to each substring that starts with "Summary."
    """
    # Split the input string using "Summary" as the delimiter
    summary_substrings = input_string.split("Summary")

    # Remove any empty strings resulting from the split
    summary_substrings = ["Summary" + substring for substring in summary_substrings if substring.strip()]

    return summary_substrings


def process(source_lineitem: dict, output_dirname: str) -> dict:
    """Processor for ca_cdi files."""
    local_path = source_lineitem.get("local_path", None)
    filename, ext = os.path.splitext(os.path.basename(local_path))
    local_dir = os.path.dirname(local_path)
    local_processed_dir = replace_path_component(local_dir, 2, output_dirname)

    # Get text
    if ext == ".txt":
        with open(local_path, "r", encoding="ISO-8859-1") as file:
            text = file.read()
    elif ext == ".pdf":
        text = extract_pdf_text(local_path)

    # Read associated metadata
    # TODO: cache the dataframe read
    df = pd.read_csv("./data/raw/ca_cdi/imr_report.csv", encoding="ISO-8859-1")
    meta_rec = df[df["Reference Number"].str.strip() == filename].iloc[0].to_dict()
    appeal_type = meta_rec["IMR Type"].strip()
    diagnosis = meta_rec["Diagnosis Subcategory"].strip()
    treatment = meta_rec["Treatment Subcategory"].strip()
    decision = meta_rec["Outcome"].strip()
    patient_race = meta_rec["Race"].strip()

    # Create a distinct record for each reviewer's summary
    # and write them as lines to outfile
    outfile = os.path.join(local_processed_dir, filename + ".jsonl")
    summaries = extract_summary_substrings(text)
    if len(summaries) == 0:
        # Error extracting text from some source pdfs
        raise Exception(f"Extraction failed for source record: {source_lineitem} and text: {text}")
    for summary in summaries:
        line_data = {
            "text": summary,
            "appeal_type": appeal_type,
            "diagnosis": diagnosis,
            "treatment": treatment,
            "decision": decision,
            "patient_race": patient_race,
            "jurisdiction": "CA",
            "insurance_type": "Commercial",
        }
        add_jsonl_line(outfile, line_data)

    # Construct updated lineitem
    source_lineitem["local_processed_path"] = outfile

    return source_lineitem
