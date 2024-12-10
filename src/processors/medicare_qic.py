import os

import pandas as pd

from src.util import add_jsonl_line, replace_path_component


def process(source_lineitem: dict, output_dirname: str) -> dict:
    """Processor for medicare QIC data."""
    local_path = source_lineitem.get("local_path", None)
    filename = os.path.splitext(os.path.basename(local_path))[0]
    local_dir = os.path.dirname(local_path)
    local_processed_dir = replace_path_component(local_dir, 2, output_dirname)

    # Process and write jsonl to output file
    outfile = os.path.join(local_processed_dir, filename + ".jsonl")

    df = pd.read_csv(local_path)

    # Only consider rows with non-empty decision_rationales
    df = df[~df["decision_rationale"].isna()]

    df_tuples = df.apply(
        lambda row: (row["decision_rationale"], row["decision"], row["appeal_type"]),
        axis=1,
    ).to_list()

    for tuple in df_tuples:
        line_data = {"text": tuple[0], "decision": tuple[1], "appeal_type": tuple[2]}
        add_jsonl_line(outfile, line_data)

    # Coverage explanations are duplcated often, so we separate them
    # to avoid highly redundant training text. This leaves some coverage
    # explanation/tag associations on the table that might be useful.
    coverage_explanations = df["coverage_rules"].dropna().unique()
    for explanation in coverage_explanations:
        line_data = {"text": explanation, "tags": ["coverage-explanation"]}
        add_jsonl_line(outfile, line_data)

    # Construct updated lineitem
    source_lineitem["local_processed_path"] = outfile

    return source_lineitem
