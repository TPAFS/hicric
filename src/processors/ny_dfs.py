import os

import pandas as pd

from src.util import add_jsonl_line, replace_path_component


def process(source_lineitem: dict, output_dirname: str) -> dict:
    """Processor for NYS DFS data."""
    local_path = source_lineitem.get("local_path", None)
    filename = os.path.splitext(os.path.basename(local_path))[0]
    local_dir = os.path.dirname(local_path)
    local_processed_dir = replace_path_component(local_dir, 2, output_dirname)

    # Process and write jsonl to output file
    outfile = os.path.join(local_processed_dir, filename + ".jsonl")

    def extract_row_summaries(row) -> list[str]:
        summaries = []
        for col in ["Summary 1", "Summary 2", "Summary 3"]:
            sum = row.get(col)
            if not pd.isna(sum):
                summaries.append(sum)
        return summaries

    df = pd.read_excel(local_path)

    df_tuples = df.apply(
        lambda row: (
            extract_row_summaries(row),
            row["Appeal Decision"],
            row["Coverage Type"],
            row["Denial Reason"],
            row["Diagnosis"].strip("[]"),
            row["Treatment"].strip("[]"),
        ),
        axis=1,
    ).to_list()

    coverage_type_map = {
        "HMO": "Commercial",
        "PPO": "Commercial",
        "EPO": "Commercial",
        "Self-Funded": "Commercial",
        "Medicaid": "Medicaid",
        "Managed Long Term Care": "Medicaid",
    }

    # Construct a record for each case summary
    # (some raw case adjudications include multiple summaries from different reviewers)
    for tuple in df_tuples:
        summaries = tuple[0]
        for summary in summaries:
            line_data = {
                "text": summary,
                "decision": tuple[1],
                "coverage_type": tuple[2],
                "appeal_type": tuple[3],
                "diagnosis": tuple[4],
                "treatment": tuple[5],
                "jurisdiction": "NY",
                "insurance_type": coverage_type_map.get(tuple[2], "Unspecified"),
            }
            add_jsonl_line(outfile, line_data)

    # Construct updated lineitem
    source_lineitem["local_processed_path"] = outfile

    return source_lineitem
