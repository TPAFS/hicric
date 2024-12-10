import html
import os

import pandas as pd

from src.util import add_jsonl_line, replace_path_component


def process(source_lineitem: dict, output_dirname: str) -> dict:
    """Processor for CA DMHC data."""
    local_path = source_lineitem.get("local_path", None)
    filename = os.path.splitext(os.path.basename(local_path))[0]
    local_dir = os.path.dirname(local_path)
    local_processed_dir = replace_path_component(local_dir, 2, output_dirname)

    # Process and write jsonl to output file
    outfile = os.path.join(local_processed_dir, filename + ".jsonl")

    df = pd.read_csv(local_path, encoding="ISO-8859-1")

    df_tuples = df.apply(
        lambda row: (
            row["Findings"],
            row["Determination"],
            row["Type"],
            row["IMRType"],
        ),
        axis=1,
    ).to_list()
    # TODO: Add "DiagnosisCategory", "DiagnosisSubCategory',
    # "TreatmentCategory", "TreatmentSubCategory" to tags after parsing

    for tuple in df_tuples:
        line_data = {
            "text": html.unescape(tuple[0]),
            "decision": tuple[1],
            "appeal_type": tuple[2],
            "appeal_expedited_status": tuple[3],
        }
        add_jsonl_line(outfile, line_data)

    # Construct updated lineitem
    source_lineitem["local_processed_path"] = outfile

    return source_lineitem
