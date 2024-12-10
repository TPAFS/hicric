import os

from src.util import add_jsonl_line, replace_path_component


def process(source_lineitem: dict, output_dirname: str) -> dict:
    """Processor for generic text file encoded via ISO-8859-1."""
    local_path = source_lineitem.get("local_path", None)
    filename = os.path.splitext(os.path.basename(local_path))[0]
    local_dir = os.path.dirname(local_path)
    local_processed_dir = replace_path_component(local_dir, 2, output_dirname)

    with open(local_path, "r", encoding="ISO-8859-1") as file:
        text = file.read()

    # Process and write jsonl to output file
    outfile = os.path.join(local_processed_dir, filename + ".jsonl")
    line_data = {"text": text}
    add_jsonl_line(outfile, line_data)

    # Construct updated lineitem
    source_lineitem["local_processed_path"] = outfile

    return source_lineitem
