import json
import os

from src.util import add_jsonl_lines, replace_path_component


def construct_hicric_jsonl(ex):
    """Construct actual data jsonl, with instance-level tags."""
    url = ex["url"]
    title = ex["title"]
    source = ex["source"]
    tags = []

    rec = {"url": None}

    if source is not None:
        tags.append(source)

    if url is not None:
        rec["url"] = url

    if title not in [None, "None"]:
        # TODO: Determine how best to incorporate title into processed schema
        # For now, heuristic
        if len(title) < 10:
            tags.append(title.lower())

    rec["tags"] = tags
    rec["text"] = ex["clean_text"]

    return rec


def process(source_lineitem: dict, output_dirname: str) -> dict:
    local_path = source_lineitem.get("local_path", None)
    filename = os.path.splitext(os.path.basename(local_path))[0]
    local_dir = os.path.dirname(local_path)
    local_processed_dir = replace_path_component(local_dir, 2, output_dirname)

    # Process and write jsonl to output file
    outfile = os.path.join(local_processed_dir, filename + ".jsonl")

    # Load meditron jsonl
    with open(local_path, "r") as f:
        ldata = list(map(json.loads, f))

    # Convert to hicric format
    hicric_jsonl = list(map(construct_hicric_jsonl, ldata))

    add_jsonl_lines(outfile, hicric_jsonl)

    # Construct updated lineitem
    source_lineitem["local_processed_path"] = outfile

    return source_lineitem
