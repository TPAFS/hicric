import os

from src.util import decompress_zip, medicare_cd_to_jsonl, replace_path_component


def process(source_lineitem: dict, output_dirname: str) -> dict:
    """Processor for Medicare coverage determinations."""

    # Decompress the zip with the coverage determination csvs
    local_path = source_lineitem.get("local_path", None)
    local_dir, _local_file = os.path.split(local_path)
    decompressed_folder = decompress_zip(local_path, local_dir)

    # Specify the useful csv / relevant parsing
    cd_filename = local_path.split("/")[-2]

    if cd_filename == "current_lcd":
        cd_file = os.path.join(os.path.split(decompressed_folder)[0], "lcd.csv")
        cois = [
            "indication",
            "summary_of_evidence",
            "analysis_of_evidence",
        ]  # columns of interest in csv

    elif cd_filename == "ncd":
        cd_file = os.path.join(decompressed_folder, "ncd_trkg.csv")
        cois = ["indctn_lmtn"]  # columns of interest in csvs

    elif cd_filename == "current_article":
        cd_file = os.path.join(os.path.split(decompressed_folder)[0], "article.csv")
        cois = ["title", "description"]  # columns of interest in csvs

    # Process and write jsonl to output file
    local_processed_dir = replace_path_component(local_dir, 2, output_dirname)
    outfile = os.path.join(local_processed_dir, f"{cd_filename}.jsonl")
    medicare_cd_to_jsonl(cd_file, cois, outfile)

    # Construct updated lineitem
    source_lineitem["local_processed_path"] = outfile

    return source_lineitem
