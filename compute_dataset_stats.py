import json
import os
from collections import defaultdict


def compute_word_char_count(jsonl_path: str) -> tuple[int, int]:
    """Get word count for all words associated with a given processed source."""
    total_words = 0
    total_chars = 0

    with open(jsonl_path, "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            try:
                json_data = json.loads(line)
                text = json_data.get("text", "")
                total_words += len(text.split())
                total_chars += len(text)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line: {line}")

    return total_words, total_chars


def get_stats_by_partition_set(jsonl_path: str, partition_set) -> None:
    doc_count = 0

    # Define the default values for stats
    stats_initialization = {
        "docs": 0,
        "words": 0,
        "chars": 0,
        "size": 0,
        "kb_docs": 0,
        "kb_words": 0,
        "kb_chars": 0,
        "kb_size": 0,
    }
    stats_dict = defaultdict(lambda: stats_initialization.copy())

    with open(jsonl_path, "r+", encoding="utf-8") as jsonl_file:
        lines = jsonl_file.readlines()

        # Move cursor back to beginning
        jsonl_file.seek(0)

        for line in lines:
            try:
                # Get metadata record
                json_data = json.loads(line)

                # Extract relevant keys from record
                processed_jsonl_path = json_data.get("local_processed_path", None)
                tags = json_data.get("tags", [])

                # Process the referenced data, update global and tag stats
                if processed_jsonl_path:
                    file_size = os.stat(processed_jsonl_path).st_size  # bytes
                    total_words, total_chars = compute_word_char_count(processed_jsonl_path)

                    for tag in partition_set:
                        if tag in tags:
                            stats_dict[tag]["docs"] += 1
                            stats_dict[tag]["words"] += total_words
                            stats_dict[tag]["chars"] += total_chars
                            stats_dict[tag]["size"] += file_size

                            # Does lineitem contribute to KB stats?
                            if "kb" in tags:
                                stats_dict[tag]["kb_docs"] += 1
                                stats_dict[tag]["kb_words"] += total_words
                                stats_dict[tag]["kb_chars"] += total_chars
                                stats_dict[tag]["kb_size"] += file_size
                            break

                else:
                    print("No 'local_processed_path' key found in JSON line.")
                doc_count += 1

            except json.JSONDecodeError:
                print(f"Error decoding JSON in line: {line}")
    return dict(stats_dict)


def construct_stats_dict(jsonl_path: str) -> dict:
    doc_count = 0

    # Define the default values for stats
    stats_initialization = {"docs": 0, "words": 0, "chars": 0, "size": 0}
    stats_dict = defaultdict(lambda: stats_initialization.copy())

    with open(jsonl_path, "r+", encoding="utf-8") as jsonl_file:
        lines = jsonl_file.readlines()

        # Move cursor back to beginning
        jsonl_file.seek(0)

        for line in lines:
            try:
                # Get metadata record
                json_data = json.loads(line)

                # Extract relevant keys from record
                processed_jsonl_path = json_data.get("local_processed_path", None)
                tags = json_data.get("tags", [])

                # Process the referenced data, update global and tag stats
                if processed_jsonl_path:
                    # Check if stats have already been written for processed source
                    stats = json_data.get("stats", None)

                    if not stats:
                        file_size = os.stat(processed_jsonl_path).st_size  # bytes
                        total_words, total_chars = compute_word_char_count(processed_jsonl_path)
                        json_data["stats"] = {
                            "size": file_size,
                            "words": total_words,
                            "chars": total_chars,
                        }
                    else:
                        file_size = stats.get("size")  # bytes
                        total_words = stats.get("words")
                        total_chars = stats.get("chars")

                    # Re-write/update the line
                    jsonl_file.write(json.dumps(json_data) + "\n")

                    # Update tag stats
                    for tag in tags:
                        stats_dict[tag]["docs"] += 1
                        stats_dict[tag]["words"] += total_words
                        stats_dict[tag]["chars"] += total_chars
                        stats_dict[tag]["size"] += file_size

                    # Update global stats
                    global_key = "_global"
                    stats_dict[global_key]["docs"] += 1
                    stats_dict[global_key]["words"] += total_words
                    stats_dict[global_key]["chars"] += total_chars
                    stats_dict[global_key]["size"] += file_size
                else:
                    print("No 'local_processed_path' key found in JSON line.")
                doc_count += 1

            except json.JSONDecodeError:
                print(f"Error decoding JSON in line: {line}")

    return stats_dict


def output_markdown_table(stats_dict: dict, tags: list, kb_only: bool = False) -> str:
    # Initialize markdown table string
    markdown_table = "| Category | Num Documents | Words | Chars | Size (GB) |\n"
    markdown_table += "| -------- | ------------- | ----- | ----- | --------- |\n"

    # Iterate over the tags
    for tag in tags:
        # Check if the tag exists in the stats_dict
        if tag in stats_dict:
            stats = stats_dict[tag]
            # Add data to the markdown table string
            if kb_only:
                markdown_table += f"| {tag} | {stats['kb_docs']:,} | {stats['kb_words']:,} | {stats['kb_chars']:,} | {stats['kb_size'] / 1e9:.2f} |\n"
            else:
                markdown_table += f"| {tag} | {stats['docs']:,} | {stats['words']:,} | {stats['chars']:,} | {stats['size'] / 1e9:.2f} |\n"

    return markdown_table


def output_latex_table(stats_dict: dict, tags: list, kb_only: bool = False) -> str:
    # Initialize LaTeX table string
    latex_table = "\\begin{table}[htbp]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{|c|c|c|c|c|}\n"
    latex_table += "\\hline\n"
    latex_table += (
        "\\textbf{Category} & \\textbf{Num Documents} & \\textbf{Words} & \\textbf{Chars} & \\textbf{Size (GB)} \\\\\n"
    )
    latex_table += "\\hline\n"

    # Iterate over the tags
    for tag in tags:
        # Check if the tag exists in the stats_dict
        if tag in stats_dict:
            stats = stats_dict[tag]
            # Add data to the LaTeX table string
            if kb_only:
                latex_table += f"{tag} & {stats['kb_docs']:,} & {stats['kb_words']:,} & {stats['kb_chars']:,} & {stats['kb_size'] / 1e9:.2f} \\\\\n"
            else:
                latex_table += f"{tag} & {stats['docs']:,} & {stats['words']:,} & {stats['chars']:,} & {stats['size'] / 1e9:.2f} \\\\\n"
            latex_table += "\\hline\n"

    # Add LaTeX table footer
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Statistics}\n"
    latex_table += "\\label{tab:my_label}\n"
    latex_table += "\\end{table}\n"

    return latex_table

    # Now compute partition breakdowns by

    return None


if __name__ == "__main__":
    VERBOSE = True
    jsonl_path_to_iterate = "./processed_sources.jsonl"
    stats_dict = construct_stats_dict(jsonl_path_to_iterate)

    if VERBOSE:
        # Print stats for each tag
        # for key, stats in stats_dict.items():
        #     print(f"Stats for tag={key}")
        #     print(f"\tDocuments: {stats['docs']:,}")
        #     print(f"\tWords: {stats['words']:,}")
        #     print(f"\tChars: {stats['chars']:,}")
        #     print(f"\tSize: {stats['size']:,} ({stats['size'] / 1e9} GB)\n")

        # Print stats for partition tags as markdown, latex tables
        partition_tags = [
            "legal",
            "regulatory-guidance",
            "contract-coverage-rule-medical-policy",
            "opinion-policy-summary",
            "case-description",
            "clinical-guidelines",
        ]
        markdown_table = output_markdown_table(stats_dict, tags=["_global", "kb"] + partition_tags)
        print(markdown_table)

        latex_table = output_latex_table(stats_dict, tags=["_global", "kb"] + partition_tags)
        print(latex_table)

        # Print stats for just KB across partition
        # TODO: remove redundant code, do this efficiently with single pass
        # full_stats_dict = get_stats_by_partition_set(
        #     jsonl_path_to_iterate, partition_tags
        # )
        # markdown_table = output_markdown_table(
        #     full_stats_dict, tags=["kb"] + partition_tags, kb_only=True
        # )
        # print(markdown_table)

        # latex_table = output_latex_table(
        #     full_stats_dict, tags=["kb"] + partition_tags, kb_only=True
        # )
        # print(latex_table)
