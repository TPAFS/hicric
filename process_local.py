import multiprocessing
import os
import shutil
from functools import partial

from src.processors import PROCESSORS
from src.util import (
    add_jsonl_line,
    normalize_unicode,
    read_jsonl_batched,
    replace_path_component,
    update_jsonl_key,
)


class ProcessingError(Exception):
    pass


def process_record(json_rec: dict, output_dirname: str) -> dict:
    """Process json_record source entry, and return processed metadata record."""
    processor_key = json_rec["preprocessor"]

    # Process Raw Download
    if processor_key is None:
        # Indicates processing was computed at download time
        # So just copy source record
        # Note: also copy "raw" source file to processed dir, for convenience in archiving entire dataset w/o symlinks
        processed_line = json_rec
        raw_path = json_rec["local_path"]
        processed_path = replace_path_component(raw_path, 2, output_dirname)
        # If necessary, make a dir
        if os.path.split(processed_path)[0]:
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        shutil.copyfile(raw_path, processed_path)
        processed_line["local_processed_path"] = processed_path

    # Use bespoke processor
    else:
        processor = PROCESSORS.get(json_rec["preprocessor"], None)

        if not processor:
            raise ValueError("Invalid processor in line.")

        try:
            processed_line = processor(json_rec, output_dirname)

        except Exception as e:
            raise ProcessingError(f"Error: {e}")
            # print(f"Error: {e}")

    # Common to all processors:

    # Fix any errant encoding issues
    processed_path = processed_line["local_processed_path"]
    update_jsonl_key(processed_path, "text", normalize_unicode)

    return processed_line


if __name__ == "__main__":
    jsonl_file_path = "sources.jsonl"
    outfile = "processed_sources.jsonl"
    output_dirname = "processed"

    process_single_record = partial(process_record, output_dirname=output_dirname)

    records_to_process = sum(1 for line in open(jsonl_file_path))

    # Sequential for debugging
    # import json

    # with open(jsonl_file_path) as file:
    #     for idx, line in enumerate(file):
    #         print(f"{idx} / {records_to_process} downloads processed.")
    #         rec = json.loads(line)
    #         try:
    #             process_record(rec, output_dirname)
    #         except ProcessingError as e:
    #             print(f"Error processing line {idx + 1}: {rec}")
    #             print(e)

    # Parallel
    processes = batch_size = 24
    for batch_idx, batch in enumerate(read_jsonl_batched(jsonl_file_path, batch_size)):
        print(f"{batch_idx*batch_size} / {records_to_process} downloads processed.")
        # process files in batch in parallel
        with multiprocessing.Pool(processes) as pool:
            try:
                processed_lines = pool.map(process_single_record, batch)
            except:
                raise
            [add_jsonl_line(outfile, processed_line) for processed_line in processed_lines]
