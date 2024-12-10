import os
import typing as t
import xml.etree.ElementTree as ET

from src.util import add_jsonl_line, replace_path_component


def extract_text_from_xml(
    xml_file_path: str,
) -> t.Optional[tuple[str, t.Optional[str]]]:
    try:
        tree = ET.parse(xml_file_path)

        root = tree.getroot()
        text = ""

        # Traverse XML tree
        namespace = "{http://xml.house.gov/schemas/uslm/1.0}"

        title_tag = None
        for element in root.iter():
            # TODO: extract additional metadata tags from the rich XML metadata
            if element.text:
                text += element.text.strip() + " "
            if element.tag == f"{namespace}title":
                title_tag = element.attrib.get("identifier")

        return text.strip(), title_tag

    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None


def process(source_lineitem: dict, output_dirname: str) -> dict:
    """Processor for xml distributables for individual USC titles"""
    local_path = source_lineitem.get("local_path", None)
    filename = os.path.basename(local_path)
    local_dir = os.path.dirname(local_path)
    local_processed_dir = replace_path_component(local_dir, 2, output_dirname)

    xml_text, title_tag = extract_text_from_xml(local_path)

    # Add additional tag based on title
    source_lineitem["tags"] = source_lineitem["tags"] + [title_tag]

    # Process and write jsonl to output file

    outfile = os.path.join(local_processed_dir, filename + ".jsonl")
    line_data = {"text": xml_text}

    add_jsonl_line(outfile, line_data)

    # Construct updated lineitem
    source_lineitem["local_processed_path"] = outfile

    return source_lineitem
