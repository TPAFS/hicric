import json
from collections import defaultdict

from src.util import generate_file_md5


def add_missing_hashes(jsonl_path: str) -> None:
    missing_hash_count = 0
    with open(jsonl_path, "r+", encoding="utf-8") as jsonl_file:
        lines = jsonl_file.readlines()

        # Move cursor back to beginning
        jsonl_file.seek(0)

        for line in lines:
            # Get metadata record
            json_data = json.loads(line)

            # Extract relevant keys from record
            file_path = json_data.get("local_path", None)
            hash = json_data.get("md5", None)

            if not hash:
                missing_hash_count += 1
                print("No hash, generating...")
                hash = generate_file_md5(file_path)
                json_data["md5"] = hash
                print("Done")

            # Update the line
            jsonl_file.write(json.dumps(json_data) + "\n")

    print(f"Generated {missing_hash_count} hashes.")


def add_missing_tags(jsonl_path: str) -> None:
    with open(jsonl_path, "r+", encoding="utf-8") as jsonl_file:
        lines = jsonl_file.readlines()

        # Move cursor back to beginning
        jsonl_file.seek(0)

        for line in lines:
            # Get metadata record
            json_data = json.loads(line)

            # Extract relevant keys from record
            tags = json_data.get("tags", None)
            path = json_data.get("local_path")

            if "./data/raw/ca_cdi/" in path:
                to_add = "case-description"
                if to_add not in tags:
                    tags.append(to_add)

            if "./data/raw/ny_dfs" in path:
                to_add = "case-description"
                if to_add not in tags:
                    tags.append(to_add)

            if "./data/raw/ca_dmhc" in path:
                to_add = "case-description"
                if to_add not in tags:
                    tags.append(to_add)

            if "./data/raw/contracts" in path:
                to_add = "contract-coverage-rule-medical-policy"
                if to_add not in tags:
                    tags.append(to_add)

            # if "./data/raw/hhs_oig" in path:
            #     to_add = "opinion-policy-summary"
            #     if to_add not in tags:
            #         tags.append(to_add)

            if "./data/raw/medicare" in path:
                to_add = "contract-coverage-rule-medical-policy"
                if to_add not in tags:
                    tags.append(to_add)

            if "./data/raw/medicare_qic" in path:
                to_add = "case-description"
                if to_add not in tags:
                    tags.append(to_add)

            if "./data/raw/opinion_policy_summary" in path:
                to_add = "opinion-policy-summary"
                if to_add not in tags:
                    tags.append(to_add)

            if "./data/raw/regulatory_guidance" in path:
                to_add = "regulatory-guidance"
                if to_add not in tags:
                    tags.append(to_add)

            json_data["tags"] = tags

            # Update the line
            jsonl_file.write(json.dumps(json_data) + "\n")


def remove_empty_tags(jsonl_path: str) -> None:
    with open(jsonl_path, "r+", encoding="utf-8") as jsonl_file:
        lines = jsonl_file.readlines()

        # Move cursor back to beginning
        jsonl_file.seek(0)

        for line in lines:
            # Get metadata record
            json_data = json.loads(line)

            # Extract relevant keys from record
            tags = json_data.get("tags", None)

            updated_tags = [tag for tag in tags if len(tag) > 0]
            json_data["tags"] = updated_tags

            # Update the line
            jsonl_file.write(json.dumps(json_data) + "\n")


def list_duplicates(seq):
    """Copied from: https://stackoverflow.com/a/5419576"""
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)


class UniqueSourceError(Exception):
    pass


def get_all_hashes(jsonl_path: str) -> list[str]:
    all_hashes = []
    with open(jsonl_path, "r+", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            # Get metadata record
            json_data = json.loads(line)

            # Extract relevant keys from record
            hash = json_data.get("md5", None)
            all_hashes.append(hash)
    return all_hashes


def verify_hash_uniqueness(jsonl_path: str) -> None:
    all_hashes = get_all_hashes(jsonl_path)

    if len(all_hashes) != len(set(all_hashes)):
        raise UniqueSourceError()


def remove_duplicate_lines(jsonl_path: str) -> None:
    all_hashes = get_all_hashes(jsonl_path)
    drop_indices = []
    for dup in sorted(list_duplicates(all_hashes)):
        _keep, drops = dup[1][0], dup[1][1:]
        drop_indices.extend(drops)

    with open(jsonl_path, "r+", encoding="utf-8") as jsonl_file:
        lines = jsonl_file.readlines()

        jsonl_file.seek(0)

        for idx, line in enumerate(lines):
            if idx not in drop_indices:
                jsonl_file.write(line)

        jsonl_file.truncate()


if __name__ == "__main__":
    jsonl_file_path = "sources.jsonl"
    # remove_empty_tags(jsonl_file_path)
    verify_hash_uniqueness(jsonl_file_path)
    # add_missing_tags(jsonl_file_path)
