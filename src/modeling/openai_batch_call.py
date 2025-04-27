import argparse
import json
import os
import time

from openai import OpenAI

from src.util import add_jsonl_lines, get_records_list

OUTCOMES_DATASET = "test_backgrounds_suff"
# MODEL_KEY = "gpt-4o-2024-05-13"
# MODEL_KEY = "gpt-4o-2024-08-06"
MODEL_KEY = "gpt-4o-mini-2024-07-18"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
SYSTEM_MESSAGE = """You are an expert in U.S. health law and health policy, as well as a medical expert. In what follows, I will provide a description of a case in which a patient has submitted an appeal of a decision by their health insurer to deny a claim submitted on their behalf. You must predict whether an independent reviewer will overturn the denial, or uphold the denial when reviewing the appeal. If they would overturn it, your decision would be "Overturned". If they would not, your decision would be "Upheld". If there is insufficient information in the context to predict this, and it could go either way depending on more details, your decision would be "Insufficient".

Very short cases are often insufficient, as are cases which describe a treatment or service without saying what it is for. Sufficiency does not mean sufficient-without-a-doubt, it just means sufficient to make a good estimated guess about which way the review will go. Most cases I present to you will have sufficient information, by my subjective standards.

You must reply with json of the following form:
{"decision": "Overturned", "probability": 0.82}

with the decision, and the associated probability that that decision is correct. The possible decision
classes are "Insufficient", "Upheld", and "Overturned".

Here are two examples:

Prompt: "An enrollee has requested Zepatier for treatment of her hepatitis C."
Desired Output: {"decision": "Overturned", "probability": 0.75}

Prompt: "An enrollee has requested emergency services provided on an emergent or urgent basis for treatment of her medical condition."
Desired Output: {"decision": "Insufficient", "probability": 0.99}
"""


def construct_request_dict(
    case_description: str,
    custom_id: str,
    model_key: str,
    system_message: str,
) -> dict:
    prompt = f"{case_description}"

    if model_key in ["gpt-4o-2024-05-13"]:
        request = {
            "custom_id": f"{custom_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_key,
                "max_completion_tokens": 50,
                "response_format": {"type": "json_object"},  # Only some models allow for json mode to be specified
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
            },
        }
    else:
        request = {
            "custom_id": f"{custom_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_key,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
            },
        }
    return request


def construct_label(outcome, sufficiency_id):
    """Construct 3-class plain-text label from 2-class outcome and sufficiency score"""
    if sufficiency_id == 0:
        return "Insufficient"
    else:
        return outcome


def construct_answer_batch(recs: list[dict], output_path: str) -> None:
    answers = []
    for idx, rec in enumerate(recs):
        custom_id = idx
        outcome = rec["decision"]
        sufficiency_id = rec["sufficiency_id"]

        # Get metadata fields (default to "Unspecified" if not present)
        jurisdiction = rec.get("jurisdiction", "Unspecified")
        insurance_type = rec.get("insurance_type", "Unspecified")

        # Include all fields in the answer
        answers.append(
            {
                "custom_id": f"{custom_id}",
                "decision": construct_label(outcome, sufficiency_id),
                "sufficiency_id": sufficiency_id,
                "jurisdiction": jurisdiction,
                "insurance_type": insurance_type,
            }
        )

    add_jsonl_lines(output_path, answers)

    return None


def construct_request_batch(recs: list[dict], model_key: str, output_path: str) -> None:
    requests = []
    for idx, rec in enumerate(recs):
        custom_id = idx
        case_description = rec["text"]
        request = construct_request_dict(case_description, custom_id, model_key, SYSTEM_MESSAGE)
        requests.append(request)

    add_jsonl_lines(output_path, requests)

    return None


def prepare_batches() -> tuple:
    recs = get_records_list(os.path.join("./data/outcomes/", OUTCOMES_DATASET + ".jsonl"))
    output_path = f"./data/provider_annotated_outcomes/openai/{OUTCOMES_DATASET}/hicric_eval_request_answers.jsonl"
    if os.path.exists(output_path):
        print(f"Answer file already exists at {output_path}. Skipping this file construction.")
    else:
        construct_answer_batch(recs, output_path)

    output_path = f"./data/provider_annotated_outcomes/openai/{OUTCOMES_DATASET}/hicric_eval_request_{MODEL_KEY}.jsonl"
    subbatch_dir = (
        f"./data/provider_annotated_outcomes/openai/{OUTCOMES_DATASET}/hicric_eval_request_subbatches_{MODEL_KEY}"
    )
    if os.path.exists(output_path) and os.path.exists(subbatch_dir):
        print(f"Request file already exists at {output_path}. Skipping this file construction.")
    else:
        # Full batch
        construct_request_batch(recs, MODEL_KEY, output_path)
        split_batch(single_batch_path=output_path, subbatch_dir=subbatch_dir)
    return subbatch_dir, recs


def split_batch(single_batch_path, subbatch_dir, subbatch_size=1500) -> None:
    """Split a batch file into subbatches. Used to satisfy enqueued token limits."""
    os.makedirs(subbatch_dir, exist_ok=True)
    batch_requests = get_records_list(single_batch_path)

    for batch_idx, start_idx in enumerate(range(0, len(batch_requests), subbatch_size)):
        subbatch_requests = batch_requests[start_idx : start_idx + subbatch_size]
        add_jsonl_lines(os.path.join(subbatch_dir, f"{batch_idx}.jsonl"), subbatch_requests)
    return None


def batch_call(filepaths: list[str], api_key: str | None = OPENAI_API_KEY) -> list[str]:
    """Submit a batch completion call for each batch file in filepaths."""
    if not api_key:
        raise Exception("You need to export an Open AI API key as an env var.")
    client = OpenAI(api_key=api_key)

    # Upload subbatch input files
    print("Uploading and submitting batch files.")
    batch_file_ids = []
    batch_request_ids = []
    for path in filepaths:
        batchname = path.split("/")[-1]
        # Upload batch req file
        batch_input_file = client.files.create(file=open(path, "rb"), purpose="batch")
        batch_file_ids.append(batch_input_file.id)

        # Submit batch req file
        req_meta = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"{MODEL_KEY}-{OUTCOMES_DATASET}-hicric-eval-batch-{batchname}"
            },  # can be used to get batch req status by name
        )
        batch_request_ids.append(req_meta.id)
    print("Done.")

    # The below can be used to retrieve batches in separate one-time script, if don't want to poll for up to 24h,
    # as we do in what follows

    # batches = client.batches.list(limit=100).data
    # batches = [batch for batch in batches if "gpt-4o-2024-05-13-hicric-eval" in batch.metadata['description']]
    # batch_request_ids = [batch.id for batch in batches]

    return batch_request_ids


def download_response(batch_id: str, subbatch_dir: str, api_key: str | None = OPENAI_API_KEY) -> str:
    client = OpenAI(api_key=api_key)
    status_meta = client.batches.retrieve(batch_id)
    download_path = os.path.join(subbatch_dir, f"response_{batch_id}.jsonl")
    client.files.content(status_meta.output_file_id).write_to_file(download_path)
    return download_path


def poll_and_download(
    batch_request_ids: list[str], subbatch_dir: str, poll_sleep_mins: float = 0.1, api_key: str | None = OPENAI_API_KEY
) -> list[str]:
    client = OpenAI(api_key=api_key)

    completed_statuses = [0] * len(batch_request_ids)
    download_paths = []
    while sum(completed_statuses) < len(completed_statuses):
        for idx, id in enumerate(batch_request_ids):
            status_meta = client.batches.retrieve(id)
            if status_meta.status == "completed":
                completed_statuses[idx] = 1
                download_path = download_response(id, subbatch_dir)
                download_paths.append(download_path)

        print(f"{sum(completed_statuses)}/{len(completed_statuses)} batches processed.")
        if sum(completed_statuses) == len(completed_statuses):
            break
        else:
            print(f"Repolling in {poll_sleep_mins} minutes.")
            time.sleep(poll_sleep_mins * 60)
    return download_paths


def merge_jsonl(paths: list[str], output_path):
    all_recs = []
    for path in paths:
        recs = get_records_list(path)
        all_recs.extend(recs)

    add_jsonl_lines(output_path, all_recs)

    return None


def synchronous_call(records: list[dict], api_key: str | None = OPENAI_API_KEY) -> list[dict]:
    """Make synchronous API calls one at a time for each record."""
    if not api_key:
        raise Exception("You need to export an Open AI API key as an env var.")

    client = OpenAI(api_key=api_key)
    results = []

    print(f"Processing {len(records)} records synchronously...")
    for idx, rec in enumerate(records):
        case_description = rec["text"]
        custom_id = idx

        print(f"Processing record {idx+1}/{len(records)}")
        try:
            response = client.chat.completions.create(
                model=MODEL_KEY,
                messages=[{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": case_description}],
            )

            # Get the response content
            completion_text = response.choices[0].message.content

            # Parse JSON response
            try:
                completion_json = json.loads(completion_text)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for record {idx}: {completion_text}")
                completion_json = {"decision": "Error", "probability": 0}

            # Create result record
            result = {
                "custom_id": str(custom_id),
                "request": {
                    "body": {
                        "messages": [
                            {"role": "system", "content": SYSTEM_MESSAGE},
                            {"role": "user", "content": case_description},
                        ]
                    }
                },
                "response": {"choices": [{"message": {"content": completion_text, "role": "assistant"}}]},
                "parsed_response": completion_json,
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing record {idx}: {str(e)}")
            # Add error record
            results.append({"custom_id": str(custom_id), "error": str(e)})

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI API calls for health insurance claim evaluations")
    parser.add_argument("--synchronous", action="store_true", help="Use synchronous API calls instead of batch")
    args = parser.parse_args()

    subbatch_dir, records = prepare_batches()

    if args.synchronous:
        # Synchronous mode
        print("Running in synchronous mode...")
        results = synchronous_call(records)

        # Save results
        output_path = (
            f"./data/provider_annotated_outcomes/openai/{OUTCOMES_DATASET}/hicric_eval_response_sync_{MODEL_KEY}.jsonl"
        )
        add_jsonl_lines(output_path, results)
        print(f"Synchronous results saved to {output_path}")
    else:
        # Batch mode (original behavior)
        filenames = [os.path.join(subbatch_dir, filename) for filename in os.listdir(subbatch_dir)]

        # Can only upload small number of files at a time due to enque limit
        all_output_files = []
        batch_size = 1
        for start_idx in range(0, len(filenames), batch_size):
            enqueued_files = filenames[start_idx : start_idx + batch_size]
            batch_request_ids = batch_call(enqueued_files)
            output_files = poll_and_download(batch_request_ids, subbatch_dir)
            all_output_files.extend(output_files)

        # Merge to single response
        output_path = (
            f"./data/provider_annotated_outcomes/openai/{OUTCOMES_DATASET}/hicric_eval_response_{MODEL_KEY}.jsonl"
        )
        merge_jsonl(all_output_files, output_path)
        print(f"Batch results saved to {output_path}")
