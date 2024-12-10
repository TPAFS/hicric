import pandas as pd

from src.util import add_jsonl_line


def produce_ny_dfs_summary(row) -> list[dict]:
    """Produce a summary prompt/response pair from NY DFS data."""
    diagnosis = row["Diagnosis"]
    treatment = row["Treatment"]
    # coverage_type = row["Coverage Type"]
    denial_reason = row["Denial Reason"]

    gender = row["Gender"]
    age_range = row["Age Range"]

    # Construct one instance for each summary
    examples = []
    for col in list(row.index):
        if "Summary" in col:
            if not pd.isna(row[col]):
                # Construct example of how the review came to be
                response = row[col]
                summary_example = {
                    "prompt": f"""Provide a summary of an insurance case adjudication outcome for an external appeal of an insurance plan's determination to deny services for {treatment} to treat {diagnosis} for a {gender} patient in the age range {age_range}. The denial was made on the basis of {denial_reason}""",
                    "response": f"""{response}""",
                }
                examples.append(summary_example)

                # Construct example summarizing the review determination
                recommended_action = None
                if "overturn" in response:
                    recommended_action = "overturn"
                if "uphold" in response or "upheld" in response:
                    recommended_action = "uphold"
                if recommended_action:
                    outcome_example = {
                        "prompt": f"This is a health insurance case adjudication review from an independent reviewer:\n {response}\n What was the verdict of this reviewer?",
                        "response": f"The reviewer determined that the insurance plan should {recommended_action} its original denial.",
                    }
                    examples.append(outcome_example)

    return examples


def produce_cms_qic_partc_summary(row) -> list[dict]:
    """Produce a summary prompt/response pair from CMS QIC appeals data."""
    decision_rationale = row["decision_rationale"]
    # appeal_type = row["appeal_type"]
    condition = row["_condition"]
    item_service = row.get("item_service", None)
    part = row["part"]

    if not item_service:
        return None

    # Construct one instance for each summary
    if not pd.isna(condition) and not pd.isna(item_service):
        summary_example = {
            "prompt": f"""A Medicare part {part} beneficiary has been denied coverage by their plan for {item_service} services as part of their treatment for {condition}. A first level appeal was submitted, and the plan upheld the original denial. Provide a summary detailing your decision as to whether or not the Plan must cover these services under Medicare part {part} coverage rules.""",
            "response": f"""{decision_rationale}""",
        }
        return summary_example

    else:
        return None


def produce_cms_qic_partd_summary(row) -> list[dict]:
    """Produce a summary prompt/response pair from CMS QIC appeals data."""
    decision_rationale = row["decision_rationale"]
    # appeal_type = row["appeal_type"]
    condition = row["_condition"]
    drug = row.get("drug", None)
    part = row["part"]

    if not drug:
        return None

    # Construct one instance for each summary
    if not pd.isna(condition) and not pd.isna(drug):
        summary_example = {
            "prompt": f"""A Medicare part {part} beneficiary has been denied coverage by their plan for the drug {drug} as part of their treatment for {condition}. A first level appeal was submitted, and the plan upheld the original denial. Provide a summary detailing your decision as to whether or not the Plan must cover these services under Medicare part {part} coverage rules.""",
            "response": f"""{decision_rationale}""",
        }
        return summary_example

    else:
        return None


def produce_ca_dmhc_summary(row) -> list[dict]:
    """Produce a summary prompt/response pair from CA DMHC data."""
    diagnosis = row["DiagnosisCategory"]
    subdiagnosis = row["DiagnosisSubCategory"]
    treatment = row["TreatmentCategory"]
    subtreatment = row["TreatmentSubCategory"]
    denial_reason = row["Type"]

    gender = row["PatientGender"]
    age_range = row["AgeRange"]
    findings = row["Findings"]
    determination = row["Determination"]

    examples = []

    # Construct example of how the review came to be
    summary_example = {
        "prompt": f"""Provide a summary of an insurance case adjudication outcome for an external appeal of an insurance plan's determination to deny services for {treatment} ({subtreatment}) to treat {diagnosis} ({subdiagnosis}) for a {gender} patient in the age range {age_range}. The denial was made on the basis of {denial_reason}""",
        "response": f"""{findings}""",
    }
    examples.append(summary_example)

    # Construct example summarizing the review determination
    recommended_action = None
    if determination == "Overturned Decision of Health Plan":
        recommended_action = "overturn"
    if determination == "Upheld Decision of Health Plan":
        recommended_action = "uphold"
    if recommended_action:
        outcome_example = {
            "prompt": f"This is a health insurance case adjudication review from an independent reviewer:\n {findings}\n What was the verdict of this reviewer?",
            "response": f"The reviewer determined that the insurance plan should {recommended_action} its original denial.",
        }
        examples.append(outcome_example)

    return examples


if __name__ == "__main__":
    ########
    # NY DFS
    ########
    path = "./data/raw/ny_dfs/nydfs.xlsx"
    ny_df = pd.read_excel(path)
    examples = ny_df.apply(lambda row: produce_ny_dfs_summary(row), axis=1)
    flattened_examples = [ex for sublist in examples for ex in sublist]

    # Write lines
    outfile = "./data/alignment/ny_dfs.jsonl"
    for ex in flattened_examples:
        add_jsonl_line(outfile, ex)

    ##############
    # Medicare QIC
    ##############
    part = "part_c"
    path = f"./data/raw/medicare_qic/{part}.csv"
    cms_df = pd.read_csv(path)

    # Only consider rows with non-empty decision_rationales
    cms_df = cms_df[~cms_df["decision_rationale"].isna()]
    examples = cms_df.apply(lambda row: produce_cms_qic_partc_summary(row), axis=1)

    # Write lines
    outfile = f"./data/alignment/medicare_{part}_appeals.jsonl"
    for ex in examples:
        if ex:
            add_jsonl_line(outfile, ex)

    part = "part_d"
    path = f"./data/raw/medicare_qic/{part}.csv"
    cms_df = pd.read_csv(path)

    # Only consider rows with non-empty decision_rationales
    cms_df = cms_df[~cms_df["decision_rationale"].isna()]
    examples = cms_df.apply(lambda row: produce_cms_qic_partd_summary(row), axis=1)

    # Write lines
    outfile = f"./data/alignment/medicare_{part}_appeals.jsonl"
    for ex in examples:
        if ex:
            add_jsonl_line(outfile, ex)

    ##############
    # CA DMHC
    ##############
    path = "./data/raw/ca_dmhc/independent-medical-review-imr-determinations-trend.csv"
    df = pd.read_csv(path, encoding="ISO-8859-1")
    examples = df.apply(lambda row: produce_ca_dmhc_summary(row), axis=1)
    flattened_examples = [ex for sublist in examples for ex in sublist]
    outfile = "./data/alignment/ca_dmhc.jsonl"
    for ex in flattened_examples:
        add_jsonl_line(outfile, ex)
