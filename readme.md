# HICRIC: A Dataset of Law, Policy, and Regulatory Guidance for Health Insurance Coverage Understanding

<p align="center"><img src="./assets/logo.png" width="300" height="300" /></p>

Health Insurance Coverage Rules Interpretation Corpus (HICRIC) is a collection of unannotated text curated to support
applications that require understanding of U.S. health insurance coverage rules.

It consists of:

- An small, unlabeled corpus of authoritative text related to law, contracts, and medicine. This corpus is intended for use in pretraining language models
and/or, independently, for use as a standalone knowledge base for retrieval applications.
- Annotated health insurance case adjudications. This data is intended for use in adjudication related modeling.

## Using the Data

### Access

#### Corpus
The corpus can be found on Huggingface:

[https://huggingface.co/datasets/Persius/hicric](https://huggingface.co/datasets/Persius/hicric)

To download the corpus for the purpose of using it with this code, use our script:

```zsh
python download_corpus_hf.py
```

#### Case Adjudications

The dataset can be found on Huggingface:

[https://huggingface.co/datasets/Persius/imr-appeals](https://huggingface.co/datasets/Persius/imr-appeals)

To download the corpus for the purpose of using with this code, use our script:

```zsh
python download_adjudications_hf.py
```

### Redistribution

Please consult the licenses for all source data for yourself if you plan to redistribute any of it. To the best of our knowledge, our redistributions abide by all such licenses.

### Risks

We believe there are numerous risks associated with our released data, which we've done our best to mitigate. Our main concerns involve:

- Potential for Propagation of Bias
- Potential for Misuse

Please see our forthcoming paper for a thorough discussion of these perceived risks.

### Limitations

There are many limitations associated with our released data, and our advice is to consider and weigh these limitations carefully to
inform resonsible and effective use. The main categories of limitation are:

- Task Shortcomings
- Simplicity of the Benchmark
- Corpus Deficiencies

Please see our forthcoming paper for a thorough discussion of these perceived risks.

## Dataset Breakdown

Each document in our dataset comes equipped with a set of plain-text _tags_. In constructing the
data we formulated a particular privileged set of partitioning tags: these are a set of tags with
the property that each document in the dataset is associated with exactly one tag in the set,
and none of the tags are unused.

The tags are the following:


- **legal**

- **regulatory-guidance**

- **contract-coverage-rule-medical-policy**

-  **opinion-policy-summary**

-  **case-description**

-  **clinical-guidelines**

In addition to this set of partitioning tags, we intoduce another privileged tag:

- **kb** This tag indicates that a document is suitable for use in a knowledge base.

    This is a subjective determination, but the intent is to label text that comes from a reputable,
    _definitive_ source. For example, a summary of Medicaid rules as stated by an employee of HHS
    during congressional testimony would not be labeled with the `kb` tag, because such testimony is
    not the definitive source for the ground truth of such rules. On the other hand, federal
    law describing those same rules would be labeled with the `kb` tag.


A high level characterization of the distribution of text in our corpus in terms of these 
privileged tags is shown in the table below.

| Category | Num Documents | Words | Chars | Size (GB) |                                                                                                                                     
| -------- | ------------- | ----- | ----- | --------- |                                                                                                                                     
| All Partition Parts | 8,310 | 417,617,646 | 2,699,256,987 | 2.81 |                                                                                                                                     
| kb | 1,434 | 170,717,368 | 1,120,961,295 | 1.13 |                                                                                                                                          
| legal | 335 | 92,357,802 | 596,044,008 | 0.60 |                                                                                                                                            
| regulatory-guidance | 1,110 | 5,536,585 | 38,607,587 | 0.04 |                                                                                                                              
| contract-coverage-rule-medical-policy | 7 | 196,156,813 | 1,228,184,524 | 1.31 |                                                                                                           
| opinion-policy-summary | 2,094 | 19,462,399 | 133,049,956 | 0.14 |                                                                                                                         
| case-description | 2,629 | 214,267,074 | 1,351,074,791 | 1.45 |
| clinical-guidelines | 2,150 | 81,955,020 | 553,041,990 | 0.56 |


## Using the Code

To use any of our downloaders, processors, or text generation scripts to reproduce the unlabeled dataset generation in full, proceed as follows:

### Setup Environment

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

### Reproduce Unlabeled Corpus

#### Download Raw Sources + Produce Source Metadata

```bash
python download_raw.py
```

#### Process Sources + Produce Processed Metadata

```bash
python process_local.py
```

### Train Outcome Predictor

To train outcome predictors, you need to either: 

- Follow the steps above to download the dataset from Huggingface, using the custom scripts.
- Reproduce the entire unlabeled corpus using our scrapers, as described above.


#### Train pseudo-labeling models

```bash
# Train background span selector
export WANDB_API_KEY=your=api-key # only necessary if using wandb, as specified in default config
python -m src.modeling.train_background_token_classification --config_path="src/modeling/config/background_token_classification/default.yaml"
```

```bash
# Train sufficiency classifier
export WANDB_API_KEY=your=api-key # only necessary if using wandb, as specified in default config
python -m src.modeling.train_sufficiency_classifier --config_path="src/modeling/config/sufficiency_classification/default.yaml"
```

Note: the models above will use the manual annotations in `./data/annotated/case-backgrounds.jsonl`

```bash
# Use models above to extract background spans and label with 3-class pseudo-label
python -m src.modeling.background_extraction
```

#### Train Outcome Prediction Models

##### By Only Finetuning a Pretrained Model on the Benchmark

```bash
# Use (background, outcome) pairs to train outcome predictor (from HF pretrained model)
export WANDB_API_KEY=your=api-key # only necessary if using wandb, as specified in default config
python -m src.modeling.train_outcome_predictor --config_path="src/modeling/config/outcome_prediction/distilbert.yaml"
```

#### By Pretraining on HICRIC, then Finetuning on the Benchmark

```bash
# Pretrain a BERT type model on HICRIC via MLM
export WANDB_API_KEY=your=api-key # only necessary if using wandb, as specified in default config
python -m src.modeling.pretrain --config_path="src/modeling/config/pretrain/distilbert.yaml"

# Then use (background, outcome pairs to train outcome predictor (from hicric pretrained variant)
export WANDB_API_KEY=your=api-key # only necessary if using wandb, as specified in default config
python -m src.modeling.train_outcome_predictor --config_path="src/modeling/config/outcome_prediction/distilbert_hicric_pretrained.yaml"
```

### Generate Alignment Data for Supervised Fine Tuning

```bash
python generate_sft_alignment_data.py
```

## Repository Organization

We separate the source metadata and download utilities from the processed metadata and processing utilities to
retain modularity in these unrelated concerns. This organizaton supports redownloading all of the raw data, or, independently, re-processing
only the subset of data scraped from pdfs with an alternate pdf processing pipeline, for example.


### Downloaders

Each data source in the dataset has an associated downloader, housed in [src/downloaders](./src/downloaders). Each downloader is
a function with the signature:


```python
def download(output_dir: str, source_meta_path: str) -> None:
    pass
```

The role of the function is to download or scrape raw data from a source, write it to disk in the specified ``output_dir``, and
write a piece of metadata that points to the downloaded artifact in a jsonl file (the file located at `source_meta_path`). The
nature of the metadata is described in the following section.

### Source Metadata

The file [sources.jsonl](./sources.jsonl) documents metadata pertaining to the raw _sources_ ultimately used in this dataset. This
means either file downloads, or scraped data (which requires a poetic license to deem "raw").

For example, the first line of sources.jsonl is:

```json
{
    "url": "https://downloads.cms.gov/medicare-coverage-database/downloads/exports/ncd.zip",
    "date_accessed": "2024-01-17",
    "local_path": "./data/raw/medicare/ncd/ncd_csv.zip",
    "tags": ["medicare", "kb", "contract-coverage-rule-medical-policy"],
    "preprocessor": "medicare_cds",
    "md5": "39bb06a088e67aad89ee2ddcb26e03ba"
}
```


This is metadata that refers to a particular subset of the Medicare Coverage Database that was downloaded from a link on the page: [https://www.cms.gov/medicare-coverage-database/downloads/downloads.aspx](https://www.cms.gov/medicare-coverage-database/downloads/downloads.aspx). Subsequently, that raw download was
parsed and processed to produce text, but such further steps are beyond the purview of [sources.jsonl](./sources.jsonl).

Each source metadata record includes a few pieces of information, including the direct download url or url from which the data was acquired (as applicable), the date a download or scrape occurred, a relative local path to the downloaded data, plain-text tags associated with the data, and a plain-text key for a _preprocessor_ with which the downloaded data can be converted to our standard processed format.

#### Metadata Description


| Name | Description | Definition | Required |Example Value |
| ----- | ---- |  ---------- | -------- | -------- |
| **url** | Source Url | A source url from which the data was obtained. | Yes | https://downloads.cms.gov/medicare-coverage-database/downloads/exports/ncd.zip |
| **date_accessed** | Date of Access |  The date at which the data was downloaded or scraped (YYYY-MM-DD). | Yes | 2024-01-17|
| **local_path** | Local Path to Data |  Relative path to local raw data download or scrape. | Yes | ./data/raw/medicare/ncd/ncd_csv.zip |
| **tags** | Source Tags |  An array of plain-text tags that pertain to the raw data.  | Yes (possibly empty) | ["medicare", "kb", "contract-coverage-rule-medical-policy"] |
| **preprocessor** | Processing Function Key | A key for a processor function that was used to transform the file at ``local_path`` to the expected standard format. | Yes. | medicare_cds |
| **md5** | MD5 Hash | MD5 hash of the file contents stored at ``local_path``. | Yes | 39bb06a088e67aad89ee2ddcb26e03ba |


**Note**: 
Text for which no further processing is desired (e.g. because it was parsed and processed into the standard format at scrape time) has a preprocessor value of `null`
in sources.jsonl.

### Processors

Each raw source item in [sources.jsonl](./sources.jsonl) is labeled with a ``preprocessor`` key for an associated processor. Processors are housed in [src/processors](./src/processors).

Each processor is a function with the signature:

```python
def process(source_lineitem: dict, output_dirname: str) -> dict:
    pass
```

The role of the function is to accept a lineitem from the source metadata, process the raw file to which that metadata points to produce
text data in our standard format, and then return an updated lineitem with metadata about the newly processed variant. The processor will write the processed copy of the data to disk in the directory specified by ``output_dirname``. The detailed nature of the updated metadata returned by these functions is described in the following section.



### Processed Metadata

The file [processed_sources.jsonl](./processed_sources.jsonl) documents metadata pertaining to the standardized constituents of our dataset, and how they were acquired from the raw records enumerated in [sources.jsonl](./sources.jsonl). For the most part, this updated metadata is the same as [sources.jsonl](./sources.jsonl). The main difference is that there are now file pointers pointing to the local, standardized, processed variants of the lineitems.


For example, the first line in processed_sources.jsonl corresponding to the source example above is:


```json
{
    "url": "https://downloads.cms.gov/medicare-coverage-database/downloads/exports/ncd.zip",
    "date_accessed": "2024-01-17",
    "local_path": "./data/raw/medicare/ncd/ncd_csv.zip",
    "tags": ["medicare", "kb", "contract-coverage-rule-medical-policy"],
    "preprocessor": "medicare_cds",
    "md5": "39bb06a088e67aad89ee2ddcb26e03ba",
    "local_processed_path": "./data/processed/medicare/ncd/ncd.jsonl",
    "stats": {"size": 600852, "words": 84013, "chars": 583462}
}
```

Note here that the metadata is exactly the same, with the exception of two new fields: `local_processed_path` and `stats`.

<!-- TODO: we should also hash the processed files, to determine if re-processing is necessary given e.g. new preprocessors, same source. -->

| Name | Description | Definition | Required |Example Value |
| ----- | ---- |  ---------- | -------- | -------- |
|**local_processed_path** | Local Path to Processed Data |  Relative path to local processed data. | Yes. | .data/processed/medicare/ncd/ncd.jsonl |
| **stats** | Some basic stats about the _text_ field of the _processed_ file. | A dict with the total size (bytes), number of words, and number of chars _in the text components_ of the processed jsonl file. | No | {"size": 600852, "words": 84013, "chars": 583462} |


#### Standardized Processed Format

The actual standardized constituents of our dataset (rather than the metadata just described) are also jsonl files. Each jsonl file must satisfy one property: each json lineitem has a `text` key with the raw text data.

| Name | Description | Definition | Required |Example Value |
| ----- | ---- |  ---------- | -------- | -------- |
| **text** | The text of the processed file. | The text of the processed file. | Yes | Summary Reviewer \n\n\nA 51-year-old female enrollee has requested reimbursement for Avastin provided on 12/16/14, 1/6/15, 1/27/15, 2/17/15, and 3/10/15.|

In addition to this minimal standardization requirement, those processed files corresponding to
case summary data (as indicated by the `case-description` tag in their processed metadata) may optionally include the following additional fields in each of their json lineitems:


| Name | Description | Definition | Required |Example Value |
| ----- | ---- |  ---------- | -------- | -------- |
| **appeal_type** | The Type of Appeal  | String indicating the grounds on which the denial being appealed was made. Specificity level not yet standardized. | No | Medical Necessity |
| **coverage_type** | The Type of Coverage  | String indicating the coverage type. Specificity level not yet standardized. | No | Commercial |
| **diagnosis** | The diagnosis or medical event in question.  | String indicating the diagnosis or medical event. Specificity level not yet standardized. | No | Metastatic Cancer |
| **treatment** | The treatment or service in question.  | String indicating the treatment or service. Specificity level not yet standardized. | No | Chemotherapy/ Cancer Medications |
| **decision** | The appeal outcome.  | String indicating the appeal outcome. Not yet standardized. | No | Insurer Denial Overturned |
| **appeal_expedited** | Appeal expedited status.  | Boolean indicating whether the appeal was expedited. | No | False |
| **patient_race** | The reported race of the patient. | The reported race of the patient. | No | Asian |




## License

**Curated Data**


**Annotations, Documentation, and Original Data**

All _original_ data, documentation, and media presented in this repository is licensed under <a rel="CC-BY-SA-4.0-license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

See [`LICENSE.CC-BY-SA-4.0`](./LICENSE.CC-BY-SA-4.0) for a full text copy of this license.

**Code**

All original source code in this repository, including that used to scrape and parse data, and to train models, is licensed under <a rel="apache-2.0-license" href="https://www.apache.org/licenses/LICENSE-2.0">Apache 2.0 License</a>.

See [`LICENSE`](./LICENSE) for a full text copy of this license.

Please start a discussion thread for any question or concerns related to licensing.


<!-- ## TODOS

- Add thresholded model comparison.
- Improve sufficiency labels, improve ground truth.
- Add some basic filtering (remove PII, names, headers, footers, tables, etc.)
- Use official FAQs to make generative QA subset.
- Add RLHF or QA targeted annotation.
- Add CI to verify hash dedupes, field existence.
- Add CI for linting, formatting.
- Add contributing docs.
- Add model to predict (covered, not covered, unknown) based on structured (diagnosis, treatment, coverage market, plan identifier).
- Add model to dynamically add tags to scraped content.
- Add MIMIC III case note data.
-->

## Attribution

If you find this data useful in your work, please consider citing it.

In adhering to the attribution clause of the license governing our original data, documentation, and other media, you can attribute this work as
"HICRIC Data", and share the url: [https://github.com/TPAFS/hicric](https://github.com/TPAFS/hicric).

For the code, you can use the following citation:

```latex
  @software{hicric,
	author ={Mike Gartner},
	title={HICRIC: Law, Policy, and Medical Guidance for Health Insurance Coverage Understanding},
	year={2024},
    url={https://github.com/TPAFS/hicric}
}
```

## Contact

For questions or comments, please reach out to `info@persius.org`