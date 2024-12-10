import asyncio

from src.downloaders.ca_cdi import download as download_ca_doi_cases
from src.downloaders.ca_dmhc import download as download_ca_dmhc
from src.downloaders.cms import download_cms_pdfs
from src.downloaders.dol import download as download_dol
from src.downloaders.fda_guidance_documents import download as download_fda
from src.downloaders.federal_regulations import download as download_fr
from src.downloaders.hhs_oig import download as download_hhs_oig
from src.downloaders.medicaid_guidance import download as download_med_gd
from src.downloaders.medicare_qic import download as download_medicare_qic
from src.downloaders.meditron_guidelines import download as download_meditron_guidelines
from src.downloaders.ny_dfs import download as download_ny_dfs
from src.downloaders.state_code import download as download_sc
from src.downloaders.us_code import download as download_usc

DOWNLOAD_CMS_REGULATIONS = False
DOWNLOAD_MEDICAID_GUIDANCE = False
DOWNLOAD_FR = False
DOWNLOAD_CPBS = False
DOWNLOAD_DOL = False
DOWNLOAD_STATE_CODE = False
DOWNLOAD_US_CODE = False
DOWNLOAD_HHS_OIG_ARCHIVES = False
DOWNLOAD_CA_CDI = False
DOWNLOAD_CA_DMHC = False
DOWNLOAD_NY_DFS = False
DOWNLOAD_MEDICARE_QIC_APPEALS = False
DOWNLOAD_MEDITRON_GUIDELINES = False
DOWNLOAD_FDA = False

SOURCE_METADATA_PATH = "./sources.jsonl"
PROCESSED_METADATA_PATH = "processed_sources.jsonl"

if __name__ == "__main__":
    if DOWNLOAD_CMS_REGULATIONS:
        # TODO: cleanup
        url = "https://web.archive.org/web/20231229155526/https://www.cms.gov/medicare/regulations-guidance/manuals/internet-only-manuals-ioms"
        table_header = "Publication #"
        secondary_page_heading = "Downloads"
        download_folder = "./data/raw/regulatory_guidance"
        download_cms_pdfs(
            url,
            table_header,
            secondary_page_heading,
            SOURCE_METADATA_PATH,
            download_folder,
            req_delay=0.5,
        )

    if DOWNLOAD_FR:
        out_dir = "./data/raw/legal/federal_regulations"
        download_fr(out_dir, SOURCE_METADATA_PATH)

    if DOWNLOAD_CPBS:
        pass

    if DOWNLOAD_MEDICAID_GUIDANCE:
        output_dir = "./data/raw/regulatory_guidance/medicaid"
        download_med_gd(output_dir, SOURCE_METADATA_PATH)

    if DOWNLOAD_DOL:
        output_dir = "./data/raw/regulatory_guidance/faqs"
        download_dol(output_dir, SOURCE_METADATA_PATH)

    if DOWNLOAD_STATE_CODE:
        output_dir = "./data/raw/legal/state_code"
        asyncio.run(download_sc(output_dir, SOURCE_METADATA_PATH))

    if DOWNLOAD_US_CODE:
        output_dir = "./data/raw/legal/us_code"
        download_usc(output_dir, SOURCE_METADATA_PATH)

    if DOWNLOAD_HHS_OIG_ARCHIVES:
        output_dir = "./data/raw/hhs_oig"
        download_hhs_oig(output_dir, SOURCE_METADATA_PATH)

    if DOWNLOAD_CA_CDI:
        output_dir = "./data/raw/ca_cdi/summaries"
        download_ca_doi_cases(output_dir, SOURCE_METADATA_PATH)

    if DOWNLOAD_CA_DMHC:
        output_dir = "./data/raw/ca_dmhc"
        download_ca_dmhc(output_dir, SOURCE_METADATA_PATH)

    if DOWNLOAD_NY_DFS:
        output_dir = "./data/raw/ny_dfs"
        download_ny_dfs(output_dir, SOURCE_METADATA_PATH)

    if DOWNLOAD_MEDICARE_QIC_APPEALS:
        output_dir = "./data/raw/medicare_qic"
        download_medicare_qic(output_dir, SOURCE_METADATA_PATH)

    if DOWNLOAD_MEDITRON_GUIDELINES:
        output_dir = "./data/raw/meditron_guidelines"
        download_meditron_guidelines(output_dir, SOURCE_METADATA_PATH)

    if DOWNLOAD_FDA:
        output_dir = "./data/raw/fda_guidance"
        download_fda(output_dir, SOURCE_METADATA_PATH)

    # TODO:
    # - Add downloader for medicare CDs...downloading was done manually thus far
    # - More state medicaid guides.
    # - Old guideline.gov archives, if we can find a cache somewhere.
    # - Proprietary coverage determinations, crowdsourced contracts, as licensing/terms allow.
    # - Federal register (+ dedupe final rules from CFR)
