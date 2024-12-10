from .ca_cdi import process as ca_cdi_processor
from .ca_dmhc import process as ca_dmhc_processor
from .medicare_cds import process as medicare_cd_processor
from .medicare_qic import process as medicare_qic_processor
from .meditron_guidelines import process as meditron_guidelines_processor
from .ny_dfs import process as ny_dfs_processor
from .pdf import process as pdf_processor
from .txt import process as txt_processor
from .us_code import process as usc_processor

PROCESSORS = {
    "medicare_cds": medicare_cd_processor,
    "pdf": pdf_processor,
    "usc-xml": usc_processor,
    "text": txt_processor,
    "ny_dfs": ny_dfs_processor,
    "ca_cdi": ca_cdi_processor,
    "ca_dmhc": ca_dmhc_processor,
    "medicare_qic": medicare_qic_processor,
    "epfl-guidelines": meditron_guidelines_processor,
}
