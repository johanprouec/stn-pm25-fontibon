"""Project root detection so code runs unmodified from VS Code or Colab.

No absolute, machine-specific paths anywhere in the codebase. Every path is
built from ``PROJECT_ROOT``, which is located by walking up from this file
until a marker (``requirements.txt``) is found. If the marker is not found
(e.g. this file was copied elsewhere without the rest of the repo), we fall
back to the current working directory, which is the right behavior for a
Colab cell that has already ``%cd``-ed into the cloned/uploaded project.
"""
from pathlib import Path

_MARKER = "requirements.txt"


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / _MARKER).exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT  # original CSVs are kept in the project root
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

BEIJING_CSV = RAW_DATA_DIR / "PRSA_data_2010.1.1-2014.12.31.csv"
FONTIBON_CSV = RAW_DATA_DIR / "Raw_Consolidated_ECA_Complete_2022.csv"
