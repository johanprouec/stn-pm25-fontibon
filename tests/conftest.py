import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.data.fontibon import build_all_partitions, load_fontibon_features
from src.utils.paths import FONTIBON_CSV


FONTIBON_SKIP_REASON = "Skipped: Fontibón institutional dataset is not distributed with the reproducible package."


@pytest.fixture(scope="session")
def fontibon_features():
    # Se omite SOLO si el archivo no existe; cualquier otro error de lectura sigue fallando.
    if not FONTIBON_CSV.exists():
        pytest.skip(FONTIBON_SKIP_REASON)
    return load_fontibon_features(FONTIBON_CSV)


@pytest.fixture(scope="session")
def fontibon_partitions(fontibon_features):
    return build_all_partitions(fontibon_features, ventana=24, limite=6)
