import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RAW_DIR = r"C:\Users\ellwalke\Data\TTU_200m_Nov2013-Oct2014"


def pytest_addoption(parser):
    parser.addoption(
        "--integration", action="store_true",
        help="run tests that exercise real data or multiple pipeline stages together",
    )


def pytest_runtest_setup(item):
    if "integration" in item.keywords and not item.config.getvalue("integration"):
        pytest.skip("need --integration option to run")


@pytest.fixture(autouse=True, scope="session")
def _isolated_ttu_tower_home(tmp_path_factory):
    """No test ever touches the real ttu-tower home."""
    home = tmp_path_factory.mktemp("ttu-tower-home")
    os.environ["TTU_TOWER_HOME"] = str(home)
    yield home
    os.environ.pop("TTU_TOWER_HOME", None)


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    """The real raw-data directory, for integration tests; overridable via TTU_RAW_DIR."""
    return Path(os.environ.get("TTU_RAW_DIR", _DEFAULT_RAW_DIR))


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return REPO_ROOT / "tests" / "fixtures"
