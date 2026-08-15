"""Project paths for PI-JWM.

PI-JWM is our own Physical-Information Joint World Model framework. Third-party
simulators and historical experiment artifacts are kept inside this project
under `reference/` and `artifacts/`, rather than being treated as the framework.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
REFERENCE_DIR = PROJECT_ROOT / "reference"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SIMULATOR_ROOT = REFERENCE_DIR / "AirFogSim"
SIMULATOR_EXAMPLES = SIMULATOR_ROOT / "examples"

EXPERIMENT_ASSET_ROOT = ARTIFACTS_DIR / "experiments" / "airfogsim_v0"
DATASET_DIR = EXPERIMENT_ASSET_ROOT / "datasets"
REPORT_DIR = EXPERIMENT_ASSET_ROOT / "reports"
FIGURE_DIR = EXPERIMENT_ASSET_ROOT / "figures"
