from pathlib import Path










#============== PATH CONFIG ======================================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESC_ROOT = PROJECT_ROOT.parents[1]

CONFIGS_DIR = PROJECT_ROOT / "configs"
PAPERS_DIR = CONFIGS_DIR / "papers"

RUNS_DIR = PROJECT_ROOT / "runs"

PUBLICATIONS_DIR = DESC_ROOT / "publications"

DEFAULT_RANDOM_SEED = 1
#==============================================================================================================