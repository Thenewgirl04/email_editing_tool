import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TASKS = ["shorten", "lengthen", "tone"]
METRICS = ["faithfulness_rating", "completeness_rating", "relevance_rating"]

DATASET_BASELINE_DIR = PROJECT_ROOT / "datasets" / "baseline"
DATASET_EDGE_DIR = PROJECT_ROOT / "datasets" / "with_edge_cases"
RESULTS_BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"
RESULTS_EDGE_DIR = PROJECT_ROOT / "results" / "with_edge_cases"
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.json"
REFERENCE_ARTIFACTS_DIR = PROJECT_ROOT / "reference_artifacts"
REFERENCE_SHORTEN_BASELINE = REFERENCE_ARTIFACTS_DIR / "evaluation_shorten_results.csv"
REFERENCE_SHORTEN_WITH_EDGE_CASES = (
    REFERENCE_ARTIFACTS_DIR / "evaluation_shorten_with_edgecases_results.csv"
)

EDGE_CASE_ID_THRESHOLD = 50
EVAL_MODELS = ["gpt-4o-mini", "gpt-4.1"]
JUDGE_MODEL = "gpt-4.1"


def dataset_dir(include_edge_cases: bool) -> Path:
    return DATASET_EDGE_DIR if include_edge_cases else DATASET_BASELINE_DIR


def results_dir(include_edge_cases: bool) -> Path:
    return RESULTS_EDGE_DIR if include_edge_cases else RESULTS_BASELINE_DIR


def dataset_paths(include_edge_cases: bool) -> list[Path]:
    dataset_root = dataset_dir(include_edge_cases)
    return [dataset_root / f"{task}.jsonl" for task in TASKS]


def load_primary_model() -> str:
    try:
        with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
            return config["primary_model"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return "gpt-4o-mini"
