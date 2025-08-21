# scripts/_common.py
from pathlib import Path
import yaml, logging, sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

def load_paths(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    root = Path(__file__).resolve().parents[1]  # repo root
    raw_dir = (root / cfg["raw_dir"]).resolve()
    processed_dir = (root / cfg["processed_dir"]).resolve()
    results_dir = (root / cfg.get("results_dir", "results")).resolve()
    reports_dir = (root / cfg.get("reports_dir", "reports")).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir, results_dir, reports_dir, root
