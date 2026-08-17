"""
Runs the full pipeline (split -> train -> evaluate) for every category
folder found under data/raw/, one after another.

Usage: python -m src.run_all_categories
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def find_categories():
    return sorted([p.name for p in DATA_RAW_DIR.iterdir() if p.is_dir()])


def run_step(module, category):
    print(f"\n{'='*60}\n{category} -> {module}\n{'='*60}")
    result = subprocess.run(
        [sys.executable, "-m", module],
        env={**__import__("os").environ, "MVTEC_CATEGORY": category},
    )
    if result.returncode != 0:
        print(f"FAILED: {module} for category '{category}'")
        return False
    return True


def main():
    categories = find_categories()
    print(f"Found categories: {categories}")

    results = {}
    for category in categories:
        ok = run_step("src.data.prepare_splits", category)
        if ok:
            ok = run_step("src.training.train", category)
        if ok:
            ok = run_step("src.evaluation.evaluate", category)
        results[category] = "done" if ok else "failed"

    print("\nSummary:")
    for category, status in results.items():
        print(f"  {category}: {status}")


if __name__ == "__main__":
    main()