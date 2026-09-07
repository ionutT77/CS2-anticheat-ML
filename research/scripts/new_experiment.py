"""Create an isolated rerun directory without overwriting published results."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def create(destination, benchmark, raw_data=None):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    if destination.is_relative_to(ROOT):
        raise ValueError("Place the rerun outside the frozen research directory.")
    if raw_data is not None:
        raw_data = Path(raw_data).resolve(strict=True)
        expected = ["no_cheater_present", "with_cheater_present"] if benchmark == "cs2cd" else ["legit", "cheaters"]
        if not all((raw_data / name).is_dir() for name in expected):
            raise ValueError(f"Raw data must contain these directories: {expected}")

    def copy(relative):
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    destination.mkdir(parents=True)
    for directory in ["anticheat", "scripts"]:
        for path in (ROOT / directory).glob("*.py"):
            copy(path.relative_to(ROOT))
    for name in ["requirements.txt", "requirements-training.txt", "METHODS.md", "DATA_SOURCES.md"]:
        copy(name)

    if benchmark == "cs2cd":
        for path in (ROOT / "data/cs2cd/metadata").glob("*.json"):
            copy(path.relative_to(ROOT))
        for name in ["protocol.json", "match_splits.csv"]:
            copy("experiments/cs2cd_v1/" + name)
        link = destination / "data/cs2cd/raw"
    else:
        for directory in ["models", "results", "reports"]:
            for path in (ROOT / directory).iterdir():
                if path.is_file():
                    copy(path.relative_to(ROOT))
        for name in ["splits.csv", "feature_names.json", "overlap_duplicate_edges.npy"]:
            copy("data/processed/" + name)
        copy("data/raw/download_manifest.json")
        link = destination / "data/raw/extracted"
    if raw_data is not None:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(raw_data, target_is_directory=True)
    print(f"Created {benchmark} rerun at {destination}")
    print("This reuses the published split. A new independent test cohort is needed for new performance claims.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--benchmark", choices=["cs2cd", "reference-lstm"], required=True)
    parser.add_argument("--raw-data", type=Path, help="Reuse a local raw-data directory through a symlink.")
    args = parser.parse_args()
    create(args.destination, args.benchmark, args.raw_data)


if __name__ == "__main__":
    main()
