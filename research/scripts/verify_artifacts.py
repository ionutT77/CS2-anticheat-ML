"""Verify distributed code, saved weights, splits and calibration provenance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    checked = {}

    def check(path, expected):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch: {path.relative_to(ROOT)}")
        checked[str(path.relative_to(ROOT))] = actual

    for path, digest in read(ROOT / "models/selection.json")["sha256"].items():
        check(ROOT / path, digest)
    lstm = ROOT / "experiments/lstm_revision"
    for path, digest in read(lstm / "selection.json")["sha256"].items():
        check(lstm / path, digest)
    protocol = read(lstm / "protocol.json")
    for group in ["original_artifact_hashes", "supplement_source_hashes"]:
        for path, digest in protocol[group].items():
            check(ROOT / path, digest)

    cs2 = ROOT / "experiments/cs2cd_v1"
    selection = read(cs2 / "selection.json")
    for key, base in [("artifact_sha256", cs2), ("code_sha256", ROOT)]:
        for path, digest in selection[key].items():
            check(base / path, digest)
    audit = read(cs2 / "data_audit.json")
    check(cs2 / "events.parquet", audit["events_sha256"])
    check(cs2 / "players.csv", audit["players_sha256"])
    calibration = read(cs2 / "calibration.json")
    check(cs2 / "selection.json", calibration["selection_sha256"])
    check(ROOT / "scripts/evaluate_cs2cd.py", calibration["evaluation_code_sha256"])
    result = read(cs2 / "test_results.json")
    check(cs2 / "calibration.json", result["calibration_sha256"])
    check(cs2 / "selection.json", result["selection_sha256"])
    print(f"Verified {len(checked)} files and the selection/calibration chain.")


if __name__ == "__main__":
    main()
