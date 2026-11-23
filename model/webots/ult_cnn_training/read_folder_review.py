"""Read the user's folder moves as proposed labels without editing the dataset."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from prepare_folder_review import FOLDERS, HOLD, INBOX, OUT


def main():
    with (OUT / "index.csv").open(newline="", encoding="utf-8-sig") as stream:
        indexed = list(csv.DictReader(stream))
    expected = {row["file"]: row for row in indexed}
    if len(expected) != len(indexed):
        raise ValueError("Duplicate filename in review index")
    folder_to_label = {folder: label for label, folder in FOLDERS.items()}
    allowed = set(folder_to_label) | {HOLD, INBOX}
    seen = {}
    for folder in allowed:
        for path in (OUT / folder).iterdir():
            if not path.is_file() or path.suffix.lower() != ".jpg":
                continue
            if path.name not in expected:
                raise ValueError(f"Unknown review image: {path}")
            if path.name in seen:
                raise ValueError(f"Image occurs in multiple folders: {path.name}")
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected[path.name]["source_sha256"]:
                raise ValueError(f"Image was changed rather than moved: {path.name}")
            seen[path.name] = folder
    missing = set(expected) - set(seen)
    if missing:
        raise ValueError(f"Review images missing: {sorted(missing)}")
    decisions = []
    for row in indexed:
        folder = seen[row["file"]]
        if folder == INBOX:
            continue
        proposed = folder_to_label.get(folder, "")
        decision = "hold" if folder == HOLD else ("confirm" if proposed == row["current_label"] else "correct")
        decisions.append({"frame_id": row["frame_id"], "split": row["split"],
                          "episode_id": row["episode_id"], "previous_label": row["current_label"],
                          "proposed_label": proposed, "decision": decision, "folder": folder})
    output = OUT / "reviewed_decisions.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=("frame_id", "split", "episode_id", "previous_label",
                                                        "proposed_label", "decision", "folder"))
        writer.writeheader()
        writer.writerows(decisions)
    summary = {"total": len(indexed), "reviewed": len(decisions), "unreviewed": len(indexed) - len(decisions),
               "decisions": dict(Counter(row["decision"] for row in decisions)),
               "reviewed_by_split": dict(Counter(row["split"] for row in decisions)),
               "note": "Proposed labels only; source images and dataset manifests are unchanged"}
    (OUT / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
