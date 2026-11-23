"""Copy a prioritized Webots review batch into an untouched folder inbox."""

import csv
import hashlib
import shutil
from pathlib import Path

from train import DEFAULT_DATASET


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "review_batch_20260927"
NEW_EPISODES = {"20260927T054940743428Z", "20260927T055138768948Z"}
FOLDERS = {
    "straight": "8_직진",
    "left": "7_좌회전",
    "sharp_left": "4_급좌회전",
    "right": "9_우회전",
    "sharp_right": "6_급우회전",
    "stop": "5_정지선",
}
INBOX = "_미검수"
HOLD = "0_보류"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sample_evenly(items, count):
    if len(items) <= count:
        return items
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def main():
    if OUT.exists():
        raise FileExistsError(f"Review folder already exists; preserve user decisions: {OUT}")
    sources = {row["frame_id"]: row for row in read_csv(DEFAULT_DATASET / "training_manifest.csv")}
    split_rows = read_csv(ROOT / "split_manifest.csv")
    current = sorted([row for row in split_rows if row["episode_id"] in NEW_EPISODES],
                     key=lambda row: int(row["frame_id"]))
    if len(current) != 27 or any(row["split"] != "train" for row in current):
        raise RuntimeError("Expected 27 new training frames in the review batch")
    external = []
    for split, count in (("test_case1_correct_3m", 7),
                         ("test_iro_official_tiles", 4),
                         ("test_iro_context_mixed_holdout_18", 2)):
        group = sorted([row for row in split_rows if row["split"] == split and row["label_name"] == "right"],
                       key=lambda row: int(row["frame_id"]))
        if len(group) < count:
            raise RuntimeError(f"Too few right-turn frames in {split}")
        external.extend(sample_evenly(group, count))
    chosen = current + external
    if len(chosen) != 40 or len({row["frame_id"] for row in chosen}) != 40:
        raise RuntimeError("Review selection must contain 40 distinct frames")
    for folder in (*FOLDERS.values(), HOLD, INBOX):
        (OUT / folder).mkdir(parents=True)
    indexed = []
    for item in chosen:
        source = sources[item["frame_id"]]
        if (item["episode_id"], item["label_name"]) != (source["episode_id"], source["label_name"]):
            raise ValueError(f"Source and split disagree on frame {item['frame_id']}")
        path = DEFAULT_DATASET / source["center_image"]
        name = f"frame_{int(item['frame_id']):07d}.jpg"
        shutil.copy2(path, OUT / INBOX / name)
        indexed.append({"frame_id": item["frame_id"], "file": name,
                        "current_label": item["label_name"], "split": item["split"],
                        "episode_id": item["episode_id"], "world": item["world"],
                        "source_image": source["center_image"],
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    with (OUT / "index.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=indexed[0].keys())
        writer.writeheader()
        writer.writerows(indexed)
    (OUT / "검수방법.txt").write_text(
        "Webots 6클래스 사진 검수 — 첫 묶음 40장\n\n"
        "_미검수 폴더의 사진을 한 장씩 열어 보고 알맞은 폴더로 옮겨 주세요.\n"
        "8_직진, 7_좌회전, 4_급좌회전, 9_우회전, 6_급우회전, 5_정지선\n"
        "화면만으로 판단할 수 없거나 복귀 조향 강도가 애매하면 0_보류로 옮겨 주세요.\n"
        "사진을 복사하지 말고 옮기면 검수 완료 여부가 명확합니다.\n"
        "현재 라벨과 모델 예측을 보지 않고 먼저 판단할 수 있도록 파일명에는 프레임 번호만 넣었습니다.\n"
        "원본 영상과 학습·평가 분할은 이 폴더를 옮겨도 바뀌지 않습니다.\n"
        "검수를 마치면 말씀해 주세요. 폴더 위치를 읽어 변경·보류 목록을 만들고, 충돌을 확인한 뒤 반영하겠습니다.\n"
        "index.csv는 프레임 출처·기존 라벨 확인용입니다. 평가 영상은 학습으로 옮기지 않습니다.\n",
        encoding="utf-8-sig")
    print(f"Created {len(indexed)} review copies: {OUT}")


if __name__ == "__main__":
    main()
