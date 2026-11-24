"""Read the original HiDF split, resolving paths relative to the uploaded dataset."""
import csv
from pathlib import Path


def read_manifest(root):
    splits = {name: [] for name in ("train", "val", "test")}
    owners = {}
    seen = set()
    with (root / "hidf_clean_source_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            label = {"real": 0, "fake": 1}[row["label"]]
            folder = ("Real-img", "Fake-img")[label]
            name = Path(row["image_path"].replace("\\", "/")).name
            key = f"{folder}/{name}"
            if key in seen or int(row["class_id"]) != label:
                raise ValueError(f"Duplicate or mislabeled image: {key}")
            seen.add(key)
            if not (root / key).is_file():
                raise FileNotFoundError(root / key)
            split = row["split"]
            for field in ("source_group", "sha256"):
                value = row[field].strip()
                if not value or owners.setdefault((field, value), split) != split:
                    raise ValueError(f"Missing or cross-split {field} for {key}")
            splits[split].append((key, label, row["source_group"]))
    for name, rows in splits.items():
        if {row[1] for row in rows} != {0, 1}:
            raise ValueError(f"Both classes are required in {name}")
    return splits
