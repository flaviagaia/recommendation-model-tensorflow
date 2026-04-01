from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd


INTERACTIONS = [
    ("U-1001", "I-1001", 5.0),
    ("U-1001", "I-1002", 4.0),
    ("U-1001", "I-1003", 1.0),
    ("U-1002", "I-1001", 4.0),
    ("U-1002", "I-1004", 5.0),
    ("U-1002", "I-1005", 4.0),
    ("U-1003", "I-1002", 5.0),
    ("U-1003", "I-1003", 4.0),
    ("U-1003", "I-1006", 5.0),
    ("U-1004", "I-1004", 2.0),
    ("U-1004", "I-1005", 5.0),
    ("U-1004", "I-1006", 4.0),
    ("U-1005", "I-1001", 1.0),
    ("U-1005", "I-1003", 5.0),
    ("U-1005", "I-1006", 4.0),
    ("U-1006", "I-1002", 2.0),
    ("U-1006", "I-1004", 4.0),
    ("U-1006", "I-1005", 5.0),
]


def ensure_interactions_dataset(base_dir: str | Path) -> str:
    base_path = Path(base_dir)
    raw_dir = base_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = raw_dir / "user_item_interactions.csv"
    dataframe = pd.DataFrame(INTERACTIONS, columns=["user_id", "item_id", "rating"])

    with NamedTemporaryFile("w", suffix=".csv", delete=False, dir=raw_dir, encoding="utf-8") as tmp_file:
        temp_path = Path(tmp_file.name)
    try:
        dataframe.to_csv(temp_path, index=False)
        temp_path.replace(dataset_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return str(dataset_path)
