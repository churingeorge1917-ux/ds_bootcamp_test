"""Независимая проверка готового ответа относительно исходных данных."""
import csv
import hashlib
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent / "retrieval_solution"
expected_queries = set(pd.read_parquet(root / "data/benchmark_queries.parquet", columns=["query_id"]).query_id)
valid_items = set(pd.read_parquet(root / "data/benchmark_items.parquet", columns=["item_id"]).item_id)
with (root / "answer.csv").open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    assert reader.fieldnames == ["query_id", "answer"]
    rows = list(reader)
assert len(rows) == len(expected_queries) == 2452
assert {row["query_id"] for row in rows} == expected_queries
for row in rows:
    items = row["answer"].split()
    assert len(items) == len(set(items)) == 50
    assert set(items) <= valid_items
digest = hashlib.sha256((root / "answer.csv").read_bytes()).hexdigest()
print(f"CSV проверен: {len(rows)} запросов, 50 уникальных объявлений в каждом, SHA-256 {digest}")
