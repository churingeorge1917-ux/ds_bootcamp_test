"""Временная локальная оценка сохранённых моделей на 800 отложенных запросах."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

root = Path(__file__).resolve().parent / "retrieval_solution"
original = root.parents[1] / "retrieval_solution"
sys.path.insert(0, str(root))
from common import topk


def recall(model):
    values = []
    candidate_values = []
    for start in range(3700, 4500, 100):
        with np.load(original / "work/features" / f"{start:05d}.npz") as data:
            scores = model.predict(data["X"], num_threads=4)
            offset = 0
            for size, denominator in zip(data["groups"], data["denominators"]):
                labels = data["y"][offset:offset + size]
                values.append(labels[topk(scores[offset:offset + size], 50)].sum() / denominator)
                candidate_values.append(labels.sum() / denominator)
                offset += size
    return float(np.mean(values)), float(np.mean(candidate_values))


for filename in ["validation_model.txt", "ranker.txt"]:
    path = original / "work" / filename if filename == "validation_model.txt" else root / filename
    model = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
    print(filename, recall(model))
