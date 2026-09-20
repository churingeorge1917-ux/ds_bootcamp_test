"""Обучение ранжирующей модели с ранее выбранными параметрами."""
import json
import lightgbm as lgb
import numpy as np
from common import ROOT, WORK, log


def load_training_data():
    #Первые 3700 запросов совпадают с итоговой обучающей выборкой исходника
    blocks = []
    for start in range(0, 3700, 100):
        with np.load(WORK / "features" / f"{start:05d}.npz") as block:
            blocks.append({key: block[key] for key in ("X", "y", "groups")})
    return {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}


def main():
    names = json.loads((WORK / "feature_names.json").read_text(encoding="utf-8"))
    training = load_training_data()
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="None", n_estimators=118,
        learning_rate=0.045, num_leaves=23, max_depth=-1,
        min_child_samples=100, colsample_bytree=0.9, reg_lambda=5,
        lambdarank_truncation_level=60, n_jobs=4, verbosity=-1,
        random_state=731, deterministic=True, force_col_wise=True,
    )
    model.fit(training["X"], training["y"], group=training["groups"], feature_name=names)
    (ROOT / "ranker.txt").write_text(model.booster_.model_to_string(), encoding="utf-8")
    config = {"model_weight": 1, "baseline_blend": 0, "features": names}
    (ROOT / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
