"""Экспериментальное обучение на большем числе запросов."""
import json
import lightgbm as lgb
import numpy as np
from common import ROOT, WORK, log, topk


def load_blocks(starts):
    blocks = []
    for start in starts:
        with np.load(WORK / 'features' / f'{start:05d}.npz') as source:
            blocks.append({key: source[key] for key in ('X', 'y', 'groups', 'denominators')})
    return {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}


def evaluate(data, scores):
    recalls = []
    coverage = []
    offset = 0
    for size, denominator in zip(data['groups'], data['denominators']):
        labels = data['y'][offset:offset + size]
        recalls.append(labels[topk(scores[offset:offset + size], 50)].sum() / denominator)
        coverage.append(labels.sum() / denominator)
        offset += size
    return float(np.mean(recalls)), float(np.mean(coverage))


def new_model():
    return lgb.LGBMRanker(
        objective='lambdarank', metric='None', n_estimators=118,
        learning_rate=.045, num_leaves=23, max_depth=-1,
        min_child_samples=100, colsample_bytree=.9, reg_lambda=5,
        lambdarank_truncation_level=60, n_jobs=4, verbosity=-1,
        random_state=731, deterministic=True, force_col_wise=True,
    )


def main():
    names = json.loads((WORK / 'feature_names.json').read_text(encoding='utf-8'))
    train_starts = list(range(0, 3000, 100)) + list(range(4500, 7500, 100))
    train = load_blocks(train_starts)
    model = new_model()
    model.fit(train['X'], train['y'], group=train['groups'], feature_name=names)
    del train
    for split, starts in [('dev', range(3000, 3700, 100)), ('test', range(3700, 4500, 100))]:
        data = load_blocks(starts)
        recall, coverage = evaluate(data, model.predict(data['X'], num_threads=4))
        log(f'{split}: Recall@50={recall:.6f}, кандидаты={coverage:.6f}')
    final_starts = train_starts + list(range(3000, 3700, 100))
    final = load_blocks(final_starts)
    model = new_model()
    model.fit(final['X'], final['y'], group=final['groups'], feature_name=names)
    (ROOT / 'ranker.txt').write_text(model.booster_.model_to_string(), encoding='utf-8')
    (ROOT / 'model_config.json').write_text(json.dumps({'model_weight': 1, 'baseline_blend': 0, 'features': names}, indent=2), encoding='utf-8')
    log('Модель обучена и сохранена')


if __name__ == '__main__':
    main()
