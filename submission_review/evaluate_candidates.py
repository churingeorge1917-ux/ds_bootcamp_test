"""Временная оценка расширенного поиска на отложенных 800 запросах."""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent / 'retrieval_solution'
sys.path.insert(0, str(root))
from common import WORK, topk
from retrieval import Retriever

retriever = Retriever()
retriever.use_translation = len(sys.argv) > 2 and sys.argv[2] == 'translation'
retriever.global_limit = 160
retriever.local_limit = 180
retriever.pool_limit = 1000

split = sys.argv[1] if len(sys.argv) > 1 else 'test'
start, end = (3000, 3700) if split == 'dev' else (3700, 4500)
queries = pd.read_parquet(WORK / 'queries.parquet').iloc[start:end]
labels = pd.read_parquet(WORK / 'labels.parquet').groupby('example_id').item_id.agg(list).to_dict()
model_path = root.parents[1] / 'retrieval_solution/work/validation_model.txt' if split == 'dev' else root / 'ranker.txt'
model = lgb.Booster(model_str=model_path.read_text(encoding='utf-8'))
recalls = []
candidate_recalls = []
cross_city_recalls = []
for number, query in enumerate(queries.to_dict('records'), start=1):
    positives = {retriever.id_to_row[item] for item in labels[query['example_id']]}
    ids, features, _ = retriever.features(query)
    scores = model.predict(features, num_threads=4)
    found = set(ids[topk(scores, 50)])
    recall = len(found & positives) / len(positives)
    recalls.append(recall)
    candidate_recalls.append(len(set(ids) & positives) / len(positives))
    if any(retriever.document_fields['item_location_id'][p] != query['search_location_id'] for p in positives):
        cross_city_recalls.append(recall)
    if number % 100 == 0:
        print(number, float(np.mean(recalls)), float(np.mean(candidate_recalls)), flush=True)
print('Итог:', float(np.mean(recalls)), float(np.mean(candidate_recalls)), float(np.mean(cross_city_recalls)))
