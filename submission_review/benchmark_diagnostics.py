"""Локальная диагностика различий обучающих и бенчмарк-запросов."""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent / 'retrieval_solution'
train = pd.read_parquet(root / 'data/train.parquet', columns=[
    'search_query', 'search_location_id', 'search_infm_params_text',
    'search_category', 'item_id', 'item_location_id'
])
benchmark = pd.read_parquet(root / 'data/benchmark_queries.parquet')
items = set(pd.read_parquet(root / 'data/benchmark_items.parquet', columns=['item_id']).item_id)
queries = pd.read_parquet(root / 'work/queries.parquet')

def basic(frame):
    words = frame.search_query.fillna('').str.split().str.len()
    return {
        'rows': len(frame),
        'unique_queries': frame.search_query.nunique(),
        'mean_words': round(float(words.mean()), 2),
        'one_word': round(float((words == 1).mean()), 3),
        'two_words': round(float((words == 2).mean()), 3),
        'has_filters': round(float(frame.search_infm_params_text.fillna('').ne('').mean()), 3),
        'top_locations': frame.search_location_id.value_counts().head(10).to_dict(),
    }

print('benchmark', basic(benchmark))
print('selected', basic(queries))
print('train_pairs', len(train), 'unique_queries', train.search_query.nunique())
train_texts = set(train.search_query)
print('benchmark exact text overlap', benchmark.search_query.isin(train_texts).sum())
contexts = train[['search_query', 'search_location_id', 'search_infm_params_text', 'search_category']].drop_duplicates()
merged = benchmark.merge(contexts, on=['search_query', 'search_location_id', 'search_infm_params_text', 'search_category'], how='inner')
print('benchmark exact context overlap', merged.query_id.nunique())
print('train positive in benchmark corpus', train.item_id.isin(items).mean())
print('train same location', (train.search_location_id == train.item_location_id).mean())
freq = train.search_query.value_counts()
print('benchmark train text frequency quantiles', freq.reindex(benchmark.search_query).fillna(0).quantile([0,.25,.5,.75,.9,.99,1]).to_dict())
print('selected train text frequency quantiles', freq.reindex(queries.search_query).fillna(0).quantile([0,.25,.5,.75,.9,.99,1]).to_dict())
