"""Строит словарь связанных слов запроса и заголовка по обучающим парам."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

root = Path(__file__).resolve().parent / 'retrieval_solution'
sys.path.insert(0, str(root))
from common import WORK, normalize, log

log('Загрузка обучающих пар')
aux = pd.read_parquet(WORK / 'aux.parquet', columns=['text_group', 'item_title_raw'])
queries = aux.text_group.fillna('').map(normalize)
titles = aux.item_title_raw.fillna('').str.slice(0, 1000).map(normalize)
log('Векторизация слов запроса')
qvec = CountVectorizer(min_df=3, max_features=30000, binary=True, token_pattern=r'(?u)\b\w+\b', dtype=np.float32)
qmat = qvec.fit_transform(queries)
log('Векторизация слов заголовка')
tvec = CountVectorizer(min_df=5, max_features=60000, binary=True, token_pattern=r'(?u)\b\w+\b', dtype=np.float32)
tmat = tvec.fit_transform(titles)
log('Подсчёт связей')
cooccurrence = (qmat.T @ tmat).tocsr()
title_frequency = np.asarray(tmat.sum(axis=0)).ravel()
title_terms = tvec.get_feature_names_out()
mapping = {}
for query_term, row in qvec.vocabulary_.items():
    start, end = cooccurrence.indptr[row:row + 2]
    ids = cooccurrence.indices[start:end]
    counts = cooccurrence.data[start:end]
    mask = counts >= 3
    ids, counts = ids[mask], counts[mask]
    if not len(ids):
        continue
    weights = counts / np.sqrt(title_frequency[ids] + 20)
    selected = np.argsort(-weights, kind='stable')[:12]
    mapping[query_term] = [str(title_terms[i]) for i in ids[selected] if title_terms[i] != query_term]
joblib.dump(mapping, WORK / 'translation.joblib')
print('Терминов:', len(mapping), flush=True)
for word in ['ремонт', 'диван', 'маникюр', 'уборка', 'стрижка', 'переезд']:
    print(word, mapping.get(word), flush=True)
