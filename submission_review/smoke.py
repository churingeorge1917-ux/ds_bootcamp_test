"""Проверка признаков и фильтра без большого набора данных."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent / 'retrieval_solution'
sys.path[:0] = [str(root), str(root.parents[1] / 'retrieval_solution/.venv/Lib/site-packages')]

import numpy as np
import pandas as pd
import retrieval

retriever = retrieval.Retriever.__new__(retrieval.Retriever)
retriever.n = 4
retriever.docs = pd.DataFrame({'item_id': ['a', 'b', 'c', 'd']})
retriever.texts = pd.DataFrame({key: ['тест'] * 4 for key in ('title', 'desc', 'params')})
retriever.document_fields = {
    'item_location_id': np.array([1, 1, 2, 2]),
    'item_latitude': np.array([0., 0., 0., 0.]),
    'item_longitude': np.array([0., 0., 0., 0.]),
    'item_category_id': np.array([1, 1, 1, 1]),
    'item_price': np.array([1., 2., 3., 4.]),
    'item_rating_reviews_count': np.zeros(4),
    'item_rating': np.ones(4),
    'item_is_phone_hidden': np.zeros(4),
    'item_is_message_forbidden': np.zeros(4),
}
retriever.centers = {}
retriever.indices = {name: None for name in retrieval.ENGINES}
retriever.history = lambda raw, normalized: (np.zeros(4), np.zeros(4), np.zeros(4), 0.)
retriever.doc_micro = np.zeros(4, dtype=int)
retriever.micro_prior = np.ones(1)
retriever.popularity = np.zeros(4)
retriever.location_counts = {}
retriever.location_totals = {}
retrieval.score_index = lambda index, text: np.array([0.1, 0.2, 0.3, 0.4])
query = {'search_query': 'тест', 'search_infm_params_text': '', 'search_location_id': 1,
         'search_category': 1, 'search_is_delivery_search': False}
ids, features, names = retriever.features(query, allow=np.array([False, True, False, True]))
expected = json.loads((root / 'model_config.json').read_text(encoding='utf-8'))['features']
assert ids.tolist() == [1, 3], ids
assert names == expected, (names, expected)
assert features.shape == (2, len(expected)), features.shape
print('Признаки и фильтрация кандидатов: OK')
