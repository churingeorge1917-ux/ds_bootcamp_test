"""Общая обработка текста и воспроизводимый выбор лучших результатов."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import re
from functools import lru_cache
from pathlib import Path
import numpy as np
import snowballstemmer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
WORK = ROOT / 'work'
WORK.mkdir(exist_ok=True)
QUERY_COLS = ['search_query', 'search_location_id', 'search_is_delivery_search',
              'search_infm_params_text', 'search_category']
STEMMER = snowballstemmer.stemmer('russian')
STOP = set('и в во на с со к ко от до по для из у а но или не это как при за под над о об'.split())

def clean(text):
    return ' '.join(re.findall(r'[a-zа-я0-9]+', str(text or '').lower().replace('ё', 'е')))

@lru_cache(maxsize=200000)
def stem(word):
    return STEMMER.stemWord(word)

def normalize(text):
    return ' '.join(stem(w) for w in clean(text).split() if w not in STOP)

def topk(scores, k):
    k = min(k, len(scores))
    if not k:
        return np.empty(0, dtype=np.int32)
    ids = np.argpartition(-scores, k-1)[:k]
    cutoff = scores[ids].min()
    above = np.flatnonzero(scores > cutoff)
    ties = np.flatnonzero(scores == cutoff)[:k-len(above)]
    ids = np.r_[above, ties]
    return ids[np.lexsort((ids, -scores[ids]))].astype(np.int32)

def log(message):
    from datetime import datetime
    print(datetime.now().strftime('%H:%M:%S'), message, flush=True)
