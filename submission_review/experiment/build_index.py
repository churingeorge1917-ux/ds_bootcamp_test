"""Построение разреженных индексов с сохранением готовых этапов."""
from common import WORK, clean, normalize, log
import numpy as np
import gc
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def bm25(texts, max_features, ngram_range, b):
    """BM25: насыщение частоты термина и нормализация длины документа."""
    vectorizer = CountVectorizer(dtype=np.float32, min_df=2, max_features=max_features,
                          ngram_range=ngram_range, token_pattern=r'(?u)\b\w+\b')
    matrix = vectorizer.fit_transform(texts).tocsr()
    lengths = np.asarray(matrix.sum(axis=1)).ravel()
    document_frequency = np.bincount(matrix.indices, minlength=matrix.shape[1])
    idf = np.log1p((matrix.shape[0]-document_frequency+0.5)/(document_frequency+0.5)).astype('float32')
    length_norm = 1.2 * (1-b+b*lengths/max(lengths.mean(), 1))
    # Обрабатываем блоками, чтобы ограничить расход памяти.
    for start in range(0, matrix.shape[0], 10000):
        end = min(start+10000, matrix.shape[0])
        left, right = matrix.indptr[start], matrix.indptr[end]
        denominator = np.repeat(length_norm[start:end], np.diff(matrix.indptr[start:end+1]))
        matrix.data[left:right] = (matrix.data[left:right]*2.2/(matrix.data[left:right]+denominator))*idf[matrix.indices[left:right]]
    return {'vectorizer': vectorizer, 'inverted': matrix.T.tocsr(), 'idf': idf, 'bm25': True}

def tfidf(texts, **kwargs):
    vectorizer = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **kwargs)
    matrix = vectorizer.fit_transform(texts)
    return {'vectorizer': vectorizer, 'inverted': matrix.T.tocsr(), 'bm25': False}

def main():
    textpath = WORK/'texts.parquet'
    if not textpath.exists():
        frame = pd.read_parquet(WORK/'corpus.parquet', columns=[
            'item_title_raw', 'item_description_raw', 'item_infm_params_text'])
        texts_frame = pd.DataFrame()
        texts_frame['title_raw'] = frame.item_title_raw.fillna('').map(clean)
        for target, source, cap in [('title', 'item_title_raw', 1000),
                                    ('params', 'item_infm_params_text', 3000),
                                    ('desc', 'item_description_raw', 6000)]:
            log(f'Нормализация поля: {target}')
            texts_frame[target] = frame[source].fillna('').str.slice(0, cap).map(normalize)
        texts_frame.to_parquet(textpath, index=False)
        del frame, texts_frame
        gc.collect()
    configs = [
        ('title_word', 'title', lambda texts: tfidf(texts, ngram_range=(1, 2), min_df=1, max_features=180000)),
        ('title_bm25', 'title', lambda texts: bm25(texts, 180000, (1, 2), .35)),
        ('title_char', 'title_raw', lambda texts: tfidf(texts, analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=120000)),
        ('desc_bm25', 'desc', lambda texts: bm25(texts, 180000, (1, 1), .65)),
        ('params', 'params', lambda texts: tfidf(texts, ngram_range=(1, 2), min_df=2, max_features=40000)),
    ]
    for name, column, build in configs:
        path = WORK/f'{name}.joblib'
        if path.exists():
            continue
        log(f'Построение индекса: {name}')
        texts = pd.read_parquet(textpath, columns=[column])[column].tolist()
        model = build(texts)
        log(f'{name}: {model["inverted"].shape}, nnz={model["inverted"].nnz}')
        joblib.dump(model, path, compress=0)
        del texts, model
        gc.collect()
    log('Индексы объявлений готовы')

if __name__ == '__main__':
    main()
