"""Поиск кандидатов и признаки пар для обучения и предсказаний"""
from common import WORK, clean, normalize, topk, log
import re
import numpy as np
import gc
import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ENGINES = ['title_word', 'title_bm25', 'title_char', 'desc_bm25', 'params']

def score_index(index, text):
    query = index['vectorizer'].transform([text]).tocsr()
    if index['bm25']:
        query.data[:] = 1
        denom = max(float(index['idf'][query.indices].sum()), 1e-6)
    else:
        denom = 1
    return (query @ index['inverted']).toarray().ravel() / denom

class Retriever:
    # Оставляем больше кандидатов из общей выдачи, включая другие города.
    global_limit = 160
    local_limit = 180
    pool_limit = 1000

    def __init__(self):
        self.indices = {n: joblib.load(WORK/f'{n}.joblib', mmap_mode='r') for n in ENGINES}
        columns = ['item_id', 'item_location_id', 'item_microcat_id', 'item_category_id',
                'item_price', 'item_rating', 'item_rating_reviews_count',
                'item_latitude', 'item_longitude', 'item_is_phone_hidden', 'item_is_message_forbidden']
        self.docs = pd.read_parquet(WORK/'corpus.parquet', columns=columns)
        self.texts = pd.read_parquet(WORK/'texts.parquet')
        self.n = len(self.docs)
        self.id_to_row = {v: j for j, v in enumerate(self.docs.item_id)}
        self.document_fields = {c: self.docs[c].fillna(0).to_numpy() for c in columns if c != 'item_id'}
        self.centers = self.docs.groupby('item_location_id')[['item_latitude', 'item_longitude']].median().to_dict('index')
        self.init_history()

    def init_history(self):
        aux = pd.read_parquet(WORK/'aux.parquet')
        groups = aux.groupby('text_group', sort=True)
        self.hist_texts = list(groups.groups)
        self.hist_titles = groups.item_title_raw.agg(lambda values: list(dict.fromkeys(values.dropna()))[:4]).tolist()
        history_rows = {query: j for j, query in enumerate(self.hist_texts)}
        micros = sorted(self.docs.item_microcat_id.dropna().unique())
        category_columns = {m: j for j, m in enumerate(micros)}
        self.doc_micro = np.array([category_columns.get(m, 0) for m in self.docs.item_microcat_id])
        rows = aux.text_group.map(history_rows).to_numpy()
        category_ids = aux.item_microcat_id.map(category_columns).fillna(-1).to_numpy(dtype=int)
        mask = category_ids >= 0
        self.hist_micro = sparse.csr_matrix((np.ones(mask.sum(), dtype='float32'), (rows[mask], category_ids[mask])), shape=(len(history_rows), len(category_columns)))
        counts = np.asarray(self.hist_micro.sum(axis=1)).ravel()
        self.hist_micro = sparse.diags(1/np.maximum(counts, 1)) @ self.hist_micro
        self.micro_prior = (np.bincount(self.doc_micro, minlength=len(category_columns))+1)/self.n
        clicked = aux.item_id.map(self.id_to_row).fillna(-1).to_numpy(dtype=int)
        mask = clicked >= 0
        self.hist_clicks = sparse.csr_matrix((np.ones(mask.sum(), dtype='float32'), (rows[mask], clicked[mask])), shape=(len(history_rows), self.n))
        self.popularity = np.bincount(clicked[mask], minlength=self.n).astype('float32')
        self.location_counts = aux.groupby(['search_location_id', 'item_location_id']).size().to_dict()
        self.location_totals = aux.groupby('search_location_id').size().to_dict()
        # Похожие обучающие запросы дают словарь услуг и вероятности подкатегорий.
        path = WORK/'history_index.joblib'
        if path.exists():
            self.history_word_vectorizer, self.history_char_vectorizer, self.history_word_matrix, self.history_char_matrix = joblib.load(path, mmap_mode='r')
        else:
            self.history_word_vectorizer = TfidfVectorizer(dtype=np.float32, ngram_range=(1, 2), max_features=120000)
            self.history_word_matrix = self.history_word_vectorizer.fit_transform([normalize(text) for text in self.hist_texts]).T.tocsr()
            self.history_char_vectorizer = TfidfVectorizer(dtype=np.float32, analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=100000)
            self.history_char_matrix = self.history_char_vectorizer.fit_transform(self.hist_texts).T.tocsr()
            joblib.dump((self.history_word_vectorizer, self.history_char_vectorizer, self.history_word_matrix, self.history_char_matrix), path)
        del aux, groups
        gc.collect()
        log(f'История готова: {len(self.hist_texts)} различных запросов')

    def history(self, raw, normalized):
        similarity = (.45*(self.history_word_vectorizer.transform([normalized]) @ self.history_word_matrix).toarray().ravel()
               +.55*(self.history_char_vectorizer.transform([raw]) @ self.history_char_matrix).toarray().ravel())
        ids = topk(similarity, 25)
        ids = ids[similarity[ids] > .28]
        micro = np.zeros(self.hist_micro.shape[1], dtype='float32')
        clicks = np.zeros(self.n, dtype='float32')
        expansion = np.zeros(self.n, dtype='float32')
        if len(ids):
            weights = similarity[ids]**6
            weights /= weights.sum()
            micro = np.asarray(weights @ self.hist_micro[ids]).ravel()
            clicks = np.asarray(weights @ self.hist_clicks[ids]).ravel()
            expansion_ids = ids[:4]
            title_texts, title_weights = [], []
            for j in expansion_ids:
                titles = self.hist_titles[j]
                for text in titles:
                    title_texts.append(normalize(text))
                    title_weights.append(float(similarity[j]**6)/max(len(titles), 1))
            if title_texts:
                weights = np.array(title_weights, dtype='float32')
                weights /= weights.sum()
                index = self.indices['title_word']
                vector = sparse.csr_matrix(weights.reshape(1, -1)) @ index['vectorizer'].transform(title_texts)
                expansion = (vector @ index['inverted']).toarray().ravel()
        return micro[self.doc_micro], clicks, expansion, float(similarity.max(initial=0))

    def features(self, query, allow=None, force=()):
        raw, normalized = clean(query['search_query']), normalize(query['search_query'])
        filters = normalize(query['search_infm_params_text'])
        text_scores = {}
        for n in ENGINES:
            text = raw if n == 'title_char' else filters if n == 'params' else normalized
            text_scores[n] = score_index(self.indices[n], text)
        micro, clicks, expansion, history_similarity = self.history(raw, normalized)
        text_scores['expansion'] = expansion
        location = int(query['search_location_id'])
        same = self.document_fields['item_location_id'] == location
        center = self.centers.get(location)
        if center and abs(center['item_latitude']) > 1:
            # Приближённого расстояния достаточно для признака ранжирования.
            lat = self.document_fields['item_latitude'].astype('float32')
            lon = self.document_fields['item_longitude'].astype('float32')
            distance = 111*np.sqrt((lat-center['item_latitude'])**2 +
                              ((lon-center['item_longitude'])*np.cos(np.deg2rad(center['item_latitude'])))**2)
            distance[(abs(lat)<1)|(abs(lon)<1)] = 20000
        else:
            distance = np.full(self.n, 20000, dtype='float32')
        distance[same] = 0
        geo_weight = .16 + .44*np.exp(-distance/60) + .4*same
        text_score = .35*text_scores['title_word'] + .25*text_scores['title_char'] + .15*text_scores['title_bm25'] + .25*text_scores['desc_bm25']
        text_scores['base'] = text_score*geo_weight + .07*text_scores['params']*geo_weight
        text_scores['expanded_base'] = (.65*text_score+.2*expansion+.15*micro)*geo_weight+.07*text_scores['params']*geo_weight
        # Сохраняем и близкие, и удалённые объявления: в данных есть онлайн-услуги.
        pool = []
        fusion = np.zeros(self.n, dtype='float32')
        for n in ['title_word', 'title_char', 'desc_bm25', 'expansion', 'base', 'expanded_base']:
            for local, k in [(False, self.global_limit), (True, self.local_limit)]:
                values = text_scores[n] * (geo_weight if local and n not in ['base', 'expanded_base'] else 1)
                if allow is not None:
                    values = np.where(allow, values, -np.inf)
                ids = topk(values, k)
                if allow is not None:
                    ids = ids[allow[ids]]
                pool.extend(ids)
                fusion[ids] += 1/(30+np.arange(len(ids)))
        ids = np.unique(pool)
        if len(ids) > self.pool_limit:
            ids = ids[topk(fusion[ids], self.pool_limit)]
        if history_similarity > .8 and clicks.max(initial=0) > 0:
            valid_clicks = clicks if allow is None else np.where(allow, clicks, 0)
            extra = topk(valid_clicks, 20)
            ids = np.union1d(ids, extra[valid_clicks[extra] > 0])
        if force:
            ids = np.union1d(ids, np.array(force, dtype='int32'))
        if allow is not None:
            ids = ids[allow[ids]]
        ids = ids.astype('int32')
        feature_values = {}
        for name, scores in text_scores.items():
            feature_values[name] = scores[ids]
            feature_values[name+'_relative'] = scores[ids]/max(float(scores.max()), 1e-6)
        feature_values['fusion'] = fusion[ids]
        feature_values['same_location'] = same[ids].astype(float)
        feature_values['distance_log'] = np.log1p(distance[ids])
        feature_values['category_match'] = (self.document_fields['item_category_id'][ids] == query['search_category']).astype(float)
        feature_values['micro_probability'] = micro[ids]
        feature_values['micro_lift'] = micro[ids]/np.sqrt(self.micro_prior[self.doc_micro[ids]])
        feature_values['history_similarity'] = np.full(len(ids), history_similarity)
        feature_values['history_clicks'] = np.log1p(clicks[ids])
        feature_values['popularity'] = np.log1p(self.popularity[ids])
        location_counts = np.array([self.location_counts.get((location, l), 0) for l in self.document_fields['item_location_id'][ids]])
        feature_values['location_count'] = np.log1p(location_counts)
        feature_values['location_probability'] = location_counts/max(self.location_totals.get(location, 0), 1)
        tokens = set(normalized.split())
        filter_tokens = set(filters.split())
        for column in ['title', 'desc', 'params']:
            texts = self.texts[column].iloc[ids].tolist()
            sets = [set(text.split()) for text in texts]
            feature_values[column+'_coverage'] = np.array([len(tokens&values)/max(len(tokens), 1) for values in sets])
            feature_values[column+'_filter_coverage'] = np.array([len(filter_tokens&values)/max(len(filter_tokens), 1) for values in sets])
            feature_values[column+'_phrase'] = np.array([float(bool(normalized) and normalized in text) for text in texts])
            feature_values[column+'_length'] = np.log1p([len(text) for text in texts])
        feature_values['query_words'] = np.full(len(ids), len(tokens))
        feature_values['query_chars'] = np.full(len(ids), len(raw))
        feature_values['filter_words'] = np.full(len(ids), len(filter_tokens))
        feature_values['delivery'] = np.full(len(ids), query['search_is_delivery_search'])
        for column in ['item_price', 'item_rating_reviews_count']:
            feature_values[column] = np.log1p(np.maximum(self.document_fields[column][ids].astype('float32'), 0))
        for column in ['item_rating', 'item_is_phone_hidden', 'item_is_message_forbidden']:
            feature_values[column] = self.document_fields[column][ids].astype('float32')
        rating = re.search(r'рейтинг пользователя\s+(\d)', str(query['search_infm_params_text']).lower())
        feature_values['rating_filter_pass'] = (self.document_fields['item_rating'][ids] >= int(rating.group(1))).astype(float) if rating else np.ones(len(ids))
        feature_matrix = np.column_stack(list(feature_values.values())).astype('float32')
        return ids, np.nan_to_num(feature_matrix, nan=0, posinf=1e6, neginf=-1e6), list(feature_values)
