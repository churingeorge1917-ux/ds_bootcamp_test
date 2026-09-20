"""Разбиение по текстам запросов без пересечений и подготовка корпуса.

Сохраняем все положительные пары выбранного контекста. Их тексты исключаем
из вспомогательной статистики. Бенчмарк предоставляет только признаки.
"""
from common import DATA, WORK, QUERY_COLS, clean, log
import numpy as np
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    log('Чтение запросов и выбранных объявлений')
    metadata = pd.read_parquet(DATA/'train.parquet', columns=QUERY_COLS + [
        'item_id', 'item_microcat_id', 'item_location_id', 'item_title_raw'])
    metadata['text_group'] = metadata.search_query.map(clean)
    contexts = metadata.drop_duplicates(QUERY_COLS).copy()
    # Выбираем тексты равновероятно: частые запросы не должны определять метрику.
    contexts = contexts.sample(frac=1, random_state=731).drop_duplicates('text_group')
    contexts = contexts.sample(frac=1, random_state=732)
    selected = contexts.iloc[:7500].copy()
    selected['split'] = np.repeat(['rank_train', 'dev', 'test', 'rank_train'], [3000, 700, 800, 3000])
    selected['example_id'] = np.arange(len(selected))
    queries = selected[QUERY_COLS+['text_group', 'split', 'example_id']]
    pairs = metadata.merge(queries, on=QUERY_COLS+['text_group'], how='inner')
    pairs = pairs[['example_id', 'item_id']].drop_duplicates()
    queries.to_parquet(WORK/'queries.parquet', index=False)
    pairs.to_parquet(WORK/'labels.parquet', index=False)
    reserved_texts = set(queries.text_group)
    history = metadata[~metadata.text_group.isin(reserved_texts)].copy()
    history.to_parquet(WORK/'aux.parquet', index=False)
    log(f'Разбиение: {queries.split.value_counts().to_dict()}; вспомогательных пар: {len(history)}')
    needed = set(pairs.item_id)
    del metadata, contexts, selected, pairs, history
    # Из больших блоков описаний оставляем только нужные объявления.
    benchmark = pq.read_table(DATA/'benchmark_items.parquet')
    needed.difference_update(benchmark['item_id'].to_pylist())
    columns = benchmark.column_names
    chunks = [benchmark]
    for batch in pq.ParquetFile(DATA/'train.parquet').iter_batches(batch_size=8192, columns=columns):
        mask = pa.array([v in needed for v in batch.column('item_id').to_pylist()])
        if pa.compute.any(mask).as_py():
            chunks.append(pa.Table.from_batches([batch]).filter(mask))
    corpus = pa.concat_tables(chunks).to_pandas().drop_duplicates('item_id').reset_index(drop=True)
    # Для вычислений преобразуем десятичные поля в числа с плавающей точкой.
    for column in ['item_price', 'item_latitude', 'item_longitude']:
        corpus[column] = pd.to_numeric(corpus[column], errors='coerce').astype('float32')
    corpus.to_parquet(WORK/'corpus.parquet', index=False)
    info = {'benchmark_items': len(benchmark), 'experiment_items': len(corpus),
            'query_splits': queries.split.value_counts().to_dict(), 'seed': 731}
    (WORK/'split_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    log(f'Корпус готов: {len(corpus)} объявлений')

if __name__ == '__main__':
    main()
