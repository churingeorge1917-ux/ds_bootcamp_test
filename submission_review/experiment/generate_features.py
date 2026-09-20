"""Расчёт кандидатов и признаков блоками с возможностью продолжения."""
from common import WORK, log
import numpy as np
import json
import pandas as pd
from retrieval import Retriever

def main():
    retriever = Retriever()
    queries = pd.read_parquet(WORK/'queries.parquet')
    label_frame = pd.read_parquet(WORK/'labels.parquet')
    labels = label_frame.groupby('example_id').item_id.agg(list).to_dict()
    output_dir = WORK/'features'
    output_dir.mkdir(exist_ok=True)
    for start in range(0, len(queries), 100):
        path = output_dir/f'{start:05d}.npz'
        if path.exists():
            continue
        retriever.pool_limit = 1000 if 3000 <= start < 4500 else 200
        feature_blocks = []
        target_blocks = []
        group_sizes = []
        denominators = []
        for query in queries.iloc[start:start+100].to_dict('records'):
            positive = {retriever.id_to_row[v] for v in labels[query['example_id']]}
            # Добавляем пропущенные положительные примеры только при обучении.
            force = list(positive) if query['split'] == 'rank_train' else []
            ids, features, names = retriever.features(query, force=force)
            targets = np.isin(ids, list(positive)).astype('int8')
            feature_blocks.append(features)
            target_blocks.append(targets)
            group_sizes.append(len(ids))
            denominators.append(len(positive))
        np.savez_compressed(
            path,
            X=np.concatenate(feature_blocks),
            y=np.concatenate(target_blocks),
            groups=np.array(group_sizes),
            denominators=np.array(denominators),
        )
        (WORK/'feature_names.json').write_text(json.dumps(names, indent=2), encoding='utf-8')
        log(f'Признаки рассчитаны: {start+len(feature_blocks)}/{len(queries)} запросов')

if __name__ == '__main__':
    main()
