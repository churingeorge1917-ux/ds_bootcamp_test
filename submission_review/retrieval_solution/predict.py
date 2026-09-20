"""Ранжирование объявлений сохранённой моделью и запись answer.csv."""
import json
import lightgbm as lgb
import pandas as pd
from common import DATA, ROOT, log, topk
from retrieval import Retriever


def main():
    retriever = Retriever()
    queries = pd.read_parquet(DATA / "benchmark_queries.parquet")
    benchmark = pd.read_parquet(DATA / "benchmark_items.parquet", columns=["item_id"])
    allowed = retriever.docs.item_id.isin(benchmark.item_id).to_numpy()
    config = json.loads((ROOT / "model_config.json").read_text(encoding="utf-8"))
    model = lgb.Booster(model_str=(ROOT / "ranker.txt").read_text(encoding="utf-8"))
    answers = []
    for number, query in enumerate(queries.to_dict("records"), start=1):
        document_ids, features, names = retriever.features(query, allow=allowed)
        if names != config["features"]:
            raise ValueError
        if len(document_ids):
            scores = model.predict(features, num_threads=4)
            selected = document_ids[topk(scores, 50)]
            answers.append(" ".join(retriever.docs.item_id.iloc[selected]))
        else:
            answers.append("")
        if number % 100 == 0:
            log(f"Обработано запросов: {number}/{len(queries)}")
    # Заменяем результат только после успешной записи всего файла.
    temporary = ROOT / "answer.tmp.csv"
    pd.DataFrame({"query_id": queries.query_id, "answer": answers}).to_csv(
        temporary, index=False, encoding="utf-8"
    )
    temporary.replace(ROOT / "answer.csv")
    


if __name__ == "__main__":
    main()
