"""Подготовка упрощённой версии архива; служебный файл вне поставки."""
import ast
import io
from pathlib import Path
import tokenize

root = Path(__file__).resolve().parent / 'retrieval_solution'
source = root.parents[1] / 'retrieval_solution/work/cleanup_sources.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
changes = next(ast.literal_eval(n.value) for n in tree.body
               if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'changes' for t in n.targets))
removed = ['analyze_errors.py', 'check_pipeline.py', 'verify_reproduction.py',
           'validation_report.json', 'feature_importance.csv', 'answer_validation.json',
           'SOLUTION.md', 'error_analysis.json', 'reproduction_report.json',
           'format_audit.json', 'file_hashes.json']
for name in removed:
    (root / name).unlink()
for name, replacements in changes.items():
    path = root / name
    if not path.exists():
        continue
    text = path.read_text(encoding='utf-8')
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text, encoding='utf-8')

path = root / 'retrieval.py'
text = path.read_text(encoding='utf-8')
text = text.replace('ids = topk(s, k)', 'ids = topk(s, k)\n                if allow is not None:\n                    ids = ids[allow[ids]]')
text = text.replace("ids = ids.astype('int32')", "if allow is not None:\n            ids = ids[allow[ids]]\n        ids = ids.astype('int32')")
path.write_text(text, encoding='utf-8')

path = root / 'generate_features.py'
text = path.read_text(encoding='utf-8').replace("qs = pd.read_parquet(WORK/'queries.parquet')", "qs = pd.read_parquet(WORK/'queries.parquet').iloc[:3700]")
path.write_text(text, encoding='utf-8')

# Меняем только переменные, сохраняя строковые имена признаков модели.
renames = {
    'build_index.py': {'vec': 'vectorizer', 'x': 'matrix', 'df': 'document_frequency', 'norm': 'length_norm', 'a': 'start', 'z': 'end', 'lo': 'left', 'hi': 'right', 'den': 'denominator', 'out': 'texts_frame', 'col': 'column', 'make': 'build', 't': 'texts'},
    'prepare.py': {'meta': 'metadata', 'q': 'queries', 'aux': 'history', 'held': 'reserved_texts', 'bench': 'benchmark', 'cols': 'columns', 'c': 'column'},
    'generate_features.py': {'r': 'retriever', 'qs': 'queries', 'q': 'query', 'x': 'features', 'y': 'targets', 'xs': 'feature_blocks', 'ys': 'target_blocks', 'eids': 'example_ids', 'out': 'output_dir'},
    'retrieval.py': {'q': 'query', 'norm': 'normalized', 'filt': 'filters', 'score': 'text_scores', 'f': 'feature_values', 'x': 'feature_matrix', 'arr': 'document_fields', 'hmap': 'history_rows', 'mmap': 'category_columns', 'cs': 'category_ids', 'sim': 'similarity', 'hsim': 'history_similarity', 'w': 'weights', 'exids': 'expansion_ids', 'idx': 'index', 'loc': 'location', 'dist': 'distance', 'geo': 'geo_weight', 'textmix': 'text_score', 'lc': 'location_counts', 'cols': 'columns', 'col': 'column', 's': 'values', 't': 'text', 'hword': 'history_word_vectorizer', 'hchar': 'history_char_vectorizer', 'hwm': 'history_word_matrix', 'hcm': 'history_char_matrix'},
}
for name, mapping in renames.items():
    path = root / name
    tokens = list(tokenize.generate_tokens(io.StringIO(path.read_text(encoding='utf-8')).readline))
    tokens = [token._replace(string=mapping.get(token.string, token.string)) if token.type == tokenize.NAME else token for token in tokens]
    text = tokenize.untokenize(tokens)
    text = text.replace('{start+len(xs)}/{len(qs)}', '{start+len(feature_blocks)}/{len(queries)}')
    text = text.replace('{q.split.value_counts().to_dict()}', '{queries.split.value_counts().to_dict()}').replace('{len(aux)}', '{len(history)}')
    path.write_text(text, encoding='utf-8')

files = {}
files['predict.py'] = '''"""Ранжирование объявлений сохранённой моделью и запись answer.csv."""
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
            raise ValueError("Порядок признаков не соответствует сохранённой модели")
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
    log("Ответ сохранён в answer.csv")


if __name__ == "__main__":
    main()
'''
files['train_ranker.py'] = '''"""Обучение ранжирующей модели с ранее выбранными параметрами."""
import json
import lightgbm as lgb
import numpy as np
from common import ROOT, WORK, log


def load_training_data():
    """Первые 3700 запросов совпадают с итоговой обучающей выборкой исходника."""
    blocks = []
    for start in range(0, 3700, 100):
        with np.load(WORK / "features" / f"{start:05d}.npz") as block:
            blocks.append({key: block[key] for key in ("X", "y", "groups")})
    return {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}


def main():
    names = json.loads((WORK / "feature_names.json").read_text(encoding="utf-8"))
    training = load_training_data()
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="None", n_estimators=118,
        learning_rate=0.045, num_leaves=23, max_depth=-1,
        min_child_samples=100, colsample_bytree=0.9, reg_lambda=5,
        lambdarank_truncation_level=60, n_jobs=4, verbosity=-1,
        random_state=731, deterministic=True, force_col_wise=True,
    )
    model.fit(training["X"], training["y"], group=training["groups"], feature_name=names)
    # Передача текста через Python поддерживает кириллицу в пути Windows.
    (ROOT / "ranker.txt").write_text(model.booster_.model_to_string(), encoding="utf-8")
    config = {"model_weight": 1, "baseline_blend": 0, "features": names}
    (ROOT / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log("Модель сохранена: 118 деревьев")


if __name__ == "__main__":
    main()
'''
files['package_solution.py'] = '''"""Сборка архива с кодом, моделью и готовым ответом."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
FILES = [
    "common.py", "prepare.py", "build_index.py", "retrieval.py",
    "generate_features.py", "train_ranker.py", "predict.py", "run.py",
    "package_solution.py", "README.md", "requirements.txt",
    "ranker.txt", "model_config.json", "answer.csv",
]


def main():
    temporary = ROOT / "submission.tmp.zip"
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for name in FILES:
            archive.write(ROOT / name, "retrieval_solution/" + name)
    temporary.replace(ROOT / "submission.zip")
    print("Архив сохранён:", ROOT / "submission.zip")


if __name__ == "__main__":
    main()
'''
for name, text in files.items():
    (root / name).write_text(text, encoding='utf-8')
