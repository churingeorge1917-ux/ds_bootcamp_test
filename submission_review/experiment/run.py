"""Последовательный запуск подготовки данных, обучения и предсказаний."""
from pathlib import Path
import argparse
import subprocess
import sys
import zipfile

def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--predict-only', action='store_true')
    mode.add_argument('--use-model', action='store_true')
    args = parser.parse_args()
    if args.archive:
        data = root/'data'
        data.mkdir(exist_ok=True)
        with zipfile.ZipFile(args.archive) as z:
            # Распаковываем только три исходных файла данных.
            for name in ['train.parquet', 'benchmark_queries.parquet', 'benchmark_items.parquet']:
                if not (data/name).exists():
                    with z.open(name) as src, (data/name).open('wb') as dst:
                        import shutil
                        shutil.copyfileobj(src, dst)
    if args.predict_only:
        stages = ['predict.py']
    elif args.use_model:
        stages = ['prepare.py', 'build_index.py', 'predict.py']
    else:
        stages = ['prepare.py', 'build_index.py', 'generate_features.py', 'train_ranker.py', 'predict.py']
    for stage in stages:
        subprocess.run([sys.executable, '-u', str(root/stage)], check=True, cwd=root)

if __name__ == '__main__':
    main()
