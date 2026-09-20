"""Сборка архива с кодом, моделью и готовым ответом."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
FILES = [
    "common.py", "prepare.py", "build_index.py", "retrieval.py",
    "generate_features.py", "train_ranker.py", "predict.py", "run.py",
    "package_solution.py", "README.md", ".gitignore", "requirements.txt",
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
