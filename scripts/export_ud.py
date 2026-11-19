"""
Обучающие и проверочные части трибанков Universal Dependencies для модели ruSyntax

Описание:
    Скачивает части train и dev UD_Russian-Taiga, UD_Russian-GSD и UD_Russian-Poetry
    (CC BY-SA 4.0) закрепленного релиза UD и проверяет их SHA-256. UD_Russian-SynTagRus
    (CC BY-NC-SA 4.0) для обучения не берется. Скачанные файлы с верной суммой повторно
    не скачиваются; рядом пишется описание источников (sources.json).

Пример использования:
    uv run python scripts/export_ud.py target/ud
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

RELEASE = "r2.19"
LICENSE = "CC BY-SA 4.0"
# Трибанк, коммит релиза и файлы с SHA-256
SOURCES = {
    "UD_Russian-Taiga": (
        "25ec1a29bedcb51b8ca48f31b26f2cd8818739db",
        {
            "ru_taiga-ud-train-a.conllu": "48205ec38c2d6c530415dc27f1277e37db296b8ed37b0acfbed7586ce7d0351f",
            "ru_taiga-ud-train-b.conllu": "e207b0518bbcb12f6325e99a726a5a46c2c98876c2dc4b0f0cf8d72791e06078",
            "ru_taiga-ud-train-c.conllu": "3858143d8b5330cca78c1a73031f89cb8e6b9181d521c60d36195f9eee113146",
            "ru_taiga-ud-train-d.conllu": "24f520ec5e374ff8ef5660ba83c37ee658c35e953c83a825f296225e88bb04a7",
            "ru_taiga-ud-train-e.conllu": "56f197a7c36f481ade530001d035e498819a628923d2b4a1f594aab274ccb311",
            "ru_taiga-ud-dev.conllu": "b7200a90860ea0b1ef61cafa77e24ad195bf7ac91b1402e8aa48673124a67603",
        },
    ),
    "UD_Russian-GSD": (
        "486a9c07204286d630944a905bdf854a6f471f20",
        {
            "ru_gsd-ud-train.conllu": "f434e52cdc24a154df0de850b86ea0fc829d51f11d33211fa09753d3e3a35584",
            "ru_gsd-ud-dev.conllu": "c83c6910d30e285adc3981693b03b360059367f18377fe0b95734a12b0a4f904",
        },
    ),
    "UD_Russian-Poetry": (
        "137976dc150d0b3e137fe5fc1a79dc937967a640",
        {
            "ru_poetry-ud-train.conllu": "219afa378b2f0fe34cd999673fc05b4ba1d9015b039cbc1a10f748f1ef05da03",
            "ru_poetry-ud-dev.conllu": "b05492c99596380c025617376167d05ef3661bf48f9156d3880e564eb2a24e66",
        },
    ),
}


def sha256(path: Path) -> str:
    """
    SHA-256 файла

    Аргументы:
        path (Path): Файл

    Вывод:
        str: Сумма в шестнадцатеричной записи
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path, digest: str) -> None:
    """
    Скачивание файла с проверкой SHA-256; файл с верной суммой не скачивается повторно

    Аргументы:
        url (str): Адрес
        path (Path): Куда записать
        digest (str): Ожидаемая сумма SHA-256

    Исключения:
        SystemExit: Сумма скачанного файла не совпадает
    """
    if path.exists() and sha256(path) == digest:
        return
    partial = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as file:
        while chunk := response.read(1 << 20):
            file.write(chunk)
    if sha256(partial) != digest:
        partial.unlink()
        raise SystemExit(f"SHA-256 не совпадает: {url}")
    partial.replace(path)


def export(output: Path) -> None:
    """
    Скачивание частей трибанков и описание источников в каталог

    Аргументы:
        output (Path): Каталог выгрузки
    """
    output.mkdir(parents=True, exist_ok=True)
    for treebank, (commit, files) in SOURCES.items():
        for name, digest in files.items():
            url = f"https://raw.githubusercontent.com/UniversalDependencies/{treebank}/{commit}/{name}"
            download(url, output / name, digest)
    sources = {
        "release": RELEASE,
        "license": LICENSE,
        "treebanks": {treebank: commit for treebank, (commit, _) in SOURCES.items()},
        "files": {name: digest for _, files in SOURCES.values() for name, digest in files.items()},
    }
    (output / "sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("output", type=Path, help="каталог выгрузки")
    export(parser.parse_args().output)


if __name__ == "__main__":
    main()
