# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "spacy==3.8.16",
#     "pymorphy3==2.0.6",
#     "pymorphy3-dicts-ru==2.4.417150.4580142",
#     "stanza==1.15.0",
#     "torch==2.14.1",
#     "transformers==4.57.6",
#     "slovnet==0.6.0",
#     "navec==0.10.0",
#     "razdel==0.5.0",
#     "ufal.udpipe==1.4.0.1",
# ]
# ///
"""
Сравнение ruSyntax с другими синтаксическими анализаторами: UAS и LAS на трибанках UD и скорость

Описание:
    Скачивает тестовые части UD_Russian-SynTagRus, UD_Russian-Taiga, UD_Russian-GSD,
    UD_Russian-PUD и UD_Russian-Poetry закрепленного релиза UD 2.19, официальный скрипт оценки
    CoNLL 2018 и модели систем (с проверкой SHA-256) в каталог данных. Модели Stanza скачивает
    сама Stanza: список файлов и их MD5 - resources_1.15.0.json из stanfordnlp/stanza-resources,
    файлы - с тега v1.15.0 репозитория stanfordnlp/stanza-ru на Hugging Face (версию задает
    версия пакета); после загрузки скрипт сверяет их SHA-256 (`STANZA_FILES`), а rubert-base-cased
    для default_accurate - коммит снимка Hugging Face (`BERT`).

    Каждая система разбирает наборы в двух режимах. tok - эталонные токены и границы
    предложений, своя морфология системы (у ruSyntax - теги ruMorph с контекстом). raw - сырой
    текст: `# text` предложений через пробел, абзацы (`# newpar`, `# newdoc`) через пустую
    строку; система получает текст по абзацам и делит его сама (у ruSyntax - parse_text: anySeg
    и ruMorph). Вершины, не образующие дерево с одним корнем, чинятся (`repair`), число правок
    пишется в результат. Оценка - evaluate скрипта CoNLL 2018: UAS и LAS, у сырого текста -
    еще F1 токенов и предложений. Качество мерится в основном процессе с числом потоков, которое
    выбирает библиотека.

    Скорость мерится после качества, каждая система - в отдельном процессе в один поток:
    слов в секунду на эталонных токенах SynTagRus test без загрузки, лучший из двух прогонов
    (у ruSyntax - вместе с тегами ruMorph; у медленных систем - первые предложения, `SPEED_WORDS`),
    время импорта и загрузки (вместе с разбором первого предложения: часть систем загружается
    при первом разборе), пик памяти процесса (RSS) и load average до и после замера.

    Пишет в каталог данных results.json (прогон части систем, `--systems`, обновляет только их
    результаты), environment.json, таблицы tables.md и tables.en.md и разборы в pred/;
    `--skip quality` или `--skip speed` пропускает замер, `--report` строит таблицы
    из results.json. Версии пакетов закреплены в compare.py.lock. ruSyntax мерится по колесу,
    собранному из репозитория: оно подается через `--with`, а rumorph и anyseg закрепляются там же
    (их версии пишутся в results.json).
    Сборка `--with-editable` отладочная, ее скорость не показательна.

Пример использования:
    rm -rf target/compare/wheels
    uvx maturin@1.15.0 build --release --locked -o target/compare/wheels
    uvx uv@0.13.5 run --with target/compare/wheels/rusyntax-*.whl --with rumorph==0.7.0 \\
        --with anyseg==0.5.0 scripts/compare.py target/compare
"""

import argparse
import functools
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

# Набор: трибанк, файл test, коммит релиза UD 2.19 и SHA-256
DATASETS = {
    "SynTagRus": (
        "UD_Russian-SynTagRus",
        "ru_syntagrus-ud-test.conllu",
        "26a8054a4ec7262af0d962534ee69e3c47218419",
        "ba12a7acbeeeb849bd58187268ba652a429ab076b89af0d2d05c3cb2265708e3",
    ),
    "Taiga": (
        "UD_Russian-Taiga",
        "ru_taiga-ud-test.conllu",
        "25ec1a29bedcb51b8ca48f31b26f2cd8818739db",
        "1f31210b17369074d526d1ccae6d41c0f1c34ddb4868dba4c52b74bb153bd48e",
    ),
    "GSD": (
        "UD_Russian-GSD",
        "ru_gsd-ud-test.conllu",
        "486a9c07204286d630944a905bdf854a6f471f20",
        "f0154fdc5a057de719bee82ae753bce2aba9ec99873d2aff4ae6c55310374c13",
    ),
    "PUD": (
        "UD_Russian-PUD",
        "ru_pud-ud-test.conllu",
        "878b61922328a8cecdec890a4352de981fcae5da",
        "777eb1776510d78550cb8a1ce2263ccc5c394c71bfad0cedd76a3be8d5a17350",
    ),
    "Poetry": (
        "UD_Russian-Poetry",
        "ru_poetry-ud-test.conllu",
        "137976dc150d0b3e137fe5fc1a79dc937967a640",
        "3d212abec2bc508fa56c0febbb62d50c094fc92136ebc17c0dcd5b1dcf99967a",
    ),
}
# Файлы: адрес и SHA-256
DOWNLOADS = {
    "conll18_ud_eval.py": (
        "https://universaldependencies.org/conll18/conll18_ud_eval.py",
        "1072e02af00b1a56205b5e8216d51dee9b8944a104d80744afaccc78859fcb16",
    ),
    "ru_core_news_sm-3.8.0-py3-none-any.whl": (
        "https://github.com/explosion/spacy-models/releases/download/ru_core_news_sm-3.8.0/"
        "ru_core_news_sm-3.8.0-py3-none-any.whl",
        "69978d47b43e2c4f329bebdb155e8e9d3861bba1a58ba25551419dae7d7e07fc",
    ),
    "navec_news_v1_1B_250K_300d_100q.tar": (
        "https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar",
        "f07270833d78523edc5781538d67038e95b43975e4a7ae757c693b687f9cbfca",
    ),
    "slovnet_syntax_news_v1.tar": (
        "https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_syntax_news_v1.tar",
        "0759ce82eaac4430f5b4ac27b2444aa6e4b6501ca38c2f45d62467265383d85e",
    ),
    # LINDAT, Universal Dependencies 2.5 Models for UDPipe (2019-12-06), hdl:11234/1-3131
    "russian-syntagrus-ud-2.5-191206.udpipe": (
        "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/"
        "f612687b-e000-4493-b078-b2d8fd9e008c/content",
        "16894d425ee328ace668a9d2f7ea5ff945468247d146333e8688f9200fa114e2",
    ),
}
# Файлы моделей Stanza 1.15.0 по пакетам и их SHA-256
STANZA_FILES = {
    "default": {
        "ru/tokenize/syntagrus_nocharlm.pt": "f33ec764f5e32d6b34ed200a71cc2beeb2132691a2369502d9c1f63122ef0f74",
        "ru/pos/syntagrus_charlm.pt": "7fc77bd096242d4cc86ff57d84d8bedf52b444f578d458c1f0e4b71a07c43a52",
        "ru/lemma/syntagrus_nocharlm.pt": "9a4649a361e7fa352441a92c740d286ad799bd457f230e54aa4a3f9d4d26ff16",
        "ru/depparse/syntagrus_charlm.pt": "cc3bff5a0b12ac856aa3842f49844957e486a4194dd31fb1aaf57824c0e70924",
        "ru/pretrain/conll17.pt": "35a5ab2fa207f484c467de547e42726c2d3a95812f6eae58504f34416565d8cb",
        "ru/forward_charlm/newswiki.pt": "6fc164c7ae8ba89b4e707844e75e0f5e78ddb8bb2d9a7ad61204e4a336b0fd93",
        "ru/backward_charlm/newswiki.pt": "eb447f476692399b2c742f2abcd17f2ba62cd9afabad0459bf468c58ce98c59f",
    },
    "default_accurate": {
        "ru/tokenize/syntagrus_charlm.pt": "6e07943ed9e5b4ad1cd4d4ea3269d4c82cbe1f97b7c0c1283ee63e214972dc96",
        "ru/pos/syntagrus_pavlov-rubert.pt": "49615a9e763d3b56b9537ff3d0648d72033658ed1a75e31e098963509fdd5496",
        "ru/lemma/syntagrus_charlm.pt": "570a1971dc70bafb8dd7613894213678879bba8c111a4295a558f1be53b0c0bc",
        "ru/depparse/syntagrus_pavlov-rubert.pt": "dedb9d5326c8f460b9f8a0d3dce458209466f792100780f525f1a737929af063",
        "ru/pretrain/conll17.pt": "35a5ab2fa207f484c467de547e42726c2d3a95812f6eae58504f34416565d8cb",
        "ru/forward_charlm/newswiki.pt": "6fc164c7ae8ba89b4e707844e75e0f5e78ddb8bb2d9a7ad61204e4a336b0fd93",
        "ru/backward_charlm/newswiki.pt": "eb447f476692399b2c742f2abcd17f2ba62cd9afabad0459bf468c58ce98c59f",
    },
}
# Трансформер Stanza default_accurate: репозиторий и коммит на Hugging Face
BERT = ("DeepPavlov/rubert-base-cased", "4036cab694767a299f2b9e6492909664d9414229")
STANZA_PROCESSORS = "tokenize,pos,lemma,depparse"
SPACY_MODEL = "ru_core_news_sm-3.8.0"
# Скорость медленных систем - на первых предложениях SynTagRus test, не меньше стольких слов
SPEED_WORDS = {"stanza-accurate": 20_000}
# Переменные числа потоков: задаются процессу замера скорости до запуска
THREADS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
MODES = ("tok", "raw")
METRICS = ("Tokens", "Sentences", "Words", "UAS", "LAS")


@dataclass
class Sentence:
    words: list[str]
    # Пробел после слова: нет SpaceAfter=No
    spaces: list[bool]
    text: str
    # Предложение начинает абзац или документ
    paragraph: bool


# Слово разбора: форма, вершина, тип связи
Word = tuple[str, int, str]
Parse = Callable[[list[Sentence]], object]
Convert = Callable[[list[Sentence], object], list[list[Word]]]


class System(NamedTuple):
    """Разбор эталонных токенов своим API системы (его скорость и мерится), перевод ответа
    в слова и разбор сырого текста по абзацам"""

    parse: Parse
    convert: Convert
    text: Callable[[list[str]], list[list[Word]]]


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
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path, digest: str) -> Path:
    """
    Скачивание файла с проверкой контрольной суммы; файл с верной суммой повторно
    не скачивается

    Аргументы:
        url (str): Адрес файла
        path (Path): Куда сохранить
        digest (str): SHA-256 файла

    Вывод:
        Path: Путь к файлу

    Исключения:
        ValueError: Если контрольная сумма не совпадает
    """
    if path.exists() and sha256(path) == digest:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "rusyntax-compare"})
    with urllib.request.urlopen(request, timeout=300) as response, part.open("wb") as file:
        while chunk := response.read(1 << 20):
            file.write(chunk)
    if sha256(part) != digest:
        part.unlink()
        raise ValueError(f"{url}: контрольная сумма не совпадает")
    part.replace(path)
    return path


def fetch(name: str, data: Path) -> Path:
    """Файл из `DOWNLOADS` в каталоге данных"""
    url, digest = DOWNLOADS[name]
    return download(url, data / name, digest)


def gold(dataset: str, data: Path) -> Path:
    """
    Тестовая часть набора в каталоге данных

    Аргументы:
        dataset (str): Набор
        data (Path): Каталог данных

    Вывод:
        Path: Файл CoNLL-U
    """
    treebank, name, commit, digest = DATASETS[dataset]
    url = f"https://raw.githubusercontent.com/UniversalDependencies/{treebank}/{commit}/{name}"
    return download(url, data / name, digest)


def sentences(path: Path) -> list[Sentence]:
    """
    Предложения CoNLL-U без многословных токенов и пустых узлов

    Аргументы:
        path (Path): Файл CoNLL-U

    Вывод:
        list[Sentence]: Предложения
    """
    result: list[Sentence] = []
    current = Sentence([], [], "", False)
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            line = line.rstrip("\n")
            if not line:
                if current.words:
                    result.append(current)
                current = Sentence([], [], "", False)
            elif line.startswith("#"):
                if line.startswith("# text = "):
                    current.text = line.removeprefix("# text = ")
                current.paragraph |= line.startswith(("# newpar", "# newdoc"))
            else:
                cols = line.split("\t")
                if cols[0].isdigit():
                    current.words.append(cols[1])
                    current.spaces.append("SpaceAfter=No" not in cols[9])
    if current.words:
        result.append(current)
    return result


def paragraphs(text: list[Sentence]) -> list[str]:
    """
    Сырой текст набора по абзацам: `# text` предложений через пробел, новый абзац -
    по `# newpar` и `# newdoc`

    Аргументы:
        text (list[Sentence]): Предложения

    Вывод:
        list[str]: Абзацы
    """
    result: list[list[str]] = []
    for sentence in text:
        if sentence.paragraph or not result:
            result.append([])
        result[-1].append(sentence.text)
    return [joined for part in result if (joined := " ".join(part).strip())]


def repair(heads: list[int], deprels: list[str]) -> int:
    """
    Дерево с одним корнем из вершин: петли и вершины вне предложения - в корни, лишние
    корни и узлы циклов - к первому корню с меткой root (или к первому корню)

    Аргументы:
        heads (list[int]): Вершины слов, меняются на месте
        deprels (list[str]): Типы связей, меняются на месте

    Вывод:
        int: Число правок
    """
    count = len(heads)
    fixes = 0
    for index, head in enumerate(heads):
        if head < 0 or head > count or head == index + 1:
            heads[index] = 0
            fixes += 1
    roots = [index for index, head in enumerate(heads) if head == 0]
    if not roots:
        root = next((i for i, deprel in enumerate(deprels) if deprel.lower() == "root"), 0)
        heads[root] = 0
        roots = [root]
        fixes += 1
    root = next((i for i in roots if deprels[i].lower() == "root"), roots[0])
    for index in roots:
        if index != root:
            heads[index] = root + 1
            deprels[index] = "dep" if deprels[index].lower() == "root" else deprels[index]
            fixes += 1
    for start in range(count):
        seen, node = set(), start
        while heads[node] != 0:
            if node in seen:
                heads[node] = root + 1
                fixes += 1
                break
            seen.add(node)
            node = heads[node] - 1
    deprels[root] = "root"
    return fixes


def write(path: Path, parsed: list[list[Word]]) -> int:
    """
    Разбор в файл CoNLL-U: форма, вершина и тип связи; вершины чинятся до дерева

    Аргументы:
        path (Path): Файл
        parsed (list[list[Word]]): Предложения

    Вывод:
        int: Число правок дерева
    """
    lines = []
    fixes = 0
    for sentence in parsed:
        if not sentence:
            continue
        heads = [head for _, head, _ in sentence]
        deprels = [deprel or "_" for _, _, deprel in sentence]
        fixes += repair(heads, deprels)
        for index, ((form, _, _), head, deprel) in enumerate(
            zip(sentence, heads, deprels, strict=True)
        ):
            lines.append(f"{index + 1}\t{form}\t_\t_\t_\t_\t{head}\t{deprel}\t_\t_")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fixes


def tok_words(text: list[Sentence], arcs: Iterator[list[tuple[int, str]]]) -> list[list[Word]]:
    """Эталонные формы с вершинами и связями системы; число слов должно совпасть"""
    return [
        [(form, head, deprel) for form, (head, deprel) in zip(s.words, a, strict=True)]
        for s, a in zip(text, arcs, strict=True)
    ]


def spacy_model(data: Path) -> Path:
    """Каталог модели spaCy в распакованном колесе"""
    return data / "spacy" / SPACY_MODEL.rsplit("-", 1)[0] / SPACY_MODEL


def stanza_ready(data: Path, package: str) -> bool:
    """Файлы модели Stanza на месте и с верными суммами"""
    folder = data / "stanza"
    files = STANZA_FILES[package]
    return (folder / "resources.json").exists() and all(
        (folder / name).exists() and sha256(folder / name) == digest
        for name, digest in files.items()
    )


def prepare(name: str, data: Path) -> None:
    """
    Модели системы в каталоге данных - до замеров

    Аргументы:
        name (str): Имя системы
        data (Path): Каталог данных

    Исключения:
        ValueError: Если контрольная сумма или коммит модели не совпадает
    """
    if name == "spacy" and not (spacy_model(data) / "meta.json").exists():
        wheel = fetch(f"{SPACY_MODEL}-py3-none-any.whl", data)
        with zipfile.ZipFile(wheel) as files:
            files.extractall(data / "spacy")
    if name in ("stanza", "stanza-accurate"):
        import stanza

        package = "default" if name == "stanza" else "default_accurate"
        if not stanza_ready(data, package):
            stanza.download(
                "ru",
                model_dir=str(data / "stanza"),
                package=package,
                processors=STANZA_PROCESSORS,
                verbose=False,
            )
            if not stanza_ready(data, package):
                raise ValueError(f"Stanza {package}: контрольная сумма модели не совпадает")
        if package == "default_accurate":
            from huggingface_hub import snapshot_download

            # Без загрузок Stanza берет трансформер из кеша по ссылке main: снимок - main
            snapshot = snapshot_download(
                BERT[0], allow_patterns=["*.json", "vocab.txt", "pytorch_model.bin"]
            )
            if Path(snapshot).name != BERT[1]:
                raise ValueError(f"{BERT[0]}: коммит {Path(snapshot).name}, ожидался {BERT[1]}")
    if name == "slovnet":
        fetch("navec_news_v1_1B_250K_300d_100q.tar", data)
        fetch("slovnet_syntax_news_v1.tar", data)
    if name == "udpipe":
        fetch("russian-syntagrus-ud-2.5-191206.udpipe", data)


def feats_string(feats: dict[str, str]) -> str:
    """Признаки ruMorph в записи CoNLL-U: по имени без учета регистра"""
    pairs = sorted(feats.items(), key=lambda item: item[0].lower())
    return "|".join(f"{key}={value}" for key, value in pairs) or "_"


def rusyntax_system(data: Path, threads: int | None) -> System:
    """ruSyntax: теги ruMorph с контекстом и разбор; сырой текст - parse_text"""
    from rumorph import Analyzer

    from rusyntax import Parser

    morph = Analyzer()
    parser = Parser()

    def parse(text: list[Sentence]) -> object:
        batch = []
        for sentence in text:
            best = [analyses[0] for analyses in morph.analyze_sentence(sentence.words)]
            pos = [a.pos for a in best]
            batch.append((sentence.words, pos, [feats_string(a.feats) for a in best]))
        return parser.parse_many(batch)

    def raw(paragraphs: list[str]) -> list[list[Word]]:
        return [
            [(w.form, w.head or 0, w.deprel or "_") for w in sentence.words]
            for paragraph in paragraphs
            for sentence in parser.parse_text(paragraph).sentences
        ]

    return System(parse, lambda text, found: tok_words(text, iter(found)), raw)


def spacy_words(span) -> list[Word]:
    """Слова предложения spaCy без пробельных токенов: их зависимые - к вершине пробела"""
    tokens = [token for token in span if not token.is_space]
    index = {token.i: number + 1 for number, token in enumerate(tokens)}

    def head(token) -> int:
        while token.head.i != token.i:
            if token.head.i in index:
                return index[token.head.i]
            token = token.head
        return 0

    return [(t.text, head(t), "root" if t.dep_ == "ROOT" else t.dep_) for t in tokens]


def spacy_system(data: Path, threads: int | None) -> System:
    """spaCy ru_core_news_sm без ner: один Doc на эталонное предложение, сырой текст - nlp.pipe"""
    import spacy
    from spacy.tokens import Doc

    nlp = spacy.load(spacy_model(data), exclude=["ner"])

    def parse(text: list[Sentence]) -> object:
        docs = [
            Doc(
                nlp.vocab,
                words=s.words,
                spaces=s.spaces,
                sent_starts=[True] + [False] * (len(s.words) - 1),
            )
            for s in text
        ]
        return list(nlp.pipe(docs))

    def convert(text: list[Sentence], docs: object) -> list[list[Word]]:
        arcs = ([(head, deprel) for _, head, deprel in spacy_words(doc[:])] for doc in docs)
        return tok_words(text, arcs)

    def raw(paragraphs: list[str]) -> list[list[Word]]:
        nlp.max_length = max(len(paragraph) for paragraph in paragraphs) + 1
        found = [spacy_words(sent) for doc in nlp.pipe(paragraphs) for sent in doc.sents]
        return [sentence for sentence in found if sentence]

    return System(parse, convert, raw)


def stanza_system(data: Path, threads: int | None, package: str = "default") -> System:
    """Stanza: tokenize, pos, lemma, depparse; эталонные токены - tokenize_pretokenized,
    сырой текст - абзацы отдельными документами (bulk_process)"""
    import stanza
    import torch
    from stanza.models.common.doc import Document

    if threads:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(threads)
    options = {
        "lang": "ru",
        "dir": str(data / "stanza"),
        "package": package,
        "processors": STANZA_PROCESSORS,
        "download_method": None,
        "use_gpu": False,
        "verbose": False,
    }
    tagged = stanza.Pipeline(tokenize_pretokenized=True, **options)

    def convert(text: list[Sentence], doc: object) -> list[list[Word]]:
        arcs = ([(w.head, w.deprel) for w in s.words] for s in doc.sentences)
        return tok_words(text, arcs)

    # Конвейер сырого текста - при первом разборе: замер скорости его не грузит
    @functools.cache
    def plain() -> stanza.Pipeline:
        return stanza.Pipeline(**options)

    def raw(paragraphs: list[str]) -> list[list[Word]]:
        docs = plain().bulk_process([Document([], text=paragraph) for paragraph in paragraphs])
        return [
            [(w.text, w.head, w.deprel) for w in sentence.words]
            for doc in docs
            for sentence in doc.sentences
        ]

    return System(lambda text: tagged([s.words for s in text]), convert, raw)


def stanza_accurate_system(data: Path, threads: int | None) -> System:
    """Stanza default_accurate: теги и разбор на rubert-base-cased"""
    return stanza_system(data, threads, "default_accurate")


def slovnet_system(data: Path, threads: int | None) -> System:
    """Slovnet (Natasha) syntax_news_v1 с navec; сырой текст - razdel, как в Natasha"""
    from navec import Navec
    from slovnet import Syntax

    syntax = Syntax.load(str(data / "slovnet_syntax_news_v1.tar"))
    syntax.navec(Navec.load(str(data / "navec_news_v1_1B_250K_300d_100q.tar")))

    def words(markups) -> list[list[Word]]:
        return [[(t.text, int(t.head_id), t.rel) for t in m.tokens] for m in markups]

    def convert(text: list[Sentence], markups: object) -> list[list[Word]]:
        arcs = ([(head, deprel) for _, head, deprel in s] for s in words(markups))
        return tok_words(text, arcs)

    def raw(paragraphs: list[str]) -> list[list[Word]]:
        from razdel import sentenize, tokenize

        batch = [
            [token.text for token in tokenize(sentence.text)]
            for paragraph in paragraphs
            for sentence in sentenize(paragraph)
        ]
        return words(syntax.map([tokens for tokens in batch if tokens]))

    return System(lambda text: list(syntax.map([s.words for s in text])), convert, raw)


def conllu_words(text: str) -> list[list[Word]]:
    """Слова предложений вывода CoNLL-U"""
    result: list[list[Word]] = [[]]
    for line in text.splitlines():
        cols = line.split("\t")
        if not line:
            if result[-1]:
                result.append([])
        elif not line.startswith("#") and cols[0].isdigit():
            result[-1].append((cols[1], int(cols[6]), cols[7]))
    return [sentence for sentence in result if sentence]


def udpipe_system(data: Path, threads: int | None) -> System:
    """UDPipe 1, модель SynTagRus UD 2.5: эталонные слова - tag и parse, сырой текст -
    Pipeline с токенизатором модели"""
    from ufal.udpipe import Model, Pipeline, ProcessingError, Sentence as UDPipeSentence

    path = data / "russian-syntagrus-ud-2.5-191206.udpipe"
    model = Model.load(str(path))
    if model is None:
        raise ValueError(f"{path}: модель не загрузилась")

    def parse(text: list[Sentence]) -> object:
        result = []
        for sentence in text:
            found = UDPipeSentence()
            for word in sentence.words:
                found.addWord(word)
            model.tag(found, Model.DEFAULT)
            model.parse(found, Model.DEFAULT)
            result.append(found)
        return result

    def convert(text: list[Sentence], found: object) -> list[list[Word]]:
        arcs = ([(w.head, w.deprel) for w in list(s.words)[1:]] for s in found)
        return tok_words(text, arcs)

    def raw(paragraphs: list[str]) -> list[list[Word]]:
        pipeline = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
        error = ProcessingError()
        result = []
        for paragraph in paragraphs:
            output = pipeline.process(paragraph, error)
            if error.occurred():
                raise RuntimeError(error.message)
            result += conllu_words(output)
        return result

    return System(parse, convert, raw)


SYSTEMS: dict[str, Callable[[Path, int | None], System]] = {
    "rusyntax": rusyntax_system,
    "spacy": spacy_system,
    "stanza": stanza_system,
    "stanza-accurate": stanza_accurate_system,
    "slovnet": slovnet_system,
    "udpipe": udpipe_system,
}
TITLES = {
    "rusyntax": "ruSyntax",
    "spacy": "spaCy ru_core_news_sm",
    "stanza": "Stanza",
    "stanza-accurate": "Stanza accurate",
    "slovnet": "Slovnet (Natasha)",
    "udpipe": "UDPipe 1",
}
MODELS = {
    "rusyntax": "встроенная модель, теги ruMorph, сегментация anySeg",
    "spacy": "ru_core_news_sm 3.8.0 без ner",
    "stanza": "default (SynTagRus, charlm), ресурсы 1.15.0",
    "stanza-accurate": f"default_accurate ({BERT[0]} {BERT[1][:7]}), ресурсы 1.15.0",
    "slovnet": "slovnet_syntax_news_v1, navec_news_v1_1B_250K_300d_100q; сырой текст - razdel",
    "udpipe": "russian-syntagrus-ud-2.5-191206",
}
PACKAGES = {
    "rusyntax": ("rusyntax", "rumorph", "anyseg"),
    "spacy": ("spacy", "thinc", "numpy", "pymorphy3"),
    "stanza": ("stanza", "torch", "numpy"),
    "stanza-accurate": ("stanza", "torch", "numpy", "transformers"),
    "slovnet": ("slovnet", "navec", "razdel", "numpy"),
    "udpipe": ("ufal.udpipe",),
}


def evaluator(data: Path) -> ModuleType:
    """
    Официальный скрипт оценки CoNLL 2018 как модуль

    Аргументы:
        data (Path): Каталог данных

    Вывод:
        ModuleType: Модуль conll18_ud_eval
    """
    path = fetch("conll18_ud_eval.py", data)
    spec = importlib.util.spec_from_file_location("conll18_ud_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    # Без __pycache__ в каталоге данных
    write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = write_bytecode
    return module


def quality(name: str, data: Path, ud: ModuleType, cache: dict[str, object]) -> dict:
    """
    UAS и LAS системы на всех наборах в режимах tok и raw

    Аргументы:
        name (str): Имя системы
        data (Path): Каталог данных
        ud (ModuleType): Скрипт оценки CoNLL 2018
        cache (dict[str, object]): Загруженные эталоны наборов, дополняется

    Вывод:
        dict: Режим - набор - метрики, число правок дерева и время разбора в секундах
    """
    system = SYSTEMS[name](data, None)
    result: dict = {mode: {} for mode in MODES}
    for dataset in DATASETS:
        path = gold(dataset, data)
        text = sentences(path)
        if dataset not in cache:
            cache[dataset] = ud.load_conllu_file(str(path))
        for mode in MODES:
            start = time.perf_counter()
            if mode == "tok":
                parsed = system.convert(text, system.parse(text))
            else:
                parsed = system.text(paragraphs(text))
            seconds = time.perf_counter() - start
            pred = data / "pred" / name / mode / f"{dataset}.conllu"
            fixes = write(pred, parsed)
            scores = ud.evaluate(cache[dataset], ud.load_conllu_file(str(pred)))
            result[mode][dataset] = {key: round(100 * scores[key].f1, 2) for key in METRICS}
            result[mode][dataset] |= {"repaired": fixes, "seconds": round(seconds, 2)}
            values = result[mode][dataset]
            print(
                f"{TITLES[name]}, {mode}, {dataset}: UAS {values['UAS']}, LAS {values['LAS']}, "
                f"{seconds:.1f} с",
                file=sys.stderr,
                flush=True,
            )
    return result


def speed_run(name: str, data: Path) -> None:
    """
    Замер в чистом процессе: время импорта и загрузки системы, первого разбора эталонных
    токенов SynTagRus test и лучшего из двух разборов, пик памяти процесса

    Аргументы:
        name (str): Имя системы
        data (Path): Каталог данных
    """
    text = sentences(data / DATASETS["SynTagRus"][1])
    if limit := SPEED_WORDS.get(name):
        count = 0
        for size, sentence in enumerate(text):
            count += len(sentence.words)
            if count >= limit:
                text = text[: size + 1]
                break
    base = peak_memory()
    start = time.perf_counter()
    system = SYSTEMS[name](data, 1)
    # Отложенная загрузка (словарь ruMorph, модель ruSyntax) - тоже загрузка
    system.parse(text[:1])
    loaded = time.perf_counter()
    system.parse(text)
    first = time.perf_counter()
    system.parse(text)
    best = min(first - loaded, time.perf_counter() - first)
    words = sum(len(sentence.words) for sentence in text)
    result = {
        "words": words,
        "sentences": len(text),
        "subset": name in SPEED_WORDS,
        "load": round(loaded - start, 3),
        "first": round(first - loaded, 3),
        "best": round(best, 3),
        "words_per_second": round(words / best),
        "memory": peak_memory(),
        "memory_before_load": base,
    }
    print(json.dumps(result))


def peak_memory() -> float | None:
    """Пик памяти процесса (RSS), МБ; None, если система его не дает"""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS дает байты, Linux - килобайты
    return round(peak / 2**20 if sys.platform == "darwin" else peak / 2**10, 1)


def load_average() -> list[float] | None:
    """Load average за 1, 5 и 15 минут; None, если система его не дает"""
    try:
        return [round(value, 2) for value in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def measure(name: str, data: Path) -> dict:
    """
    Замер `speed_run` в отдельном процессе в один поток

    Аргументы:
        name (str): Имя системы
        data (Path): Каталог данных

    Вывод:
        dict: Слова, время в секундах, слов в секунду, память в МБ и load average

    Исключения:
        RuntimeError: Если процесс замера упал
    """
    before = load_average()
    run = subprocess.run(
        [sys.executable, __file__, str(data), "--speed-run", name],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **THREADS, "PYTHONWARNINGS": "ignore"},
    )
    if run.returncode:
        raise RuntimeError(f"{name}: замер скорости упал\n{run.stderr[-3000:]}")
    result = json.loads(run.stdout.strip().splitlines()[-1])
    return result | {"threads": 1, "load_average": {"before": before, "after": load_average()}}


def number(value: float, digits: int, english: bool) -> str:
    """Число с разделителем разрядов: по-русски - пробел и запятая"""
    text = f"{value:,.{digits}f}"
    return text if english else text.replace(",", " ").replace(".", ",")


def tables(results: dict, english: bool = False) -> str:
    """
    Таблицы Markdown: UAS / LAS по режимам, токены и предложения сырого текста, скорость

    Аргументы:
        results (dict): Результаты систем
        english (bool): Подписи и числа по-английски

    Вывод:
        str: Таблицы
    """
    names = [name for name in SYSTEMS if name in results]
    if english:
        captions = {
            "tok": (
                "Gold tokenization",
                "UAS / LAS, %: gold tokens and sentence boundaries, own morphology of the system "
                "(ruMorph tags for ruSyntax)",
            ),
            "raw": (
                "Raw text",
                "UAS / LAS, %: own tokenization and sentence segmentation of the system "
                "(anySeg for ruSyntax)",
            ),
            "segmentation": ("Raw text: tokens and sentences", "Tokens / Sentences F1, %"),
        }
        speed_title = "Speed"
        speed_caption = (
            "One thread, gold tokens of SynTagRus test, best of two runs without loading "
            "(ruSyntax - together with ruMorph tags); peak memory - RSS of the process"
        )
        speed_header = ["System", "words/s", "import and load, s", "memory, MB", "words measured"]
        system = "System"
    else:
        captions = {
            "tok": (
                "Эталонные токены",
                "UAS / LAS, %: эталонные токены и границы предложений, своя морфология системы "
                "(у ruSyntax - теги ruMorph)",
            ),
            "raw": (
                "Сырой текст",
                "UAS / LAS, %: своя токенизация и деление на предложения (у ruSyntax - anySeg)",
            ),
            "segmentation": ("Сырой текст: токены и предложения", "F1 токенов / предложений, %"),
        }
        speed_title = "Скорость"
        speed_caption = (
            "Один поток, эталонные токены SynTagRus test, лучший из двух прогонов без загрузки "
            "(у ruSyntax - вместе с тегами ruMorph); память - пик RSS процесса"
        )
        speed_header = ["Система", "слов/с", "импорт и загрузка, с", "память, МБ", "слов в замере"]
        system = "Система"
    lines = []
    for key, (title, caption) in captions.items():
        mode, metrics = ("raw", ("Tokens", "Sentences")) if key == "segmentation" else (key, None)
        rows = [name for name in names if mode in results[name]]
        if not rows:
            continue
        lines += [f"## {title}", "", caption, ""]
        lines.append(f"| {system} | " + " | ".join(DATASETS) + " |")
        lines.append("| :-- | " + " | ".join("--:" for _ in DATASETS) + " |")
        for name in rows:
            cells = []
            for dataset in DATASETS:
                values = results[name][mode][dataset]
                first, second = metrics or ("UAS", "LAS")
                pair = (number(values[first], 1, english), number(values[second], 1, english))
                cells.append(" / ".join(pair))
            lines.append(f"| {TITLES[name]} | " + " | ".join(cells) + " |")
        lines.append("")
    rows = [name for name in names if "speed" in results[name]]
    if rows:
        lines += [f"## {speed_title}", "", speed_caption, ""]
        lines.append("| " + " | ".join(speed_header) + " |")
        lines.append("| :-- | " + " | ".join("--:" for _ in speed_header[1:]) + " |")
        for name in rows:
            speed = results[name]["speed"]
            memory = "-" if speed["memory"] is None else number(speed["memory"], 0, english)
            cells = [
                TITLES[name],
                number(speed["words_per_second"], 0, english),
                number(speed["load"], 2, english),
                memory,
                number(speed["words"], 0, english),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def versions(packages: tuple[str, ...]) -> dict[str, str | None]:
    """Версии пакетов; None - пакет не установлен"""
    found: dict[str, str | None] = {}
    for package in packages:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            found[package] = None
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("data", type=Path, help="каталог данных")
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    parser.add_argument(
        "--skip", choices=("quality", "speed"), help="не мерить качество или скорость"
    )
    parser.add_argument(
        "--report", action="store_true", help="таблицы из results.json без замеров"
    )
    parser.add_argument("--speed-run", metavar="SYSTEM", help=argparse.SUPPRESS)
    args = parser.parse_args()
    data: Path = args.data.resolve()
    # Кеш Hugging Face (модели Stanza и rubert-base-cased) - в каталоге данных
    os.environ["HF_HOME"] = str(data / "stanza" / "hf")
    if args.speed_run:
        speed_run(args.speed_run, data)
        return
    saved = data / "results.json"
    results = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else {}

    def save() -> None:
        saved.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.report:
        data.mkdir(parents=True, exist_ok=True)
        for name in args.systems:
            prepare(name, data)
        # Прогон части систем обновляет только их результаты
        if args.skip != "quality":
            ud = evaluator(data)
            cache: dict[str, object] = {}
            for name in args.systems:
                entry = results.setdefault(name, {})
                entry |= quality(name, data, ud, cache)
                entry |= {"model": MODELS[name], "versions": versions(PACKAGES[name])}
                save()
        # Скорость - после качества, когда другие замеры не нагружают машину
        if args.skip != "speed":
            gold("SynTagRus", data)
            for name in args.systems:
                results.setdefault(name, {})["speed"] = measure(name, data)
                speed = results[name]["speed"]
                print(
                    f"{TITLES[name]}: {speed['words_per_second']:,} слов/с, "
                    f"load average {speed['load_average']['before']}",
                    file=sys.stderr,
                    flush=True,
                )
                save()
        environment = {"python": sys.version.split()[0], "platform": platform.platform()}
        environment |= {"machine": platform.machine(), "processor": platform.processor()}
        packages = sorted({package for name in SYSTEMS for package in PACKAGES[name]})
        environment |= versions(tuple(packages))
        (data / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (data / "tables.md").write_text(tables(results), encoding="utf-8")
    (data / "tables.en.md").write_text(tables(results, english=True), encoding="utf-8")
    print(tables(results))


if __name__ == "__main__":
    main()
