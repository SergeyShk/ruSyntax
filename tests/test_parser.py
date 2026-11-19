import copy
import pickle
import shutil
import threading
from pathlib import Path

import pytest
from anyseg import Document, Segmenter
from rumorph import Analyzer

from rusyntax import Parser

ROOT = Path(__file__).parents[1]
MODEL = ROOT / "data" / "ru-syntax.bin"
WORDS = ["Мама", "мыла", "раму", "."]
UPOS = ["NOUN", "VERB", "NOUN", "PUNCT"]
FEATS = ["Case=Nom|Gender=Fem|Number=Sing", "Gender=Fem|Tense=Past", "Case=Acc", None]


@pytest.fixture(scope="module")
def parser():
    return Parser()


def is_tree(relations):
    heads = [head for head, _ in relations]
    if heads.count(0) != 1 or any(not 0 <= head <= len(heads) for head in heads):
        return False
    for start in range(1, len(heads) + 1):
        node, steps = start, 0
        while node and steps <= len(heads):
            node, steps = heads[node - 1], steps + 1
        if node:
            return False
    return True


def test_parse(parser):
    relations = parser.parse(WORDS, UPOS, FEATS)
    assert relations == [(2, "nsubj"), (0, "root"), (2, "obj"), (2, "punct")]
    assert all(deprel in parser.relations for _, deprel in relations)


def test_parse_empty_and_unknown(parser):
    assert parser.parse([], [], []) == []
    relations = parser.parse(["Ёжик", "", "zzz"], ["FOO", "_", "X"], ["Bar=Baz", "", "_"])
    assert is_tree(relations)


def test_parse_accepts_tuples_and_generators(parser):
    expected = parser.parse(WORDS, UPOS, FEATS)
    assert parser.parse(tuple(WORDS), (u for u in UPOS), FEATS) == expected


def test_parse_many_matches_parse(parser):
    sentences = [
        (WORDS, UPOS, FEATS),
        ([], [], []),
        (["Спасибо", "!"], ["INTJ", "PUNCT"], ["_", "_"]),
        (["дом"] * 40, ["NOUN"] * 40, ["Case=Nom"] * 40),
    ]
    many = parser.parse_many(sentences)
    assert many == [parser.parse(*sentence) for sentence in sentences]
    assert parser.parse_many([]) == []
    assert parser.parse_many(iter(sentences[:1])) == many[:1]


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        (("Мама мыла", ["NOUN"], ["_"]), TypeError, "не строка"),
        ((["Мама"], ["NOUN", "VERB"], ["_"]), ValueError, "слов 1, частей речи 2"),
        (([1], ["NOUN"], ["_"]), TypeError, "элемент 0 должен быть строкой, а не int"),
        ((["Мама"], [None], ["_"]), TypeError, "должен быть строкой, а не NoneType"),
        ((5, [], []), TypeError, "ожидается набор строк, а не int"),
    ],
)
def test_parse_errors(parser, args, error, message):
    with pytest.raises(error, match=message):
        parser.parse(*args)


@pytest.mark.parametrize(
    ("sentences", "error", "message"),
    [
        (5, TypeError, "Ожидается набор предложений, а не int"),
        (["abc"], TypeError, "Предложение 0 должно быть тройкой, а не str"),
        ([(WORDS, UPOS)], ValueError, "получено элементов: 2"),
        ([(WORDS, UPOS, FEATS[:2])], ValueError, "признаков 2"),
    ],
)
def test_parse_many_errors(parser, sentences, error, message):
    with pytest.raises(error, match=message):
        parser.parse_many(sentences)


def test_annotate(parser):
    document = Segmenter("ru").segment("Мама мыла раму. Спасибо!")
    Analyzer().annotate(document)
    parser.annotate(document)
    assert document.column("head") == [2, 0, 2, 2, 0, 1]
    assert document.column("deprel") == ["nsubj", "root", "obj", "punct", "root", "punct"]
    # Повторный разбор заменяет столбцы теми же значениями
    parser.annotate(document)
    assert document.column("head") == [2, 0, 2, 2, 0, 1]


def test_annotate_empty_document(parser):
    document = Segmenter("ru").segment("")
    parser.annotate(document)
    assert document.column("head") == []


def test_annotate_needs_morphology(parser):
    document = Segmenter("ru").segment("Мама мыла раму.")
    with pytest.raises(ValueError, match=r"rumorph\.Analyzer\(\)\.annotate"):
        parser.annotate(document)


class Word:
    def __init__(self, index):
        self.index = index


class Sentence:
    def __init__(self, indices):
        self.words = [Word(i) for i in indices]


class FakeDocument:
    """Документ по протоколу anySeg: столбцы и предложения задаются явно"""

    def __init__(self, sentences, upos=None):
        self.forms = ["Мама", "мыла", "раму", "."]
        self.upos = upos or UPOS
        self.sentences = sentences
        self.columns = {}

    def column(self, name):
        return {"form": self.forms, "upos": self.upos, "feats": FEATS}[name]

    def set_columns(self, **columns):
        self.columns = columns


def test_annotate_protocol(parser):
    document = FakeDocument([Sentence([0, 1]), Sentence([2, 3])])
    parser.annotate(document)
    assert set(document.columns) == {"head", "deprel"}
    assert len(document.columns["head"]) == 4


@pytest.mark.parametrize(
    "sentences",
    [
        [Sentence([0, 1]), Sentence([3])],
        [Sentence([0, 1])],
        [Sentence([1, 2, 3])],
        [Sentence([0, 1, 2, 3]), Sentence([4])],
    ],
)
def test_annotate_unordered(parser, sentences):
    with pytest.raises(ValueError, match="не идут подряд"):
        parser.annotate(FakeDocument(sentences))


def test_annotate_bad_types(parser):
    document = FakeDocument([Sentence([0, 1, 2, 3])])
    document.sentences[0].words[0].index = "0"
    with pytest.raises(TypeError, match="Номер слова должен быть целым числом, а не str"):
        parser.annotate(document)
    document = FakeDocument([Sentence([0, 1, 2, 3])], upos=["NOUN", 5, "NOUN", "PUNCT"])
    with pytest.raises(TypeError, match="upos: элемент 1"):
        parser.annotate(document)


def test_surrogates(parser):
    # Непарный суррогат пропускают anySeg и ruMorph - значит, и разбор
    document = parser.parse_text("Мама \ud800 мыла раму.")
    assert len(document.column("head")) == len(document.column("form"))
    assert is_tree(parser.parse(["Мама", "\ud800"], ["NOUN", "X"], ["_", "_"]))


def test_annotate_generator_columns(parser):
    document = FakeDocument([Sentence([0, 1, 2, 3])])
    document.column = lambda name: iter({"form": WORDS, "upos": UPOS, "feats": FEATS}[name])
    parser.annotate(document)
    assert len(document.columns["head"]) == 4


def test_annotate_huge_index(parser):
    document = FakeDocument([Sentence([0, 1, 2, 3])])
    document.sentences[0].words[0].index = 2**64
    with pytest.raises(ValueError, match="не идут подряд"):
        parser.annotate(document)


def test_parse_text(parser):
    document = parser.parse_text("Мама мыла раму.")
    assert isinstance(document, Document)
    assert [(w.form, w.upos, w.head, w.deprel) for w in document.words] == [
        ("Мама", "NOUN", 2, "nsubj"),
        ("мыла", "VERB", 0, "root"),
        ("раму", "NOUN", 2, "obj"),
        (".", "PUNCT", 2, "punct"),
    ]
    with pytest.raises(TypeError, match="Текст должен быть строкой, а не bytes"):
        parser.parse_text(b"text")


def test_model_from_path(tmp_path, parser):
    custom = Parser(MODEL)
    assert custom.path == MODEL
    assert repr(custom) == f"Parser({str(MODEL)!r})"
    assert custom.parse(WORDS, UPOS, FEATS) == parser.parse(WORDS, UPOS, FEATS)
    assert repr(parser) == "Parser()"
    assert parser.path is None


def test_model_errors(tmp_path):
    missing = tmp_path / "нет.bin"
    with pytest.raises(FileNotFoundError, match="Нет файла модели") as error:
        Parser(missing)
    assert error.value.errno == 2
    assert error.value.filename == str(missing)
    damaged = tmp_path / "damaged.bin"
    data = bytearray(MODEL.read_bytes())
    data[-1] ^= 1
    damaged.write_bytes(bytes(data))
    with pytest.raises(ValueError, match=f"{damaged}: .*контрольная сумма не совпадает"):
        Parser(damaged)
    other = tmp_path / "other.bin"
    other.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="нет сигнатуры"):
        Parser(other)


def test_sources(parser):
    sources = parser.sources
    assert sources["ud"] == "r2.19"
    assert set(sources["treebanks"]) == {"UD_Russian-Taiga", "UD_Russian-GSD", "UD_Russian-Poetry"}
    assert sources["tags"].startswith("rumorph ")


def test_pickle_and_copy(tmp_path, parser):
    assert copy.copy(parser) is parser
    assert copy.deepcopy(parser) is parser
    restored = pickle.loads(pickle.dumps(parser))
    assert restored.path is None
    assert restored.parse(WORDS, UPOS, FEATS) == parser.parse(WORDS, UPOS, FEATS)
    model = tmp_path / "model.bin"
    shutil.copyfile(MODEL, model)
    custom = pickle.loads(pickle.dumps(Parser(model)))
    assert custom.path == model


def test_threads(parser):
    expected = parser.parse(WORDS, UPOS, FEATS)
    results = []

    def work():
        results.extend(parser.parse(WORDS, UPOS, FEATS) == expected for _ in range(50))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 400 and all(results)
