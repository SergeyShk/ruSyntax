import copy
import pickle
import re
import shutil
import threading
import weakref
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


def test_parse_empty_and_unknown_feats(parser):
    assert parser.parse([], [], []) == []
    relations = parser.parse(["Ёжик", "", "zzz"], ["NOUN", "X", "X"], ["Bar=Baz", "", "_"])
    assert is_tree(relations)


def test_parse_feats_mapping(parser):
    expected = parser.parse(WORDS, UPOS, FEATS)
    feats = [
        {"Number": "Sing", "Gender": "Fem", "Case": "Nom"},
        {"Gender": "Fem", "Tense": "Past"},
        {"Case": "Acc"},
        {},
    ]
    assert parser.parse(WORDS, UPOS, feats) == expected
    with pytest.raises(TypeError, match="у слова 0 признаки и их значения должны быть строками"):
        parser.parse(["Мама"], ["NOUN"], [{"Case": 1}])
    with pytest.raises(
        TypeError, match="элемент 0 должен быть строкой, словарем или None, а не int"
    ):
        parser.parse(["Мама"], ["NOUN"], [5])


def test_parse_rumorph_analyses(parser):
    # Готовые слова: теги ruMorph с контекстом, признаки - словарем
    words = ["Кошка", "спала", "на", "крыльце", "."]
    best = [analyses[0] for analyses in Analyzer().analyze_sentence(words)]
    relations = parser.parse(words, [a.pos for a in best], [a.feats for a in best])
    document = parser.parse_text("Кошка спала на крыльце.")
    assert relations == list(zip(document.column("head"), document.column("deprel"), strict=True))


def test_root_relation_only_at_root(parser):
    for text in ["!!! ??? ... --- *** +++", "Мама мыла раму."]:
        document = parser.parse_text(text)
        heads, deprels = document.column("head"), document.column("deprel")
        assert all(
            (head == 0) == (deprel == "root") for head, deprel in zip(heads, deprels, strict=True)
        )


def test_long_sentence(parser):
    # Предложение длиннее 1024 слов: вершина ищется в окне, время растет линейно
    words = (["мама", "мыла", "раму", "а", "папа", "читал", "газету"] * 600)[:4000]
    upos = ["NOUN", "VERB", "NOUN", "CCONJ", "NOUN", "VERB", "NOUN"] * 600
    relations = parser.parse(words, upos[:4000], [None] * 4000)
    assert len(relations) == 4000
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
        ((["Мама"], ["noun"], ["_"]), ValueError, "неизвестная часть речи 'noun' у слова 0"),
        (
            (["Мама", "-"], ["NOUN", "PNCT"], ["_", "_"]),
            ValueError,
            "'PNCT' у слова 1: ожидается одна из 17",
        ),
        ((["Мама"], [""], ["_"]), ValueError, "неизвестная часть речи '' у слова 0"),
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
        ([(WORDS, UPOS, FEATS[:2])], ValueError, "У каждого слова предложения 0 .*признаков 2"),
        (
            [(WORDS, UPOS, FEATS), (["Мама", "мыла"], ["NOUN", "verb"], ["_", "_"])],
            ValueError,
            "'verb' у слова 1 предложения 1:",
        ),
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


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("Мама мыла раму.", "а не строка: текст разбирает parse_text"),
        (5, "Ожидается документ anySeg или объект с column, sentences и set_columns, а не int"),
    ],
)
def test_annotate_not_document(parser, document, message):
    with pytest.raises(TypeError, match=message):
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
    document = FakeDocument(5)
    with pytest.raises(TypeError, match="Предложения документа должны быть набором, а не int"):
        parser.annotate(document)


def test_annotate_bad_columns(parser):
    document = FakeDocument([Sentence([0, 1, 2, 3])], upos=["NOUN", "verb", "NOUN", "PUNCT"])
    with pytest.raises(ValueError, match="неизвестная часть речи 'verb' у слова 1"):
        parser.annotate(document)
    document = FakeDocument([Sentence([0, 1, 2, 3])], upos=["NOUN", "_", "NOUN", "PUNCT"])
    with pytest.raises(ValueError, match="нет частей речи"):
        parser.annotate(document)
    document = FakeDocument([Sentence([0, 1, 2])], upos=["NOUN", "VERB", "NOUN"])
    with pytest.raises(ValueError, match="разной длины: form 4, upos 3, feats 4"):
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


class Index:
    """Целое по протоколу __index__, как numpy.int64"""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def test_annotate_index_kinds(parser):
    document = FakeDocument([Sentence([0, 1, 2, 3])])
    document.sentences[0].words[0].index = Index(0)
    parser.annotate(document)
    assert len(document.columns["head"]) == 4
    for huge in (2**64, -(2**64), -1):
        document.sentences[0].words[0].index = huge
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


def test_relative_model_path(tmp_path, monkeypatch):
    shutil.copyfile(MODEL, tmp_path / "model.bin")
    monkeypatch.chdir(tmp_path)
    custom = Parser("model.bin")
    assert custom.path == Path.cwd() / "model.bin"
    data = pickle.dumps(custom)
    monkeypatch.chdir(tmp_path.parent)
    assert pickle.loads(data).path == custom.path


def test_changed_model_file_is_reloaded(tmp_path):
    model = tmp_path / "model.bin"
    shutil.copyfile(MODEL, model)
    first = Parser(model)
    model.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="нет сигнатуры"):
        Parser(model)
    assert first.parse(WORDS, UPOS, FEATS)


@pytest.mark.parametrize(
    ("path", "error", "message"),
    [
        (5, TypeError, "Путь к модели должен быть строкой или os.PathLike, а не int"),
        (b"model.bin", TypeError, "а не bytes"),
        ("model\0.bin", ValueError, "нулевой символ"),
    ],
)
def test_model_path_errors(path, error, message):
    with pytest.raises(error, match=message):
        Parser(path)


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
    with pytest.raises(ValueError, match=f"{re.escape(str(damaged))}: .*контрольная сумма"):
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


def test_equality_and_weakref(tmp_path, parser):
    assert Parser() == parser
    assert hash(Parser()) == hash(parser)
    model = tmp_path / "model.bin"
    shutil.copyfile(MODEL, model)
    assert Parser(model) == Parser(model)
    assert hash(Parser(model)) == hash(Parser(model))
    assert Parser(model) != parser
    assert parser != "Parser()"
    assert pickle.loads(pickle.dumps(parser)) == parser
    assert weakref.ref(parser)() is parser


def test_rumorph_tags_match_training():
    # Модель обучена на тегах ruMorph версии из sources: теги установленного ruMorph должны совпадать
    sentences, words = [], []
    for line in [*(ROOT / "data" / "check.conllu").read_text(encoding="utf-8").splitlines(), ""]:
        if line:
            words.append(line.split("\t"))
        elif words:
            sentences.append(words)
            words = []
    analyzer = Analyzer()
    total = upos = feats = 0
    for sentence in sentences:
        best = [
            analyses[0] for analyses in analyzer.analyze_sentence([word[1] for word in sentence])
        ]
        for word, analysis in zip(sentence, best, strict=True):
            total += 1
            upos += word[3] != analysis.pos
            pairs = sorted(analysis.feats.items(), key=lambda pair: pair[0].lower())
            feats += word[5] != ("|".join(f"{name}={value}" for name, value in pairs) or "_")
    assert total > 4000
    assert upos <= total * 0.005, f"части речи расходятся у {upos} из {total} слов"
    assert feats <= total * 0.005, f"признаки расходятся у {feats} из {total} слов"


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
