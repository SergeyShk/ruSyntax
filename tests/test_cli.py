import errno
import io
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

import anyseg
import pytest
import rumorph

import rusyntax
from rusyntax import Parser, __main__ as command

ROOT = Path(__file__).parents[1]
TEXT = "Мама мыла раму. Папа спал.\n\nКот ест.\n"
WORD = "1\tМы\t_\t_\t_\t_\t_\t_\t_\t_"
FEATS = [
    "Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing",
    "Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act",
    "Animacy=Inan|Case=Acc|Gender=Fem|Number=Sing",
]
TAGGED = (
    "# sent_id = a\n# text = Мама мыла раму\n"
    f"1\tМама\t_\tNOUN\t_\t{FEATS[0]}\t_\t_\t_\t_\n"
    f"2\tмыла\t_\tVERB\t_\t{FEATS[1]}\t_\t_\t_\t_\n"
    f"3\tраму\t_\tNOUN\t_\t{FEATS[2]}\t_\t_\t_\t_\n\n"
)


@pytest.fixture(scope="module")
def parser():
    return Parser()


def call(*args, data=b""):
    stdout, stderr = io.BytesIO(), io.StringIO()
    code = command.run(list(args), io.BytesIO(data), stdout, stderr)
    return code, stdout.getvalue().decode(), stderr.getvalue()


def run(*args, data=b""):
    return subprocess.run(
        [sys.executable, "-m", "rusyntax", *args], input=data, capture_output=True, check=False
    )


def sentence_ids(conllu):
    return [int(number) for number in re.findall(r"^# sent_id = (\d+)$", conllu, re.MULTILINE)]


class Chunks:
    """Вход по size байт за чтение: куски кончаются посреди строк и символов"""

    def __init__(self, data, size=1):
        self.data, self.size = data, size

    def read(self, size):
        chunk, self.data = self.data[: self.size], self.data[self.size :]
        return chunk


class NonBlocking:
    """Неблокирующий дескриптор без данных: read возвращает None"""

    def read(self, size):
        return None


def test_text(parser):
    code, out, err = call(data=TEXT.encode())
    assert (code, err) == (0, "")
    assert out == parser.parse_text(TEXT).to_conllu()
    assert out.splitlines()[4].split("\t")[6:8] == ["2", "nsubj"]


def test_text_like_rumorph(tmp_path):
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text(TEXT, encoding="utf-8")
    second.write_bytes("﻿Кот спит.\r\nИ все.\r\n".encode())
    code, out, err = call(str(first), str(second))
    assert (code, err) == (0, "")
    assert out.count("# newdoc\n") == 2
    assert sentence_ids(out) == [1, 2, 3, 4, 5]
    morphology = subprocess.run(
        [sys.executable, "-m", "rumorph", "-f", "conllu", str(first), str(second)],
        capture_output=True,
        encoding="utf-8",
        check=True,
    ).stdout

    def without_syntax(conllu):
        return [re.sub(r"^((?:[^\t]*\t){6})[^\t]*\t[^\t]*", r"\1_\t_", line) for line in conllu]

    assert without_syntax(out.splitlines()) == without_syntax(morphology.splitlines())


def test_text_in_pieces(parser):
    # Длинный абзац без пустых строк сегментатор режет на куски: номера и абзац идут дальше
    text = "Длиннющееслово превосходнейшее. " * 9000 + "\n\nКот спит.\n"
    code, out, err = call(data=text.encode())
    assert (code, err) == (0, "")
    assert sentence_ids(out) == list(range(1, 9002))
    assert (out.count("# newdoc\n"), out.count("# newpar\n")) == (1, 2)
    # Куски по байту: строки и символы рвутся на границах чтения
    stdout = io.BytesIO()
    assert command.run([], Chunks(TEXT.encode()), stdout, io.StringIO()) == 0
    assert stdout.getvalue().decode() == parser.parse_text(TEXT).to_conllu()


def test_long_line(parser):
    # Строка длиннее куска чтения отдается частями по границам символов
    text = "Ёж и кот спят. " * 6000 + "\n"
    code, out, _ = call(data=text.encode())
    assert code == 0
    assert out == parser.parse_text(text).to_conllu()


def test_non_blocking_input():
    # «Данных пока нет» - ошибка чтения, а не конец входа
    stdout, stderr = io.BytesIO(), io.StringIO()
    assert command.run([], NonBlocking(), stdout, stderr) == 1
    assert stderr.getvalue() == f"rusyntax: -: {os.strerror(errno.EAGAIN)}\n"
    assert stdout.getvalue() == b""


def test_empty_input():
    assert call() == (0, "", "")
    assert call(data=b"\n \n\n") == (0, "", "")
    stdout, stderr = io.BytesIO(), io.StringIO()
    assert command.run([], None, stdout, stderr) == 0
    assert stdout.getvalue() == b""


def test_conllu_tagged(parser):
    code, out, err = call("-i", "conllu", data=TAGGED.encode())
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[:2] == ["# sent_id = a", "# text = Мама мыла раму"]
    expected = parser.parse(["Мама", "мыла", "раму"], ["NOUN", "VERB", "NOUN"], FEATS)
    assert expected == [(2, "nsubj"), (0, "root"), (2, "obj")]
    assert [tuple(line.split("\t")[6:8]) for line in lines[2:5]] == [
        (str(head), deprel) for head, deprel in expected
    ]
    # Леммы входа не заполняются: морфологию не трогают
    assert [line.split("\t")[2] for line in lines[2:5]] == ["_", "_", "_"]


def test_conllu_untagged(parser):
    # Предложение без частей речи размечает ruMorph, размеченное остается как есть
    untagged = "# newdoc id = x\n" + WORD + "\n2\tстали\t_\t_\t_\t_\t_\t_\t_\t_\n\n"
    code, out, err = call("--input=conllu", data=(untagged + TAGGED).encode())
    assert (code, err) == (0, "")
    first, second = anyseg.read_conllu(untagged)[0], anyseg.read_conllu(TAGGED)[0]
    rumorph.Analyzer().annotate(first)
    parser.annotate(first)
    parser.annotate(second)
    assert out == first.to_conllu() + second.to_conllu()
    assert out.splitlines()[1].split("\t")[2:4] == ["мы", "PRON"]


def test_conllu_errors(tmp_path):
    path = tmp_path / "broken.conllu"
    path.write_text(TAGGED + WORD + "\n2\tстали\n\n" + TAGGED, encoding="utf-8")
    code, out, err = call("-iconllu", str(path), str(path))
    assert code == 1
    # Предложения до ошибки разобраны, остальные файлы тоже
    assert out.count("# sent_id = a\n") == 2
    message = f"rusyntax: {path}: строка 8: нужно 10 столбцов через табуляцию, а не 2\n"
    assert err == message * 2
    mixed = TAGGED + WORD.replace("\t_\t_\t_\t_\t_", "\t_\tPRON\t_\t_\t_") + "\n" + WORD + "\n\n"
    code, out, err = call("-i", "conllu", data=mixed.encode())
    assert (code, out.count("# sent_id = a\n")) == (1, 1)
    assert err == (
        "rusyntax: -: строка 8: у слова нет части речи, а у других слов предложения она есть\n"
    )
    code, _, err = call("-i", "conllu", data=TAGGED.replace("VERB", "verb").encode())
    assert code == 1
    assert err.startswith("rusyntax: -: строка 4: неизвестная часть речи 'verb': ожидается")
    # Из двух ошибок предложения сообщается первая по строкам
    broken = "1\tМама\t_\tNOUN\t_\t_\tz\t_\t_\t_\n2\tспит\t_\tverb\t_\t_\t_\t_\t_\t_\n\n"
    assert (
        call("-i", "conllu", data=broken.encode())[2]
        == 'rusyntax: -: строка 1: неверная вершина "z"\n'
    )
    assert call("-i", "conllu", data=broken.replace("verb", "VERB").encode())[2] == (
        'rusyntax: -: строка 1: неверная вершина "z"\n'
    )


@pytest.mark.parametrize(
    ("conllu", "message"),
    [
        (WORD.replace("1\t", "x\t", 1) + "\n", 'строка 1: ожидается слово 1, а не "x"'),
        (WORD.replace("1\t", "1.x\t", 1) + "\n", 'строка 1: неверный номер слова "1.x"'),
        ("1\tМы\t_\t_\t_\t_\tz\t_\t_\t_\n", 'строка 1: неверная вершина "z"'),
        (
            "1-2" + WORD[1:] + "\n" + WORD + "\n",
            "строка 3: у многословного токена нет всех его слов",
        ),
        ("2-3" + WORD[1:] + "\n", "строка 1: многословный токен 2-3 не начинается со слова 1"),
        ("# newdoc\n", "строка 2: в предложении нет слов"),
        (WORD + "\n# c\n", "строка 2: комментарий после строк токенов"),
        (
            "# text = Кот\n" + WORD + "\n",
            'строка 2: форма "Мы" не совпадает с текстом предложения',
        ),
        (
            "# text = Мы и\n" + WORD + "\n",
            "строка 1: в тексте предложения есть символы, которых нет ни в одном токене",
        ),
    ],
)
def test_conllu_messages(conllu, message):
    # Сообщения anySeg переводятся: тест падает, если anySeg их изменит
    code, _, err = call("-i", "conllu", data=conllu.encode())
    assert (code, err) == (1, f"rusyntax: -: {message}\n")


def test_encoding(tmp_path):
    damaged = "Мама мыла раму.\n\n".encode() + b"\xff \xd0\xba\xd0\xbe\xd1\x82\n"
    code, out, err = call(data=damaged)
    assert code == 1
    assert "�" in out and "кот" in out
    assert err == "rusyntax: -: строка 3: не UTF-8, неверные байты заменены на U+FFFD\n"
    # Файл в cp1251 с ASCII в начале (субтитры) не разбирается и частично
    subtitles = tmp_path / "cp1251.srt"
    subtitles.write_bytes("1\n00:00:01,000 --> 00:00:02,000\nМама мыла раму\n".encode("cp1251"))
    assert call(str(subtitles)) == (
        1,
        "",
        f"rusyntax: {subtitles}: файл не в UTF-8 (строка 3)\n",
    )
    assert (
        call(data="Кот спит\n".encode("utf-16"))[2] == "rusyntax: -: файл не в UTF-8 (строка 1)\n"
    )
    assert call(data=b"A\r\nB\x00\n")[2] == "rusyntax: -: файл не в UTF-8 (строка 2)\n"
    # После строки UTF-8 NUL - часть текста, а неверные байты в CoNLL-U - тоже замена
    assert call(data="Кот\nB\x00\n".encode())[0] == 0
    conllu = (
        (TAGGED + "1\tМ\xff\t_\tNOUN\t_\t_\t_\t_\t_\t_\n\n").encode().replace(b"\xc3\xbf", b"\xff")
    )
    code, out, err = call("-i", "conllu", data=conllu)
    assert (code, err) == (
        1,
        "rusyntax: -: строка 7: не UTF-8, неверные байты заменены на U+FFFD\n",
    )
    assert "М�" in out


def test_files(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text(TEXT, encoding="utf-8")
    code, out, err = call(str(tmp_path / "нет.txt"), str(tmp_path), str(path))
    assert code == 1
    assert sentence_ids(out) == [1, 2, 3]
    assert err == (
        f"rusyntax: {tmp_path / 'нет.txt'}: файл не найден\nrusyntax: {tmp_path}: это каталог\n"
    )
    # После «--» имя с дефиса - файл
    dashed = tmp_path / "-text.txt"
    dashed.write_text(TEXT, encoding="utf-8")
    code, out, _ = call("--", str(dashed), data=b"")
    assert code == 0 and sentence_ids(out) == [1, 2, 3]
    assert call("-", data=TEXT.encode())[1] == out


def test_output_file(tmp_path):
    path, output = tmp_path / "text.txt", tmp_path / "text.conllu"
    path.write_text(TEXT, encoding="utf-8")
    assert call("-o", str(output), str(path)) == (0, "", "")
    assert output.read_text(encoding="utf-8") == call(str(path))[1]
    assert call("--output", "-", str(path))[1] == output.read_text(encoding="utf-8")
    # Файл вывода, который затер бы вход или модель, не создается
    message = f"rusyntax: {path}: файл вывода совпадает с входным файлом или файлом модели\n"
    assert call("-o", str(path), str(path)) == (1, "", message)
    model = tmp_path / "model.bin"
    shutil.copy(ROOT / "data" / "ru-syntax.bin", model)
    assert call("-m", str(model), "-o", str(model), str(path))[0] == 1
    assert path.read_text(encoding="utf-8") == TEXT
    assert call("-o", str(tmp_path), str(path)) == (1, "", f"rusyntax: {tmp_path}: это каталог\n")
    # Стандартный ввод из файла вывода
    if sys.platform != "win32":
        with path.open("rb") as stdin:
            result = subprocess.run(
                [sys.executable, "-m", "rusyntax", "-o", str(path)],
                stdin=stdin,
                capture_output=True,
                check=False,
            )
        assert result.returncode == 1
        assert path.read_text(encoding="utf-8") == TEXT
        # Вход /dev/stdin: на macOS stat этого пути - файл devfs, а не перенаправленный файл
        for name in ("/dev/stdin", "/dev/fd/0"):
            with path.open("rb") as stdin:
                result = subprocess.run(
                    [sys.executable, "-m", "rusyntax", "-o", str(path), name],
                    stdin=stdin,
                    capture_output=True,
                    check=False,
                )
            assert result.returncode == 1, name
            assert path.read_text(encoding="utf-8") == TEXT


def test_model_option(tmp_path, monkeypatch):
    model = tmp_path / "model.bin"
    shutil.copy(ROOT / "data" / "ru-syntax.bin", model)
    assert call("--model", str(model), data=TEXT.encode())[1] == call(data=TEXT.encode())[1]
    model.write_bytes(b"not a model")
    code, out, err = call("-m", str(model), data=TEXT.encode())
    assert (code, out) == (1, "")
    assert err.startswith(f"rusyntax: {model}: ")
    # Модель, которую не открыть, - как входные файлы: путь как задан и причина по-русски
    missing = tmp_path / "нет.bin"
    assert call("-m", str(missing)) == (1, "", f"rusyntax: {missing}: файл не найден\n")
    assert call("-m", str(tmp_path)) == (1, "", f"rusyntax: {tmp_path}: это каталог\n")
    # Относительный путь модели неверного формата - тоже как задан
    monkeypatch.chdir(tmp_path)
    code, _, err = call("-m", "model.bin")
    assert code == 1
    assert err.startswith("rusyntax: model.bin: неверный формат модели: ")


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--format", "xml"], "неизвестный параметр --format"),
        (["-i", "tokens"], "неизвестное значение параметра -i: tokens (допустимые: text, conllu)"),
        (["--model"], "параметру --model нужно значение"),
        (["-hx"], "неизвестный параметр -hx"),
        (["--version=1"], "неизвестный параметр --version=1"),
    ],
)
def test_usage_errors(args, message):
    assert call(*args) == (2, "", f"rusyntax: {message}\nСправка: rusyntax --help\n")


def test_help_and_version():
    code, out, _ = call("--help")
    assert code == 0 and out.startswith("rusyntax - синтаксический анализ русского текста")
    assert call("-h")[1] == out
    versions = f"anyseg {anyseg.__version__}, rumorph {rumorph.__version__}"
    assert call("-V") == (0, f"rusyntax {rusyntax.__version__} ({versions})\n", "")


class Closed(io.BytesIO):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def write(self, data):
        raise self.error


def test_write_errors():
    stderr = io.StringIO()
    assert command.run([], io.BytesIO(TEXT.encode()), Closed(BrokenPipeError()), stderr) == 0
    assert command.run(["-V"], None, Closed(BrokenPipeError()), stderr) == 0
    assert stderr.getvalue() == ""
    assert command.run([], io.BytesIO(TEXT.encode()), Closed(OSError(28, "No space")), stderr) == 1
    assert stderr.getvalue() == "rusyntax: ошибка записи: [Errno 28] No space\n"


def test_closed_pipe(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text(TEXT * 3000, encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "rusyntax", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None and process.stderr is not None
    assert process.stdout.readline().decode() == "# newdoc\n"
    process.stdout.close()
    assert process.wait(timeout=60) == 0
    assert process.stderr.read() == b""


@pytest.mark.parametrize(
    ("args", "paragraph"), [([], "Мы стали друзьями\n\n"), (["-i", "conllu"], WORD + "\n\n")]
)
def test_paragraph_by_paragraph(args, paragraph):
    # Команда отвечает на каждый абзац, не дожидаясь конца ввода: ее можно держать открытой
    process = subprocess.Popen(
        [sys.executable, "-m", "rusyntax", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )
    assert process.stdin is not None and process.stdout is not None
    stdout = process.stdout
    try:
        for _ in range(2):
            process.stdin.write(paragraph)
            process.stdin.flush()
            lines: list[str] = []

            def read(out: list[str] = lines) -> None:
                while not out or out[-1] not in {"\n", ""}:
                    out.append(stdout.readline())

            reader = threading.Thread(target=read)
            reader.start()
            reader.join(timeout=60)
            assert lines[-1] == "\n" and lines[-2].split("\t")[1] in {"друзьями", "Мы"}
    finally:
        process.stdin.close()
        process.wait(timeout=60)


def test_console_script():
    script = shutil.which("rusyntax", path=str(Path(sys.executable).parent))
    assert script is not None
    result = subprocess.run([script, "-V"], capture_output=True, encoding="utf-8", check=False)
    assert result.stdout.startswith(f"rusyntax {rusyntax.__version__} (")


def test_main_in_process(tmp_path, monkeypatch, capfd):
    path = tmp_path / "text.txt"
    path.write_text(TEXT, encoding="utf-8")
    handlers = []
    monkeypatch.setattr(signal, "signal", lambda *args: handlers.append(args))
    monkeypatch.setattr(sys, "argv", ["rusyntax", str(path)])
    with pytest.raises(SystemExit) as exit_info:
        command.main()
    assert exit_info.value.code == 0
    assert handlers[0] == (signal.SIGINT, signal.SIG_DFL)
    assert capfd.readouterr().out.startswith("# newdoc\n# newpar\n# sent_id = 1\n")
    # Игнорируемый SIGINT остается игнорируемым
    handlers.clear()
    monkeypatch.setattr(signal, "getsignal", lambda _: signal.SIG_IGN)
    with pytest.raises(SystemExit):
        command.main()
    assert handlers == []


def test_subprocess_smoke():
    result = run("-i", "conllu", data=TAGGED.encode())
    assert (result.returncode, result.stderr) == (0, b"")
    assert result.stdout.decode().splitlines()[3].split("\t")[6:8] == ["0", "root"]
