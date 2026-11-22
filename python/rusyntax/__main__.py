# Синтаксический анализатор русского языка (ruSyntax)
#
# Copyright (C) 2026
# Автор: Шкарин Сергей <kouki.sergey@gmail.com>
# URL: <https://github.com/SergeyShk/ruSyntax>

"""
Команда rusyntax: синтаксический анализ текста из файлов и стандартного ввода

Пример использования:
    rusyntax text.txt
    rusyntax --input conllu tagged.conllu
    python -m rusyntax --help
"""

import codecs
import contextlib
import errno
import io
import os
import re
import signal
import stat
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import BinaryIO, NoReturn, Protocol, TextIO

import anyseg
import rumorph

from ._rusyntax import Parser, __version__

HELP = """\
rusyntax - синтаксический анализ русского текста: вершина и тип связи Universal Dependencies
для каждого слова

Использование: rusyntax [параметры] [файл ...]

Файлы читаются по очереди, без файлов и вместо «-» - стандартный ввод; вход - в UTF-8.
Результат - CoNLL-U - пишется в стандартный вывод или в файл параметра -o.

Параметры:
  -i, --input ВХОД   вход:
                       text - текст: пустая строка заканчивает абзац, токены и предложения
                       находит сегментатор anySeg, части речи и признаки - ruMorph
                       (по умолчанию)
                       conllu - CoNLL-U с частями речи и признаками; предложение без частей
                       речи сначала разбирает ruMorph
  -m, --model ФАЙЛ   файл модели ruSyntax (по умолчанию - модель пакета)
  -o, --output ФАЙЛ  записать результат в файл
  -h, --help         показать эту справку и выйти
  -V, --version      показать версии ruSyntax, anySeg и ruMorph и выйти
"""

# Вход читается кусками: столько байт за чтение, и не длиннее незаконченная строка в памяти
CHUNK = 1 << 16
UPOS = frozenset((
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
))  # fmt: skip
SENT_ID = re.compile(r"^# sent_id = (\d+)$", re.MULTILINE)
# Концы строк, как у команд anyseg и rumorph: LF, CR и CRLF
LINE_END = re.compile(r"\r\n|\r|\n")
CONLLU_ERROR = re.compile(r"line (\d+): (.*)", re.DOTALL)
# Ошибки чтения CoNLL-U anySeg по-русски; незнакомая остается как есть
CONLLU_MESSAGES = [
    (
        re.compile(r"expected 10 tab-separated columns, got (\d+)"),
        r"нужно 10 столбцов через табуляцию, а не \1",
    ),
    (re.compile(r"cannot read the ID (.*)"), r"неверный номер слова \1"),
    (re.compile(r"expected word (\d+), got (.*)"), r"ожидается слово \1, а не \2"),
    (re.compile(r"cannot read the head (.*)"), r"неверная вершина \1"),
    (
        re.compile(r"multiword token (\S+) does not start at word (\d+)"),
        r"многословный токен \1 не начинается со слова \2",
    ),
    (
        re.compile(r"a multiword token without all its words"),
        "у многословного токена нет всех его слов",
    ),
    (re.compile(r"a sentence without words"), "в предложении нет слов"),
    (re.compile(r"a comment after token lines"), "комментарий после строк токенов"),
    (
        re.compile(r"the form (.*) does not match the text of the sentence"),
        r"форма \1 не совпадает с текстом предложения",
    ),
    (
        re.compile(r"the text of the sentence holds characters that no token has"),
        "в тексте предложения есть символы, которых нет ни в одном токене",
    ),
    (re.compile(r"the values of a document exceed 2 GiB"), "значения документа больше 2 ГиБ"),
]


class _Raw(Protocol):
    def read(self, size: int, /) -> bytes | None: ...


class UsageError(Exception):
    """Неверные параметры команды"""


class EncodingError(Exception):
    """Файл не в UTF-8: номер строки, по которой это видно"""

    def __init__(self, line: int) -> None:
        super().__init__(line)
        self.line = line


class InputError(Exception):
    """Неверная строка CoNLL-U: номер строки и сообщение"""

    def __init__(self, line: int, message: str) -> None:
        super().__init__(line, message)
        self.line = line
        self.message = message


class WriteError(Exception):
    """Ошибка записи результата"""

    def __init__(self, error: OSError) -> None:
        super().__init__(error)
        self.error = error


@dataclass
class Options:
    """Параметры запуска команды"""

    conllu: bool
    model: str | None
    output: str | None
    files: list[str]


def parse_args(args: Sequence[str]) -> Options | str:
    """
    Разбор аргументов командной строки

    Аргументы:
        args (Sequence[str]): Аргументы без имени программы

    Вывод:
        Options | str: Параметры запуска или "help" и "version" для справки и версии

    Исключения:
        UsageError: Если параметр неизвестен, у него нет значения или значение неверно
    """
    conllu, model, output, files = False, None, None, []
    options = True
    rest = iter(args)
    for arg in rest:
        if not (options and arg.startswith("-") and arg != "-"):
            files.append(arg)
            continue
        if arg.startswith("--"):
            name, equals, attached = arg.partition("=")
            inline = attached if equals else None
        elif len(arg) > 2:
            name, inline = arg[:2], arg[2:]
        else:
            name, inline = arg, None
        if name in {"--", "-h", "--help", "-V", "--version"}:
            if inline is not None:
                raise UsageError(f"неизвестный параметр {arg}")
            if name == "--":
                options = False
                continue
            return "help" if name in {"-h", "--help"} else "version"
        if name not in {"-i", "--input", "-m", "--model", "-o", "--output"}:
            raise UsageError(f"неизвестный параметр {arg}")
        value = inline if inline is not None else next(rest, None)
        if value is None:
            raise UsageError(f"параметру {name} нужно значение")
        if name in {"-i", "--input"}:
            if value not in {"text", "conllu"}:
                raise UsageError(
                    f"неизвестное значение параметра {name}: {value} (допустимые: text, conllu)"
                )
            conllu = value == "conllu"
        elif name in {"-m", "--model"}:
            model = value
        else:
            output = None if value == "-" else value
    return Options(conllu, model, output, files)


class Input:
    """
    Вход команды в UTF-8, как у команд anyseg и rumorph

    Описание:
        Байты читаются кусками и отдаются целыми строками, а строка длиннее куска - частями
        по границам символов. Неверные байты UTF-8 и NUL до первой строки UTF-8 с не-ASCII
        символами значат другую кодировку файла (cp1251, UTF-16), после - испорченную строку:
        ее неверные байты заменяются на U+FFFD. Перед каждым чтением, которое может ждать
        ввода, вызывается waiting: результат уже разобранного успевает уйти к читателю

    Аргументы:
        raw (_Raw): Файл, открытый на чтение в двоичном режиме без буфера
        waiting (Callable[[], None]): Что сделать перед чтением
    """

    def __init__(self, raw: _Raw, waiting: Callable[[], None]) -> None:
        self._raw = raw
        self._waiting = waiting
        self._rest = b""
        self._ready = b""
        self._line = 1
        self._utf8 = False
        self._start = True
        self._end = False
        # Первая строка, в которой неверные байты заменены на U+FFFD
        self.invalid: int | None = None

    def read(self, size: int = CHUNK, /) -> bytes:
        """
        Следующие байты входа

        Аргументы:
            size (int): Сколько байт отдать самое большее

        Вывод:
            bytes: Байты UTF-8, пустые - в конце входа

        Исключения:
            EncodingError: Если вход не в UTF-8
            OSError: Если файл не читается
        """
        while not self._ready and not self._end:
            self._fill()
        ready, self._ready = self._ready[:size], self._ready[size:]
        return ready

    def _fill(self) -> None:
        self._waiting()
        data = self._raw.read(CHUNK)
        # Неблокирующий дескриптор без данных - ошибка чтения, как у команд anyseg и rumorph
        if data is None:
            raise BlockingIOError(errno.EAGAIN, os.strerror(errno.EAGAIN))
        if self._start and data.startswith((b"\xff\xfe", b"\xfe\xff")):
            raise EncodingError(1)
        self._start = self._start and not data
        if not data:
            self._end = True
            chunk, self._rest = self._rest, b""
        else:
            data = self._rest + data
            # CR в конце куска может оказаться началом CRLF
            cut = max(data.rfind(b"\n"), data.rfind(b"\r", 0, len(data) - 1)) + 1
            if not cut:
                if len(data) < CHUNK:
                    self._rest = data
                    return
                cut = _boundary(data)
            chunk, self._rest = data[:cut], data[cut:]
        self._ready = self._check(chunk)

    def _check(self, chunk: bytes) -> bytes:
        if self._utf8 or b"\0" not in chunk:
            try:
                chunk.decode()
            except UnicodeDecodeError:
                pass
            else:
                self._utf8 = self._utf8 or not chunk.isascii()
                self._line += chunk.count(b"\n") + chunk.count(b"\r") - chunk.count(b"\r\n")
                return chunk
        lines = []
        for line in chunk.splitlines(keepends=True):
            if not self._utf8 and b"\0" in line:
                raise EncodingError(self._line)
            try:
                line.decode()
            except UnicodeDecodeError:
                if not self._utf8:
                    raise EncodingError(self._line) from None
                self.invalid = self.invalid or self._line
                line = line.decode(errors="replace").encode()
            else:
                self._utf8 = self._utf8 or not line.isascii()
            lines.append(line)
            self._line += line.endswith((b"\n", b"\r"))
        return b"".join(lines)

    def lines(self) -> Iterator[tuple[int, str] | None]:
        """
        Строки входа с номерами, без концов строк

        Описание:
            None приходит перед чтением, которое может ждать ввода: накопленное пора разобрать

        Вывод:
            Iterator[tuple[int, str] | None]: Номер и строка или None
        """
        # Начало незаконченной строки - кусками: длинная строка склеивается один раз
        pending: list[str] = []
        number, start = 0, True
        # Кусок может кончиться посреди символа
        decoder = codecs.getincrementaldecoder("utf-8")()
        while True:
            yield None
            chunk = self.read()
            if not chunk:
                break
            data = decoder.decode(chunk)
            if start and data:
                data, start = data.removeprefix("\ufeff"), False
            pending.append(data)
            if not LINE_END.search(data):
                continue
            text, held = "".join(pending), ""
            # CR в конце куска может оказаться началом CRLF: ждет следующего куска
            if text.endswith("\r"):
                text, held = text[:-1], "\r"
            *complete, rest = LINE_END.split(text)
            pending = [rest + held] if rest or held else []
            for line in complete:
                number += 1
                yield number, line
        if pending:
            yield number + 1, "".join(pending).removesuffix("\r")


def _boundary(data: bytes) -> int:
    """Длина начала data без незаконченного символа UTF-8 в конце"""
    for back in range(1, min(4, len(data)) + 1):
        byte = data[-back]
        if byte & 0xC0 != 0x80:
            need = 1 if byte < 0xC0 else 2 if byte < 0xE0 else 3 if byte < 0xF0 else 4
            return len(data) - back if need > back else len(data)
    return len(data)


class Output:
    """
    Вывод результата: ошибки записи - WriteError, в терминал - сразу

    Аргументы:
        file (BinaryIO): Файл, открытый на запись в двоичном режиме
        interactive (bool): Сбрасывать буфер после каждой записи
    """

    def __init__(self, file: BinaryIO, interactive: bool) -> None:
        self.file = file
        self._interactive = interactive

    def write(self, text: str) -> None:
        """
        Запись текста в UTF-8

        Аргументы:
            text (str): Текст

        Исключения:
            WriteError: Если запись не удалась
        """
        try:
            self.file.write(text.encode(errors="replace"))
            if self._interactive:
                self.file.flush()
        except OSError as error:
            raise WriteError(error) from None

    def flush(self) -> None:
        """
        Сброс буфера

        Исключения:
            WriteError: Если запись не удалась
        """
        try:
            self.file.flush()
        except OSError as error:
            raise WriteError(error) from None


@dataclass
class Block:
    """Предложение CoNLL-U: номер первой строки и строки"""

    start: int
    lines: list[str]
    tagged: bool = True

    def check(self) -> InputError | None:
        """
        Части речи слов предложения

        Описание:
            Предложение без частей речи помечается для ruMorph; часть речи не у всех слов
            и незнакомая часть речи - ошибки

        Вывод:
            InputError | None: Ошибка с номером строки или None
        """
        words = []
        for number, line in enumerate(self.lines, self.start):
            fields = line.split("\t")
            if len(fields) == 10 and fields[0].isdigit():
                words.append((number, fields[3]))
        self.tagged = any(upos != "_" for _, upos in words)
        for number, upos in words:
            if upos == "_" and self.tagged:
                return InputError(
                    number, "у слова нет части речи, а у других слов предложения она есть"
                )
            if upos != "_" and upos not in UPOS:
                return InputError(
                    number,
                    f"неизвестная часть речи '{upos}': ожидается одна из 17 частей речи UD "
                    "заглавными буквами",
                )
        return None


class Command:
    """
    Разбор входа команды и запись результата

    Аргументы:
        parser (Parser): Синтаксический анализатор
        output (Output): Вывод результата
    """

    def __init__(self, parser: Parser, output: Output) -> None:
        self.parser = parser
        self.output = output
        self.segmenter = anyseg.Segmenter("ru")
        self._analyzer: rumorph.Analyzer | None = None
        self._sentences = 0

    @property
    def analyzer(self) -> rumorph.Analyzer:
        """Морфологический анализатор ruMorph, загруженный при первом обращении"""
        if self._analyzer is None:
            self._analyzer = rumorph.Analyzer()
        return self._analyzer

    def text(self, source: Input) -> None:
        """
        Разбор текста по абзацам

        Описание:
            Файл начинает комментарий newdoc, абзац - newpar; номера предложений идут подряд
            по всем файлам, как у команд anyseg и rumorph

        Аргументы:
            source (Input): Вход
        """
        stream = self.segmenter.stream(source)
        newdoc = True
        for document in stream:
            if not document.sentences:
                continue
            self.analyzer.annotate(document)
            self.parser.annotate(document)
            conllu = document.to_conllu().removeprefix("# newdoc\n")
            if stream.continued:
                conllu = conllu.removeprefix("# newpar\n")
            conllu = SENT_ID.sub(self._sent_id, conllu)
            self._sentences += len(document.sentences)
            self.output.write("# newdoc\n" + conllu if newdoc else conllu)
            newdoc = False

    def _sent_id(self, match: re.Match[str]) -> str:
        return f"# sent_id = {int(match[1]) + self._sentences}"

    def conllu(self, source: Input) -> None:
        """
        Разбор CoNLL-U по предложениям

        Описание:
            Предложения разбираются пачками: все, что прочитано до чтения, которое может
            ждать ввода; результат - как вход, с заполненными HEAD и DEPREL

        Аргументы:
            source (Input): Вход

        Исключения:
            InputError: Если строка CoNLL-U неверна; предложения до нее разобраны
        """
        pending: list[Block] = []
        lines: list[str] = []
        start = 0
        for item in source.lines():
            if item is None:
                self._blocks(pending)
                pending.clear()
                continue
            number, line = item
            if line.strip():
                start = start if lines else number
                lines.append(line)
            elif lines:
                pending.append(Block(start, lines))
                lines = []
        if lines:
            pending.append(Block(start, lines))
        self._blocks(pending)

    def _blocks(self, blocks: list[Block]) -> None:
        error = None
        for index, block in enumerate(blocks):
            error = block.check()
            if error is not None:
                # Ошибка строк того же предложения выше по файлу - первая
                try:
                    anyseg.read_conllu(_join([block]))
                except ValueError as found:
                    _, earlier = _locate([block], found)
                    error = earlier if earlier.line < error.line else error
                blocks = blocks[:index]
                break
        run: list[Block] = []
        for block in blocks:
            if run and run[0].tagged != block.tagged:
                self._run(run)
                run = []
            run.append(block)
        self._run(run)
        if error is not None:
            raise error

    def _run(self, blocks: list[Block]) -> None:
        if not blocks:
            return
        try:
            documents = anyseg.read_conllu(_join(blocks))
        except ValueError as error:
            index, failure = _locate(blocks, error)
            self._run(blocks[:index])
            raise failure from None
        for document in documents:
            if not blocks[0].tagged:
                self.analyzer.annotate(document)
            self.parser.annotate(document)
            self.output.write(document.to_conllu())


def _join(blocks: list[Block]) -> str:
    """Текст CoNLL-U предложений: каждое и пустая строка после него"""
    return "".join("\n".join(block.lines) + "\n\n" for block in blocks)


def _locate(blocks: list[Block], error: ValueError) -> tuple[int, InputError]:
    """Ошибка чтения CoNLL-U anySeg: номер предложения и ошибка с номером строки в файле по-русски"""
    match = CONLLU_ERROR.fullmatch(str(error))
    line = int(match[1]) if match else 1
    index, first = 0, 1
    while index + 1 < len(blocks) and first + len(blocks[index].lines) < line:
        first += len(blocks[index].lines) + 1
        index += 1
    message = match[2] if match else str(error)
    for pattern, russian in CONLLU_MESSAGES:
        message = pattern.sub(russian, message)
    # Ошибку предложения целиком anySeg ставит на пустую строку после него: номер - его первая строка
    offset = line - first
    if offset >= len(blocks[index].lines):
        offset = 0
    return index, InputError(blocks[index].start + offset, message)


def run(args: Sequence[str], stdin: _Raw | None, stdout: BinaryIO, stderr: TextIO) -> int:
    """
    Выполнение команды rusyntax

    Аргументы:
        args (Sequence[str]): Аргументы без имени программы
        stdin (_Raw | None): Стандартный ввод в двоичном режиме
        stdout (BinaryIO): Стандартный вывод в двоичном режиме
        stderr (TextIO): Поток сообщений об ошибках

    Вывод:
        int: Код выхода: 0 - успех, 1 - ошибка входа, модели или записи, 2 - неверные параметры
    """

    def report(message: str) -> None:
        # Если закрыт и поток ошибок, сказать уже некому
        with contextlib.suppress(OSError, ValueError):
            stderr.write(f"rusyntax: {message}\n")
            stderr.flush()

    try:
        options = parse_args(args)
    except UsageError as error:
        report(f"{error}\nСправка: rusyntax --help")
        return 2
    if isinstance(options, str):
        versions = f"anyseg {anyseg.__version__}, rumorph {rumorph.__version__}"
        text = HELP if options == "help" else f"rusyntax {__version__} ({versions})\n"
        output = Output(stdout, interactive=False)
        try:
            output.write(text)
            output.flush()
        except WriteError as error:
            return _write_error(error, stdout, report)
        return 0
    try:
        parser = Parser(options.model)
    except OSError as error:
        model = options.model
        report(f"{model}: {_describe(error, model)}" if model is not None else str(error))
        return 1
    except ValueError as error:
        # Модуль пишет абсолютный путь, команда - путь как задан, как у входных файлов
        _, marker, detail = str(error).partition(": неверный формат модели: ")
        if options.model is not None and marker:
            report(f"{options.model}{marker}{detail}")
        else:
            report(str(error))
        return 1
    files = options.files or ["-"]
    if options.output is None:
        output = Output(stdout, interactive=_isatty(stdout))
    else:
        opened = _open_output(options, files, stdin, report)
        if opened is None:
            return 1
        output = Output(opened, interactive=False)
    command = Command(parser, output)
    code = 0
    try:
        for file in files:
            code = max(code, _process(command, file, stdin, options.conllu, report))
        output.flush()
    except WriteError as error:
        code = max(code, _write_error(error, stdout if options.output is None else None, report))
    finally:
        if options.output is not None:
            with contextlib.suppress(OSError):
                output.file.close()
    return code


def _process(
    command: Command, file: str, stdin: _Raw | None, conllu: bool, report: Callable[[str], None]
) -> int:
    """Разбор одного файла; код выхода: 0 или 1"""
    try:
        with contextlib.ExitStack() as stack:
            if file == "-":
                raw: _Raw = stdin if stdin is not None else _Empty()
            else:
                raw = stack.enter_context(open(file, "rb", buffering=0))
            source = Input(raw, command.output.flush)
            if conllu:
                command.conllu(source)
            else:
                command.text(source)
    except EncodingError as error:
        report(f"{file}: файл не в UTF-8 (строка {error.line})")
        return 1
    except InputError as error:
        report(f"{file}: строка {error.line}: {error.message}")
        return 1
    except OSError as error:
        report(f"{file}: {_describe(error, file)}")
        return 1
    if source.invalid is not None:
        report(f"{file}: строка {source.invalid}: не UTF-8, неверные байты заменены на U+FFFD")
        return 1
    return 0


class _Empty:
    """Пустой вход: стандартного ввода нет (pythonw)"""

    def read(self, size: int, /) -> bytes:
        return b""


def _open_output(
    options: Options, files: list[str], stdin: _Raw | None, report: Callable[[str], None]
) -> BinaryIO | None:
    """Файл вывода, создаваемый заново; None - он не создается или затер бы вход или модель"""
    path = options.output or ""
    with contextlib.suppress(OSError):
        target = os.stat(path)
        if stat.S_ISREG(target.st_mode):
            sources: list[os.stat_result] = []
            for file in files:
                with contextlib.suppress(OSError, AttributeError, ValueError):
                    if file == "-":
                        sources.append(os.fstat(stdin.fileno()))  # type: ignore[union-attr]
                    else:
                        sources.append(_opened_stat(file))
            if options.model is not None:
                with contextlib.suppress(OSError):
                    sources.append(os.stat(options.model))
            if any(os.path.samestat(target, source) for source in sources):
                report(f"{path}: файл вывода совпадает с входным файлом или файлом модели")
                return None
    try:
        return open(path, "wb")
    except OSError as error:
        report(f"{path}: {_describe(error, path)}")
        return None


def _opened_stat(path: str) -> os.stat_result:
    """Метаданные входного файла по открытому дескриптору: канал и сокет не открываются"""
    # На macOS stat("/dev/stdin") - файл devfs, а не перенаправленный в него файл
    info = os.stat(path)
    if stat.S_ISFIFO(info.st_mode) or stat.S_ISSOCK(info.st_mode):
        return info
    try:
        with open(path, "rb", buffering=0) as file:
            return os.fstat(file.fileno())
    except OSError:
        return info


def _describe(error: OSError, path: str) -> str:
    """Ошибка ввода-вывода по-русски для частых случаев"""
    if isinstance(error, IsADirectoryError) or (
        isinstance(error, PermissionError) and os.path.isdir(path)
    ):
        return "это каталог"
    if isinstance(error, FileNotFoundError):
        return "файл не найден"
    if isinstance(error, PermissionError):
        return "нет доступа"
    return error.strerror or str(error)


def _isatty(file: BinaryIO) -> bool:
    try:
        return file.isatty()
    except (OSError, ValueError):
        return False


def _write_error(error: WriteError, stdout: BinaryIO | None, report: Callable[[str], None]) -> int:
    """Код выхода после ошибки записи: закрытый канал (rusyntax ... | head) - не ошибка"""
    closed = isinstance(error.error, BrokenPipeError) or (
        sys.platform == "win32" and error.error.errno == errno.EINVAL
    )
    if not closed:
        report(f"ошибка записи: {error.error}")
        return 1
    # Остаток буфера уходит в никуда, а не в закрытый канал при выходе из Python
    if stdout is not None:
        with contextlib.suppress(OSError, ValueError, AttributeError):
            null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null, stdout.fileno())
            os.close(null)
    return 0


def main() -> NoReturn:
    """
    Точка входа команды rusyntax

    Описание:
        Ctrl+C завершает команду сразу, как обычную программу, а игнорируемый SIGINT (фоновая
        команда сценария) остается игнорируемым; обработчик Python возвращается после команды

    Исключения:
        SystemExit: Всегда, с кодом выхода команды
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.flush()
    handler = None
    main_thread = threading.current_thread() is threading.main_thread()
    if main_thread and signal.getsignal(signal.SIGINT) is signal.default_int_handler:
        handler = signal.signal(signal.SIGINT, signal.SIG_DFL)
    # Без консоли (pythonw) стандартных потоков нет
    stdin = getattr(sys.stdin, "buffer", None)
    stdout = getattr(sys.stdout, "buffer", None) or io.BytesIO()
    stderr: TextIO = sys.stderr or io.StringIO()
    # Сообщения в файл или канал - в UTF-8, как результат и как у команд anyseg и rumorph (в Windows
    # Python до 3.15 пишет их в кодировке ANSI); консоль Python и так пишет в Юникоде
    utf8 = None
    error_buffer = getattr(sys.stderr, "buffer", None)
    if (
        error_buffer is not None
        and not getattr(error_buffer, "closed", True)
        and not _isatty(error_buffer)
    ):
        # Концы строк - LF, как у команд на Rust, и в Windows
        stderr = utf8 = io.TextIOWrapper(
            error_buffer,
            encoding="utf-8",
            errors="backslashreplace",
            newline="\n",
            write_through=True,
        )
    try:
        code = run(sys.argv[1:], getattr(stdin, "raw", stdin), stdout, stderr)
    finally:
        if utf8 is not None:
            # Обертка не закрывает поток ошибок Python
            with contextlib.suppress(ValueError):
                utf8.detach()
        if handler is not None:
            signal.signal(signal.SIGINT, handler)
    sys.exit(code)


if __name__ == "__main__":
    main()
