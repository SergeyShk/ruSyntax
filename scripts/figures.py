"""
Картинки документации: как ruSyntax разбирает предложение

Описание:
    Рисует по настоящему разбору модели пакета: конвейер anySeg → ruMorph → ruSyntax
    (docs/img/pipeline.svg), дуги разбора со школьными названиями членов предложения
    (docs/img/arcs.svg), устройство модели (docs/img/model.svg) и матрицу вероятностей
    вершин неоднозначного предложения (docs/img/heads.svg); к каждой - английская версия
    (*.en.svg). Вероятности вершин считает прямой проход модели на numpy по файлу весов
    (data/ru-syntax.bin), сверенный с разбором ядра на Rust. Запуск - в окружении проекта:
    нужен собранный rusyntax, а numpy подключается на время запуска.

Пример использования:
    uv run --with numpy python scripts/figures.py
"""

import json
import struct
import zlib
from pathlib import Path

import numpy as np

from rusyntax import Parser

ROOT = Path(__file__).parents[1]
MODEL = ROOT / "data" / "ru-syntax.bin"
IMG = ROOT / "docs" / "img"
EXAMPLE = "Вчера маленькая кошка долго спала на теплом крыльце."
AMBIGUOUS = "Я видел девочку с биноклем."


def read_model(path: Path) -> dict:
    """
    Записи файла модели ruSyntax: строки, матрицы (int8 по строкам - деквантованные) и числа

    Аргументы:
        path (Path): Файл модели

    Вывод:
        dict: Записи по именам

    Исключения:
        ValueError: Если файл не модель ruSyntax или поврежден
    """
    data = path.read_bytes()
    if data[:8] != b"RUSYNTAX" or zlib.crc32(data[16:]) != struct.unpack_from("<I", data, 12)[0]:
        raise ValueError(f"{path}: не модель ruSyntax или файл поврежден")
    at = 20
    entries = {}
    for _ in range(struct.unpack_from("<I", data, 16)[0]):
        (length,) = struct.unpack_from("<H", data, at)
        name = data[at + 2 : at + 2 + length].decode()
        at += 2 + length
        kind = data[at]
        at += 1
        if kind == 0:
            (count,) = struct.unpack_from("<I", data, at)
            at += 4
            items = []
            for _ in range(count):
                (size,) = struct.unpack_from("<I", data, at)
                items.append(data[at + 4 : at + 4 + size].decode())
                at += 4 + size
            entries[name] = items
        elif kind in (1, 2):
            rows, cols = struct.unpack_from("<II", data, at)
            at += 8
            if kind == 1:
                matrix = np.frombuffer(data, "<f4", rows * cols, at).reshape(rows, cols)
                at += 4 * rows * cols
            else:
                scales = np.frombuffer(data, "<f4", rows, at)
                at += 4 * rows
                values = np.frombuffer(data, "i1", rows * cols, at).reshape(rows, cols)
                at += rows * cols
                matrix = values.astype(np.float32) * scales[:, None]
            entries[name] = matrix.astype(np.float32)
        else:
            (count,) = struct.unpack_from("<I", data, at)
            entries[name] = list(struct.unpack_from(f"<{count}I", data, at + 4))
            at += 4 + 4 * count
    return entries


class Numpy:
    """
    Прямой проход модели на numpy до вероятностей вершин - для картинок, не для разбора

    Аргументы:
        path (Path): Файл модели
    """

    def __init__(self, path: Path) -> None:
        self.e = read_model(path)
        keys = (
            "hidden",
            "layers",
            "word_dim",
            "suffix_dim",
            "tag_dim",
            "arc_dim",
            "rel_dim",
            "max_feats",
        )
        self.config = dict(zip(keys, self.e["config"], strict=True))
        self.index = {
            name: {text: i + 2 for i, text in enumerate(self.e[f"vocab.{name}"])}
            for name in ("words", "suffixes", "upos", "feats")
        }

    def row(self, table: str, matrix: str, key: str | None) -> np.ndarray:
        return self.e[f"embed.{matrix}"][1 if key is None else self.index[table].get(key, 1)]

    def embed(self, forms: list[str], upos: list[str], feats: list[str]) -> np.ndarray:
        rows = []
        for i in range(len(forms) + 1):
            form = None if i == 0 else forms[i - 1].lower().replace("ё", "е")
            word = "^" + (form or "")
            parts = [self.row("words", "word", form)]
            parts += [
                self.row("suffixes", "suffix", None if form is None else f"{k}:{word[-k:]}")
                for k in range(1, 5)
            ]
            parts.append(self.row("upos", "upos", None if i == 0 else upos[i - 1]))
            total = np.zeros(self.config["tag_dim"], np.float32)
            if i:
                pairs = [p for p in feats[i - 1].split("|") if p and p != "_"]
                for pair in pairs[: self.config["max_feats"]]:
                    total += self.row("feats", "feats", pair)
            parts.append(total)
            rows.append(np.concatenate(parts))
        return np.stack(rows)

    def lstm(self, x: np.ndarray, prefix: str, reverse: bool) -> np.ndarray:
        ih, hh, bias = (self.e[f"{prefix}.{name}"] for name in ("ih", "hh", "bias"))
        hidden = hh.shape[1]
        h = np.zeros(hidden, np.float32)
        c = np.zeros(hidden, np.float32)
        out = np.zeros((len(x), hidden), np.float32)
        steps = range(len(x) - 1, -1, -1) if reverse else range(len(x))
        for t in steps:
            g = ih @ x[t] + bias[0] + hh @ h
            i, f, cand, o = np.split(g, 4)
            c = 1 / (1 + np.exp(-f)) * c + 1 / (1 + np.exp(-i)) * np.tanh(cand)
            h = 1 / (1 + np.exp(-o)) * np.tanh(c)
            out[t] = h
        return out

    def dense(self, x: np.ndarray, name: str) -> np.ndarray:
        y = x @ self.e[f"{name}.weight"].T + self.e[f"{name}.bias"][0]
        return np.where(y > 0, y, 0.1 * y)

    def head_probabilities(
        self, forms: list[str], upos: list[str], feats: list[str]
    ) -> np.ndarray:
        """
        Вероятности вершин: строка - слово (с 1), столбец - кандидат (0 - корень)

        Аргументы:
            forms (list[str]): Формы
            upos (list[str]): Части речи
            feats (list[str]): Признаки

        Вывод:
            np.ndarray: Матрица n × (n + 1)
        """
        x = self.embed(forms, upos, feats)
        for layer in range(self.config["layers"]):
            x = np.concatenate(
                [
                    self.lstm(x, f"lstm.{layer}.forward", False),
                    self.lstm(x, f"lstm.{layer}.backward", True),
                ],
                1,
            )
        dep = self.dense(x, "arc_dep")
        head = self.dense(x, "arc_head")
        dep = np.concatenate([dep, np.ones((len(dep), 1), np.float32)], 1)
        scores = dep @ self.e["arc.weight"] @ head.T
        np.fill_diagonal(scores, -np.inf)
        scores = scores[1:]
        scores -= scores.max(1, keepdims=True)
        probs = np.exp(scores)
        return probs / probs.sum(1, keepdims=True)


DARK, LIGHT, INK, MUTED, SOFT = "#1976d2", "#64b5f6", "#263238", "#78909c", "#e3f2fd"
FONT = "PT Sans, Source Sans Pro, Arial, sans-serif"
SUB = '<tspan baseline-shift="sub" font-size="70%">{}</tspan>'
SUP = '<tspan baseline-shift="super" font-size="70%">{}</tspan>'
TEXTS = {
    "ru": {
        "pos": {
            "NOUN": "сущ.",
            "VERB": "глаг.",
            "ADJ": "прил.",
            "ADV": "нареч.",
            "ADP": "предлог",
            "PRON": "мест.",
            "PUNCT": "знак",
            "PROPN": "имя",
            "DET": "мест.",
            "NUM": "числ.",
        },
        "case": {
            "Nom": "им. п.",
            "Gen": "род. п.",
            "Dat": "дат. п.",
            "Acc": "вин. п.",
            "Ins": "твор. п.",
            "Loc": "пр. п.",
        },
        "rel": {
            "root": "сказуемое",
            "nsubj": "подлежащее",
            "obj": "дополнение",
            "iobj": "дополнение",
            "obl": "обстоятельство",
            "advmod": "обстоятельство",
            "amod": "определение",
            "det": "определение",
            "nmod": "определение",
            "case": "предлог",
            "punct": "знак",
        },
        "steps": (
            "1. Текст",
            "2. anySeg делит текст на слова",
            "3. ruMorph ставит часть речи и падеж",
            "4. ruSyntax находит, от какого слова зависит каждое",
        ),
        "root": "ROOT",
        "model": (
            ("форма", "окончания 1-4", "UPOS", "признаки"),
            "вектор слова",
            "BiLSTM {layers} × {hidden}, в обе стороны: {out} на слово",
            ("MLP вершины: {arc} и {arc}", "MLP связи: {rel} и {rel}"),
            (
                "биаффинная оценка s(i, j) = [d"
                + SUB.format("i")
                + "; 1]"
                + SUP.format("T")
                + " W h"
                + SUB.format("j"),
                "линейный слой по [d" + SUB.format("i") + "; h вершины]",
            ),
            ("softmax по j: P(вершина j | слово i)", "{labels} типов связи UD"),
            "Чу - Лю - Эдмондс: дерево с одним корнем",
            "вершина и тип связи каждого слова · {params} млн параметров, int8 по строкам",
        ),
        "heads": (
            "Слово",
            "Кандидат в вершины",
            "P(вершина | слово): softmax биаффинных оценок по строке; "
            "рамка - вершина в дереве разбора",
        ),
    },
    "en": {
        "pos": {
            "NOUN": "noun",
            "VERB": "verb",
            "ADJ": "adj.",
            "ADV": "adv.",
            "ADP": "prep.",
            "PRON": "pron.",
            "PUNCT": "punct.",
            "PROPN": "name",
            "DET": "det.",
            "NUM": "num.",
        },
        "case": {
            "Nom": "nom.",
            "Gen": "gen.",
            "Dat": "dat.",
            "Acc": "acc.",
            "Ins": "ins.",
            "Loc": "loc.",
        },
        "rel": {
            "root": "predicate",
            "nsubj": "subject",
            "obj": "object",
            "iobj": "object",
            "obl": "adverbial",
            "advmod": "adverbial",
            "amod": "modifier",
            "det": "modifier",
            "nmod": "modifier",
            "case": "preposition",
            "punct": "punct.",
        },
        "steps": (
            "1. Text",
            "2. anySeg splits the text into words",
            "3. ruMorph gives the part of speech and case",
            "4. ruSyntax finds the word every word depends on",
        ),
        "root": "ROOT",
        "model": (
            ("form", "endings 1-4", "UPOS", "features"),
            "word vector",
            "BiLSTM {layers} × {hidden}, both directions: {out} per word",
            ("head MLP: {arc} and {arc}", "relation MLP: {rel} and {rel}"),
            (
                "biaffine score s(i, j) = [d"
                + SUB.format("i")
                + "; 1]"
                + SUP.format("T")
                + " W h"
                + SUB.format("j"),
                "linear layer over [d" + SUB.format("i") + "; h of the head]",
            ),
            ("softmax over j: P(head j | word i)", "{labels} UD relations"),
            "Chu - Liu - Edmonds: a tree with one root",
            "the head and relation of every word · {params} million parameters, int8 by rows",
        ),
        "heads": (
            "Word",
            "Head candidate",
            "P(head | word): softmax of the biaffine scores over the row; "
            "a frame marks the head in the parse tree",
        ),
    },
}


def width(text: str, size: float) -> float:
    return len(text) * size * 0.56 + 6


def text(
    x: float,
    y: float,
    value: str,
    size: float = 16,
    color: str = INK,
    bold: bool = False,
    anchor: str = "middle",
) -> str:
    weight = ' font-weight="bold"' if bold else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}"{weight} '
        f'text-anchor="{anchor}">{value}</text>'
    )


def arrow(x: float, y: float, size: float = 7, color: str = LIGHT) -> str:
    return (
        f'<path d="M{x - size:.1f} {y - 1.6 * size:.1f}L{x + size:.1f} {y - 1.6 * size:.1f}'
        f'L{x:.1f} {y:.1f}Z" fill="{color}"/>'
    )


def save(name: str, lang: str, w: float, h: float, body: list[str]) -> None:
    """Картинка с белой подложкой: в темной теме сайта она остается читаемой"""
    suffix = "" if lang == "ru" else ".en"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" '
        f'height="{h:.0f}" font-family="{FONT}" role="img">\n'
        f'<rect width="{w:.0f}" height="{h:.0f}" rx="12" fill="white"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )
    (IMG / f"{name}{suffix}.svg").write_text(svg, encoding="utf-8")


def layout(
    words: list[dict], lang: str, size: float, gap: float, x0: float
) -> tuple[list[float], float]:
    xs, x = [], x0
    for w in words:
        wd = max(width(w["form"], size), width(TEXTS[lang]["pos"].get(w["upos"], ""), 13))
        xs.append(x + wd / 2)
        x += wd + gap
    return xs, x - gap + x0


def arcs(
    words: list[dict], lang: str, size: float = 22, x0: float = 40
) -> tuple[list[str], list[float], float, float]:
    """Дуги от вершины к зависимому над строкой слов (строка - на y = 0); высота - по длине связи"""
    rel = TEXTS[lang]["rel"]
    xs, total = layout(words, lang, size, 34, x0)
    body, top = [], 0.0
    order = sorted(range(len(words)), key=lambda i: abs(words[i]["head"] - 1 - i))
    for i in order:
        w = words[i]
        if w["head"] == 0:
            continue
        h = w["head"] - 1
        x1, x2 = xs[h], xs[i]
        height = 30 + 46 * abs(h - i)
        y, cy = -8, -8 - height
        top = min(top, cy)
        start = x1 + (6 if x2 > x1 else -6)
        body.append(
            f'<path fill="none" stroke="{LIGHT}" stroke-width="2.5" '
            f'd="M{start:.1f} {y}C{x1:.1f} {cy:.1f} {x2:.1f} {cy:.1f} {x2:.1f} {y - 9}"/>'
        )
        body.append(arrow(x2, y))
        name = rel.get(w["deprel"], w["deprel"])
        mx, my = (x1 + x2) / 2, cy + 0.25 * height - 6
        body.append(
            f'<rect x="{mx - width(name, 13) / 2 - 4:.1f}" y="{my - 14:.1f}" '
            f'width="{width(name, 13) + 8:.1f}" height="30" rx="5" fill="white"/>'
        )
        body.append(text(mx, my, name, 13, DARK, True))
        body.append(text(mx, my + 12, w["deprel"], 10, MUTED))
    return body, xs, total, top


def draw_arcs(words: list[dict], lang: str) -> None:
    body, xs, total, top = arcs(words, lang)
    root = next(i for i, w in enumerate(words) if w["head"] == 0)
    body += [
        f'<path fill="none" stroke="{DARK}" stroke-width="2.5" '
        f'd="M{xs[root]:.1f} {top - 30:.1f}L{xs[root]:.1f} -18"/>',
        arrow(xs[root], -8, 7, DARK),
        text(xs[root] + 10, top - 20, TEXTS[lang]["rel"]["root"], 13, DARK, True, "start"),
    ]
    for w, x in zip(words, xs, strict=True):
        body.append(text(x, 20, w["form"], 22, INK, True))
        body.append(text(x, 40, TEXTS[lang]["pos"].get(w["upos"], w["upos"]), 13, MUTED))
    shift = -top + 50
    save(
        "arcs",
        lang,
        total,
        shift + 60,
        [f'<g transform="translate(0,{shift:.0f})">', *body, "</g>"],
    )


def case_of(feats: str, lang: str) -> str:
    for pair in feats.split("|"):
        if pair.startswith("Case="):
            return TEXTS[lang]["case"].get(pair[5:], "")
    return ""


def draw_pipeline(words: list[dict], lang: str) -> None:
    steps = TEXTS[lang]["steps"]
    body = [
        text(20, 30, steps[0], 15, DARK, True, "start"),
        text(20, 60, EXAMPLE, 20, INK, False, "start"),
    ]
    xs, total = layout(words, lang, 18, 24, 20)
    body.append(text(20, 100, steps[1], 15, DARK, True, "start"))
    for w, x in zip(words, xs, strict=True):
        bw = width(w["form"], 18) + 10
        body.append(
            f'<rect x="{x - bw / 2:.1f}" y="110" width="{bw:.1f}" height="30" rx="6" fill="white" '
            f'stroke="{LIGHT}" stroke-width="1.5"/>'
        )
        body.append(text(x, 131, w["form"], 18))
    body.append(text(20, 170, steps[2], 15, DARK, True, "start"))
    for w, x in zip(words, xs, strict=True):
        body.append(text(x, 198, w["form"], 18))
        body.append(text(x, 216, TEXTS[lang]["pos"].get(w["upos"], ""), 12, DARK))
        body.append(text(x, 230, case_of(w["feats"], lang), 12, MUTED))
    body.append(text(20, 270, steps[3], 15, DARK, True, "start"))
    arc, axs, atotal, atop = arcs(words, lang, 18, 20)
    shift = 290 - atop
    body.append(f'<g transform="translate(0,{shift:.0f})">')
    body += arc + [text(x, 18, w["form"], 18, INK, True) for w, x in zip(words, axs, strict=True)]
    body.append("</g>")
    save("pipeline", lang, max(total, atotal) + 20, shift + 40, body)


def box(
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    fill: str = SOFT,
    color: str = INK,
    size: float = 15,
) -> list[str]:
    return [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" fill="{fill}" '
        f'stroke="{DARK}" stroke-width="1.5"/>',
        text(x + w / 2, y + h / 2 + size * 0.35, label, size, color),
    ]


def down(x: float, y1: float, y2: float) -> list[str]:
    return [
        f'<path d="M{x:.1f} {y1:.1f}L{x:.1f} {y2 - 9:.1f}" stroke="{DARK}" stroke-width="2"/>',
        arrow(x, y2, 6, DARK),
    ]


def draw_model(model: Numpy, lang: str, labels: int, params: float) -> None:
    c = model.config
    t = TEXTS[lang]["model"]
    din = c["word_dim"] + 4 * c["suffix_dim"] + 2 * c["tag_dim"]
    fmt = {
        "layers": c["layers"],
        "hidden": c["hidden"],
        "out": 2 * c["hidden"],
        "arc": c["arc_dim"],
        "rel": c["rel_dim"],
        "labels": labels,
        "params": f"{params:.1f}".replace(".", "," if lang == "ru" else "."),
    }
    w, body = 880, []
    dims = (str(c["word_dim"]), f"4 × {c['suffix_dim']}", str(c["tag_dim"]), f"Σ {c['tag_dim']}")
    for k, (name, dim) in enumerate(zip(t[0], dims, strict=True)):
        x = 40 + k * 205
        body += box(x, 20, 185, 54, "", "white")
        body += [text(x + 92, 43, name, 15), text(x + 92, 63, dim, 13, MUTED)]
        body += down(x + 92, 74, 104)
    body += box(40, 104, 800, 40, f"{t[1]}: {din}")
    body += down(440, 144, 172)
    body += box(40, 172, 800, 44, t[2].format(**fmt), DARK, "white", 16)
    left, right = 40, 470
    body += down(left + 185, 216, 246) + down(right + 185, 216, 246)
    rows = [(t[3], 246), (t[4], 312), (t[5], 378)]
    for pair, y in rows:
        body += box(left, y, 370, 44, pair[0].format(**fmt), size=14)
        body += box(right, y, 370, 44, pair[1].format(**fmt), size=14)
        if y < 378:
            body += down(left + 185, y + 44, y + 66) + down(right + 185, y + 44, y + 66)
    body += down(left + 185, 422, 444)
    body += box(left, 444, 370, 44, t[6], size=14)
    body += down(left + 185, 488, 516) + down(right + 185, 422, 516)
    body += box(40, 516, 800, 44, t[7].format(**fmt), DARK, "white", 15)
    save("model", lang, w, 580, body)


def draw_heads(words: list[dict], probs: np.ndarray, lang: str) -> None:
    t = TEXTS[lang]["heads"]
    n = len(words)
    cell, left, topy = 74, 150, 110
    candidates = [TEXTS[lang]["root"]] + [w["form"] for w in words]
    body = [
        text(left + cell * (n + 1) / 2, 30, t[1], 15, DARK, True),
        text(30, topy + cell * n / 2, t[0], 15, DARK, True, "start"),
    ]
    for j, name in enumerate(candidates):
        body.append(text(left + j * cell + cell / 2, topy - 14, name, 15, INK, True))
    for i, w in enumerate(words):
        y = topy + i * cell
        body.append(text(left - 12, y + cell / 2 + 5, w["form"], 16, INK, True, "end"))
        for j in range(n + 1):
            x = left + j * cell
            if j == i + 1:
                body.append(
                    f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" fill="#eceff1"/>'
                )
                continue
            p = float(probs[i, j])
            alpha = min(1.0, p) ** 0.6
            body.append(
                f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" fill="{DARK}" '
                f'fill-opacity="{alpha:.3f}"/>'
            )
            if p >= 0.005:
                value = f"{p:.2f}".replace(".", "," if lang == "ru" else ".")
                body.append(
                    text(
                        x + cell / 2,
                        y + cell / 2 + 6,
                        value,
                        17,
                        "white" if p > 0.35 else INK,
                        True,
                    )
                )
            if words[i]["head"] == j:
                body.append(
                    f'<rect x="{x + 3}" y="{y + 3}" width="{cell - 6}" height="{cell - 6}" fill="none" '
                    f'stroke="{INK}" stroke-width="3" rx="4"/>'
                )
    body.append(text(30, topy + cell * n + 34, t[2], 14, MUTED, False, "start"))
    save("heads", lang, left + cell * (n + 1) + 30, topy + cell * n + 54, body)


def tagged(parser: Parser, text: str) -> list[dict]:
    document = parser.parse_text(text)
    return [
        {
            "form": w.form,
            "upos": w.upos,
            "feats": w.feats or "_",
            "head": w.head,
            "deprel": w.deprel,
        }
        for w in document.words
    ]


def main() -> None:
    parser = Parser()
    model = Numpy(MODEL)
    params = sum(v.size for v in model.e.values() if isinstance(v, np.ndarray)) / 1e6
    example, ambiguous = tagged(parser, EXAMPLE), tagged(parser, AMBIGUOUS)
    probs = model.head_probabilities(
        [w["form"] for w in ambiguous],
        [w["upos"] for w in ambiguous],
        [w["feats"] for w in ambiguous],
    )
    # Прямой проход на numpy выбирает те же вершины, что ядро на Rust
    assert [int(np.argmax(row)) for row in probs] == [w["head"] for w in ambiguous]
    for lang in TEXTS:
        draw_pipeline(example, lang)
        draw_arcs(example, lang)
        draw_model(model, lang, len(parser.relations), params)
        draw_heads(ambiguous, probs, lang)
    print(json.dumps({"params_million": round(params, 2), "ambiguous": probs.round(3).tolist()}))


if __name__ == "__main__":
    main()
