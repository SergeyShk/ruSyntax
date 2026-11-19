# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "numpy==2.5.5",
#     "rumorph==0.6.0",
#     "torch==2.15.0",
# ]
# ///
"""
Обучение модели ruSyntax: вершина и тип связи каждого слова по словам с тегами ruMorph

Описание:
    Берет части train и dev UD_Russian-Taiga, UD_Russian-GSD и UD_Russian-Poetry из каталога
    `scripts/export_ud.py`, ставит словам часть речи и признаки ruMorph с контекстом (на тех же
    тегах парсер работает у пользователя) и учит модель: вход слова - форма (20 тыс. самых
    частых, строчные, «ё» как «е»), окончания длиной 1-4 с меткой начала, UPOS и сумма
    вложений первых 12 пар признаков; BiLSTM 2×128; вершина - биаффинная оценка и дерево
    Чу - Лю - Эдмондса с одним корнем, связь - линейная оценка пары «зависимое - вершина».
    Точка обучения выбирается по среднему LAS dev Taiga и GSD. Веса квантуются в int8
    по строкам и пишутся в формат ядра (`--output`, запись `sources` - источники и dev);
    в `--check` пишется разбор части dev той же квантованной моделью на PyTorch - эталон теста
    крейта, сверяющего прямой проход на Rust. Версии пакетов закреплены в train.py.lock.
    На MPS и GPU обучение недетерминировано: повторный запуск дает близкие, но не те же веса.

Пример использования:
    uv run scripts/train.py target/ud --output data/ru-syntax.bin --check data/check.conllu
"""

import argparse
import json
import random
import struct
import time
import zlib
from collections import Counter
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path

import numpy as np
import rumorph
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

TRAIN = [f"ru_taiga-ud-train-{part}.conllu" for part in "abcde"] + [
    "ru_gsd-ud-train.conllu",
    "ru_poetry-ud-train.conllu",
]
DEV = {
    "Taiga": "ru_taiga-ud-dev.conllu",
    "GSD": "ru_gsd-ud-dev.conllu",
    "Poetry": "ru_poetry-ud-dev.conllu",
}
# Наборы dev, по которым выбирается точка обучения
SELECT = ("Taiga", "GSD")
# Предложений каждого dev в файле сверки с ядром
CHECK = 100
CONFIG = {
    "hidden": 128,
    "layers": 2,
    "word_dim": 64,
    "suffix_dim": 16,
    "tag_dim": 50,
    "arc_dim": 128,
    "rel_dim": 64,
    "max_feats": 12,
}
WORDS = 20000
DROPOUT = 0.33
WORD_DROPOUT = 0.25
PAD, UNKNOWN = 0, 1
MAGIC = b"RUSYNTAX"
FORMAT = 1

# Предложение: формы, UPOS, признаки, вершины, связи
Sentence = tuple[list[str], list[str], list[str], list[int], list[str]]


def read(path: Path) -> Iterator[list[list[str]]]:
    """
    Слова предложений CoNLL-U без многословных токенов и пустых узлов

    Аргументы:
        path (Path): Файл CoNLL-U

    Вывод:
        Iterator[list[list[str]]]: Строки слов предложения по 10 колонок
    """
    words: list[list[str]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                if words:
                    yield words
                words = []
            elif not line.startswith("#"):
                cols = line.split("\t")
                if cols[0].isdigit():
                    words.append(cols)
    if words:
        yield words


def feats_text(feats: dict[str, str]) -> str:
    """
    Признаки в записи CoNLL-U, упорядоченные без учета регистра, как в UD

    Аргументы:
        feats (dict[str, str]): Признаки разбора

    Вывод:
        str: Запись `A=B|C=D` или `_`
    """
    pairs = sorted(feats.items(), key=lambda pair: pair[0].lower())
    return "|".join(f"{name}={value}" for name, value in pairs) or "_"


def load(paths: list[Path], analyzer: rumorph.Analyzer) -> list[Sentence]:
    """
    Предложения с тегами ruMorph и эталонными вершинами и связями

    Аргументы:
        paths (list[Path]): Файлы CoNLL-U
        analyzer (rumorph.Analyzer): Морфологический анализатор

    Вывод:
        list[Sentence]: Предложения
    """
    out = []
    for path in paths:
        for words in read(path):
            forms = [w[1] for w in words]
            best = [analyses[0] for analyses in analyzer.analyze_sentence(forms)]
            out.append(
                (
                    forms,
                    [a.pos for a in best],
                    [feats_text(a.feats) for a in best],
                    [int(w[6]) for w in words],
                    [w[7] for w in words],
                )
            )
    return out


def norm(form: str) -> str:
    """
    Форма для словаря модели: строчные буквы, «ё» как «е» (как в ядре)

    Аргументы:
        form (str): Форма слова

    Вывод:
        str: Нормализованная форма
    """
    return form.lower().replace("ё", "е")


def suffixes(form: str) -> list[str]:
    """
    Ключи окончаний длиной 1-4 с меткой начала «^» (как в ядре)

    Аргументы:
        form (str): Форма слова

    Вывод:
        list[str]: Ключи `k:окончание`
    """
    word = "^" + norm(form)
    return [f"{k}:{word[-k:]}" for k in range(1, 5)]


def pairs(feats: str) -> list[str]:
    """
    Пары «признак=значение» записи CoNLL-U, не больше `max_feats` (как в ядре)

    Аргументы:
        feats (str): Запись признаков

    Вывод:
        list[str]: Пары
    """
    return [pair for pair in feats.split("|") if pair and pair != "_"][: CONFIG["max_feats"]]


class Vocab:
    """
    Словари модели: строки 0 и 1 таблиц - пустая и неизвестная

    Аргументы:
        sentences (list[Sentence]): Обучающие предложения
    """

    def __init__(self, sentences: list[Sentence]) -> None:
        words = Counter(norm(w) for s in sentences for w in s[0])
        sufs = Counter(x for s in sentences for w in s[0] for x in suffixes(w))
        self.words = sorted(w for w, c in words.most_common(WORDS) if c >= 2)
        self.suffixes = sorted(x for x, c in sufs.items() if c >= 3)
        self.upos = sorted({u for s in sentences for u in s[1]})
        self.feats = sorted({f for s in sentences for fs in s[2] for f in pairs(fs)})
        self.relations = sorted({r for s in sentences for r in s[4]})
        self.index = {
            name: {text: i + 2 for i, text in enumerate(getattr(self, name))}
            for name in ("words", "suffixes", "upos", "feats")
        }
        self.relation = {r: i for i, r in enumerate(self.relations)}

    def number(self, table: str, key: str) -> int:
        return self.index[table].get(key, UNKNOWN)


def encode(vocab: Vocab, sentences: list[Sentence]) -> tuple[torch.Tensor, ...]:
    """
    Тензоры пакета: позиция 0 - корень (строки неизвестного, без признаков)

    Аргументы:
        vocab (Vocab): Словари
        sentences (list[Sentence]): Предложения пакета

    Вывод:
        tuple[torch.Tensor, ...]: Слова, окончания, UPOS, признаки, вершины, связи, длины
    """
    n = max(len(s[0]) for s in sentences) + 1
    b = len(sentences)
    words = torch.zeros(b, n, dtype=torch.long)
    sufs = torch.zeros(b, n, 4, dtype=torch.long)
    upos = torch.zeros(b, n, dtype=torch.long)
    feats = torch.zeros(b, n, CONFIG["max_feats"], dtype=torch.long)
    heads = torch.zeros(b, n, dtype=torch.long)
    rels = torch.zeros(b, n, dtype=torch.long)
    lens = torch.tensor([len(s[0]) + 1 for s in sentences])
    for i, (forms, tags, fs, hs, rs) in enumerate(sentences):
        words[i, 0], sufs[i, 0], upos[i, 0] = UNKNOWN, UNKNOWN, UNKNOWN
        for j, form in enumerate(forms, 1):
            words[i, j] = vocab.number("words", norm(form))
            for k, key in enumerate(suffixes(form)):
                sufs[i, j, k] = vocab.number("suffixes", key)
            upos[i, j] = vocab.number("upos", tags[j - 1])
            for k, pair in enumerate(pairs(fs[j - 1])):
                feats[i, j, k] = vocab.number("feats", pair)
            heads[i, j] = hs[j - 1]
            rels[i, j] = vocab.relation.get(rs[j - 1], 0)
    return words, sufs, upos, feats, heads, rels, lens


def batches(sentences: list[Sentence], size: int, rng: random.Random | None) -> list[list[int]]:
    """
    Пакеты не больше `size` позиций с дополнением, предложения близкой длины вместе

    Аргументы:
        sentences (list[Sentence]): Предложения
        size (int): Позиций в пакете
        rng (random.Random | None): Перемешать пакеты

    Вывод:
        list[list[int]]: Номера предложений пакетов
    """
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i][0]))
    out: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for i in order:
        n = len(sentences[i][0]) + 1
        if current and max(longest, n) * (len(current) + 1) > size:
            out.append(current)
            current, longest = [], 0
        current.append(i)
        longest = max(longest, n)
    if current:
        out.append(current)
    if rng:
        rng.shuffle(out)
    return out


class Dense(nn.Module):
    """Слой LeakyReLU(Wx + b) с наклоном 0,1 и dropout"""

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(nn.functional.leaky_relu(self.lin(x), 0.1))


class Parser(nn.Module):
    """
    Модель: вложения, BiLSTM, биаффинная оценка вершин и линейная оценка связей

    Аргументы:
        vocab (Vocab): Словари
    """

    def __init__(self, vocab: Vocab) -> None:
        super().__init__()
        c = CONFIG
        self.word = nn.Embedding(len(vocab.words) + 2, c["word_dim"], padding_idx=PAD)
        self.suffix = nn.Embedding(len(vocab.suffixes) + 2, c["suffix_dim"], padding_idx=PAD)
        self.upos = nn.Embedding(len(vocab.upos) + 2, c["tag_dim"], padding_idx=PAD)
        self.feats = nn.EmbeddingBag(
            len(vocab.feats) + 2, c["tag_dim"], mode="sum", padding_idx=PAD
        )
        n_in = c["word_dim"] + 4 * c["suffix_dim"] + 2 * c["tag_dim"]
        self.lstm = nn.LSTM(
            n_in, c["hidden"], c["layers"], batch_first=True, bidirectional=True, dropout=DROPOUT
        )
        self.drop = nn.Dropout(DROPOUT)
        h2 = 2 * c["hidden"]
        self.arc_dep, self.arc_head = Dense(h2, c["arc_dim"]), Dense(h2, c["arc_dim"])
        self.rel_dep, self.rel_head = Dense(h2, c["rel_dim"]), Dense(h2, c["rel_dim"])
        # Биаффинная матрица (a + 1) × a: последняя строка - смещение зависимого
        self.arc = nn.Parameter(torch.zeros(c["arc_dim"] + 1, c["arc_dim"]))
        self.label = nn.Linear(2 * c["rel_dim"], len(vocab.relations))

    def forward(
        self,
        words: torch.Tensor,
        sufs: torch.Tensor,
        upos: torch.Tensor,
        feats: torch.Tensor,
        lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n = words.shape
        if self.training:
            drop = torch.rand(words.shape, device=words.device) < WORD_DROPOUT
            words = words.masked_fill(drop & (words > PAD), UNKNOWN)
        x = torch.cat(
            [
                self.word(words),
                self.suffix(sufs).view(b, n, -1),
                self.upos(upos),
                self.feats(feats.view(b * n, -1)).view(b, n, -1),
            ],
            -1,
        )
        packed = pack_padded_sequence(
            self.drop(x), lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=n)
        out = self.drop(out)
        dep = self.arc_dep(out)
        dep = torch.cat([dep, torch.ones_like(dep[..., :1])], -1)
        scores = dep @ self.arc @ self.arc_head(out).transpose(1, 2)
        mask = torch.arange(n, device=words.device)[None, :] < lens[:, None]
        eye = torch.eye(n, dtype=torch.bool, device=words.device)[None]
        scores = scores.masked_fill(~mask[:, None, :] | eye, float("-inf"))
        return scores, self.rel_dep(out), self.rel_head(out)

    def labels(
        self, rel_dep: torch.Tensor, rel_head: torch.Tensor, heads: torch.Tensor
    ) -> torch.Tensor:
        """Оценки связей по паре «зависимое - вершина» для заданных вершин: [b, n, связи]"""
        head = rel_head.gather(1, heads[..., None].expand(-1, -1, rel_head.shape[-1]))
        return self.label(torch.cat([rel_dep, head], -1))


def cycle(heads: list[int]) -> list[int] | None:
    """
    Узлы первого цикла в вершинах (0 - корень)

    Аргументы:
        heads (list[int]): Вершины позиций

    Вывод:
        list[int] | None: Цикл или None
    """
    color = [0] * len(heads)
    for start in range(1, len(heads)):
        path, v = [], start
        while v != 0 and color[v] == 0:
            color[v] = 1
            path.append(v)
            v = heads[v]
        if v != 0 and color[v] == 1:
            return path[path.index(v) :]
        for p in path:
            color[p] = 2
    return None


def edmonds(s: np.ndarray) -> list[int]:
    """
    Дерево максимального веса с корнем 0: s[d, h], строка 0 и диагональ - -inf

    Аргументы:
        s (np.ndarray): Оценки вершин

    Вывод:
        list[int]: Вершины позиций
    """
    n = len(s)
    heads = [0] + [int(np.argmax(s[d])) for d in range(1, n)]
    cyc = cycle(heads)
    if cyc is None:
        return heads
    in_cycle = np.zeros(n, bool)
    in_cycle[cyc] = True
    rest = np.where(~in_cycle)[0]
    cyc_a = np.array(cyc)
    m = len(rest)
    t = np.full((m + 1, m + 1), -np.inf)
    t[:m, :m] = s[np.ix_(rest, rest)]
    out_scores = s[np.ix_(rest, cyc_a)]
    best_out = cyc_a[out_scores.argmax(1)]
    t[:m, m] = out_scores.max(1)
    in_scores = s[np.ix_(cyc_a, rest)] - s[cyc_a, np.array(heads)[cyc_a]][:, None]
    best_in = cyc_a[in_scores.argmax(0)]
    t[m, :m] = in_scores.max(0)
    t[0, :] = -np.inf
    inner = edmonds(t)
    result = list(heads)
    for i in range(1, m):
        result[rest[i]] = int(best_out[i]) if inner[i] == m else int(rest[inner[i]])
    h = inner[m]
    result[int(best_in[h])] = int(rest[h])
    return result


def tree(scores: np.ndarray) -> list[int]:
    """
    Дерево с одним корнем по логарифмам вероятностей вершин (как в ядре на обычных предложениях)

    Аргументы:
        scores (np.ndarray): Логарифмы вероятностей s[d, h]

    Вывод:
        list[int]: Вершины позиций
    """
    s = scores.astype(np.float64)
    np.fill_diagonal(s, -np.inf)
    s[0, :] = -np.inf
    heads = edmonds(s)
    roots = [d for d in range(1, len(s)) if heads[d] == 0]
    if len(roots) == 1:
        return heads
    best, best_heads = -np.inf, heads
    for root in roots:
        t = s.copy()
        t[:, 0] = -np.inf
        t[root, 0] = s[root, 0]
        candidate = edmonds(t)
        total = sum(t[d, candidate[d]] for d in range(1, len(s)))
        if total > best:
            best, best_heads = total, candidate
    return best_heads


def predict(
    model: Parser, vocab: Vocab, sentences: list[Sentence], device: torch.device
) -> list[tuple[list[int], list[str]]]:
    """
    Вершины и связи предложений

    Аргументы:
        model (Parser): Модель
        vocab (Vocab): Словари
        sentences (list[Sentence]): Предложения
        device (torch.device): Устройство

    Вывод:
        list[tuple[list[int], list[str]]]: Вершины и связи каждого предложения
    """
    model.eval()
    out: list[tuple[list[int], list[str]]] = [([], [])] * len(sentences)
    with torch.no_grad():
        for idx in batches(sentences, 4000, None):
            words, sufs, upos, feats, _, _, lens = (
                t.to(device) for t in encode(vocab, [sentences[i] for i in idx])
            )
            scores, rel_dep, rel_head = model(words, sufs, upos, feats, lens)
            logp = torch.log_softmax(scores, -1).cpu().numpy()
            heads = torch.zeros(words.shape, dtype=torch.long)
            for k in range(len(idx)):
                n = int(lens[k])
                heads[k, :n] = torch.tensor(tree(logp[k, :n, :n]))
            labels = model.labels(rel_dep, rel_head, heads.to(device)).argmax(-1).cpu()
            for k, i in enumerate(idx):
                n = int(lens[k])
                out[i] = (
                    heads[k, 1:n].tolist(),
                    [vocab.relations[r] for r in labels[k, 1:n].tolist()],
                )
    return out


def las(sentences: list[Sentence], parses: list[tuple[list[int], list[str]]]) -> float:
    """
    LAS: доля слов с верными вершиной и типом связи без подтипа, как у скрипта CoNLL 2018

    Аргументы:
        sentences (list[Sentence]): Эталон
        parses (list[tuple[list[int], list[str]]]): Разбор

    Вывод:
        float: LAS, %
    """
    total = right = 0
    for s, (heads, rels) in zip(sentences, parses, strict=True):
        for gh, gr, ph, pr in zip(s[3], s[4], heads, rels, strict=True):
            total += 1
            right += gh == ph and gr.split(":")[0] == pr.split(":")[0]
    return 100 * right / total


def loss_fn(model: Parser, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    words, sufs, upos, feats, heads, rels, lens = batch
    scores, rel_dep, rel_head = model(words, sufs, upos, feats, lens)
    mask = torch.arange(words.shape[1], device=words.device)[None, :] < lens[:, None]
    mask[:, 0] = False
    arc = nn.functional.cross_entropy(scores[mask], heads[mask])
    rel = nn.functional.cross_entropy(model.labels(rel_dep, rel_head, heads)[mask], rels[mask])
    return arc + rel


def train(
    model: Parser,
    vocab: Vocab,
    sentences: list[Sentence],
    dev: dict[str, list[Sentence]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """
    Обучение с выбором точки по среднему LAS dev `SELECT`: лучшие веса остаются в модели

    Аргументы:
        model (Parser): Модель
        vocab (Vocab): Словари
        sentences (list[Sentence]): Обучающие предложения
        dev (dict[str, list[Sentence]]): Наборы dev
        args (argparse.Namespace): Аргументы запуска
        device (torch.device): Устройство

    Вывод:
        dict[str, float]: LAS dev лучшей точки
    """
    rng = random.Random(args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.75 ** (1 / 5000))
    per_epoch = len(batches(sentences, args.batch, None))
    every = min(per_epoch, 1000)
    best: tuple[float, dict[str, float], dict[str, torch.Tensor]] = (-1.0, {}, {})
    step = bad = 0
    start = time.time()
    while step < args.steps and bad < args.patience:
        model.train()
        for idx in batches(sentences, args.batch, rng):
            batch = tuple(t.to(device) for t in encode(vocab, [sentences[i] for i in idx]))
            loss = loss_fn(model, batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % every == 0 or step == args.steps:
                scores = {
                    name: las(s, predict(model, vocab, s, device)) for name, s in dev.items()
                }
                select = sum(scores[name] for name in SELECT) / len(SELECT)
                print(
                    f"шаг {step}, эпоха {step / per_epoch:.1f}, {(time.time() - start) / 60:.1f} мин: "
                    + ", ".join(f"{name} {value:.2f}" for name, value in scores.items()),
                    flush=True,
                )
                if select > best[0]:
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best, bad = (select, scores, state), 0
                else:
                    bad += 1
                model.train()
            if step >= args.steps or bad >= args.patience:
                break
    model.load_state_dict(best[2])
    return best[1]


def quantize_(model: Parser) -> None:
    """
    int8 по строкам, как в файле весов: матрицы заменяются на деквантованные значения

    Аргументы:
        model (Parser): Модель
    """
    with torch.no_grad():
        for _, tensor in matrices(model):
            scale = tensor.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127
            tensor.copy_((tensor / scale).round().clamp(-127, 127) * scale)


def matrices(model: Parser) -> list[tuple[str, torch.Tensor]]:
    """
    Матрицы модели под именами файла весов (они пишутся в int8); смещения - отдельно, в f32

    Аргументы:
        model (Parser): Модель

    Вывод:
        list[tuple[str, torch.Tensor]]: Имя записи и матрица
    """
    out = [
        ("embed.word", model.word.weight),
        ("embed.suffix", model.suffix.weight),
        ("embed.upos", model.upos.weight),
        ("embed.feats", model.feats.weight),
        ("arc.weight", model.arc),
        ("label.weight", model.label.weight),
    ]
    for layer in range(CONFIG["layers"]):
        for direction, suffix in (("forward", ""), ("backward", "_reverse")):
            out.append(
                (
                    f"lstm.{layer}.{direction}.ih",
                    getattr(model.lstm, f"weight_ih_l{layer}{suffix}"),
                )
            )
            out.append(
                (
                    f"lstm.{layer}.{direction}.hh",
                    getattr(model.lstm, f"weight_hh_l{layer}{suffix}"),
                )
            )
    out.extend(
        (f"{name}.weight", getattr(model, name).lin.weight)
        for name in ("arc_dep", "arc_head", "rel_dep", "rel_head")
    )
    return out


def write(model: Parser, vocab: Vocab, sources: dict, path: Path) -> None:
    """
    Файл весов ядра: сигнатура, версия формата, CRC32 и записи; матрицы - int8 по строкам

    Аргументы:
        model (Parser): Модель (матрицы уже квантованы `quantize_`)
        vocab (Vocab): Словари
        sources (dict): Описание обучения для записи `sources`
        path (Path): Файл весов
    """

    def strings(items: list[str]) -> bytes:
        out = bytearray([0]) + struct.pack("<I", len(items))
        for text in items:
            data = text.encode("utf-8")
            out += struct.pack("<I", len(data)) + data
        return bytes(out)

    def floats(tensor: torch.Tensor) -> bytes:
        t = tensor.detach().float().reshape(1, -1).cpu()
        return bytes([1]) + struct.pack("<II", *t.shape) + t.numpy().astype("<f4").tobytes()

    def int8(tensor: torch.Tensor) -> bytes:
        t = tensor.detach().float().cpu()
        scale = t.abs().amax(1).clamp(min=1e-12) / 127
        q = (t / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        return (
            bytes([2])
            + struct.pack("<II", *t.shape)
            + scale.numpy().astype("<f4").tobytes()
            + q.numpy().tobytes()
        )

    c = CONFIG
    entries = {
        "config": bytes([3]) + struct.pack("<I8I", 8, *(c[k] for k in CONFIG)),
        "sources": strings([json.dumps(sources, ensure_ascii=False)]),
        "vocab.words": strings(vocab.words),
        "vocab.suffixes": strings(vocab.suffixes),
        "vocab.upos": strings(vocab.upos),
        "vocab.feats": strings(vocab.feats),
        "relations": strings(vocab.relations),
        "label.bias": floats(model.label.bias),
    }
    entries.update({name: int8(tensor) for name, tensor in matrices(model)})
    for layer in range(c["layers"]):
        for direction, suffix in (("forward", ""), ("backward", "_reverse")):
            bias = getattr(model.lstm, f"bias_ih_l{layer}{suffix}") + getattr(
                model.lstm, f"bias_hh_l{layer}{suffix}"
            )
            entries[f"lstm.{layer}.{direction}.bias"] = floats(bias)
    for name in ("arc_dep", "arc_head", "rel_dep", "rel_head"):
        entries[f"{name}.bias"] = floats(getattr(model, name).lin.bias)
    body = struct.pack("<I", len(entries))
    for name, value in entries.items():
        data = name.encode("utf-8")
        body += struct.pack("<H", len(data)) + data + value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MAGIC + struct.pack("<II", FORMAT, zlib.crc32(body)) + body)


def write_check(
    sentences: list[Sentence], parses: list[tuple[list[int], list[str]]], path: Path
) -> None:
    """
    Файл сверки с ядром: формы, UPOS и признаки ruMorph, вершины и связи модели на PyTorch

    Аргументы:
        sentences (list[Sentence]): Предложения
        parses (list[tuple[list[int], list[str]]]): Разбор квантованной модели
        path (Path): Файл CoNLL-U
    """
    with path.open("w", encoding="utf-8") as file:
        for s, (heads, rels) in zip(sentences, parses, strict=True):
            for i, (form, upos, feats, head, rel) in enumerate(
                zip(s[0], s[1], s[2], heads, rels, strict=True), 1
            ):
                file.write(f"{i}\t{form}\t_\t{upos}\t_\t{feats}\t{head}\t{rel}\t_\t_\n")
            file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("data", type=Path, help="каталог scripts/export_ud.py")
    parser.add_argument("--output", type=Path, default=Path("data/ru-syntax.bin"))
    parser.add_argument("--check", type=Path, default=Path("data/check.conllu"))
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument(
        "--patience", type=int, default=4, help="оценок dev без роста до остановки"
    )
    parser.add_argument("--batch", type=int, default=5000, help="позиций в пакете")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    analyzer = rumorph.Analyzer()
    sentences = load([args.data / name for name in TRAIN], analyzer)
    dev = {name: load([args.data / file], analyzer) for name, file in DEV.items()}
    vocab = Vocab(sentences)
    model = Parser(vocab).to(device)
    print(
        f"train: {len(sentences)} предложений, {sum(len(s[0]) for s in sentences)} слов",
        flush=True,
    )
    best = train(model, vocab, sentences, dev, args, device)
    model = model.cpu()
    quantize_(model)
    quantized = {
        name: round(las(s, predict(model, vocab, s, torch.device("cpu"))), 2)
        for name, s in dev.items()
    }
    exported = json.loads((args.data / "sources.json").read_text(encoding="utf-8"))
    sources = {
        "ud": exported["release"],
        "license": exported["license"],
        "treebanks": exported["treebanks"],
        "train": TRAIN,
        "select": list(SELECT),
        "tags": f"rumorph {metadata.version('rumorph')}",
        "torch": metadata.version("torch"),
        "seed": args.seed,
        "dev_las": {name: round(value, 2) for name, value in best.items()},
        "dev_las_int8": quantized,
    }
    write(model, vocab, sources, args.output)
    check = [s for name in DEV for s in dev[name][:CHECK]]
    write_check(check, predict(model, vocab, check, torch.device("cpu")), args.check)
    print(
        f"{args.output}: {args.output.stat().st_size} байт; LAS dev int8: {quantized}", flush=True
    )


if __name__ == "__main__":
    main()
