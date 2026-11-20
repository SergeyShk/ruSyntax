import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, SupportsIndex, final, overload

from anyseg import Document

__all__ = ["Parser", "__version__"]

__version__: str

# Признаки слова: запись CoNLL-U, словарь «признак - значение» (как feats у ruMorph) или None
_Feats = str | Mapping[str, str] | None

class _Word(Protocol):
    @property
    def index(self) -> SupportsIndex: ...

class _Sentence(Protocol):
    @property
    def words(self) -> Iterable[_Word]: ...

class _Document(Protocol):
    # Документ anyseg.Document или объект с тем же интерфейсом
    def column(self, name: str, /) -> Iterable[object]: ...
    @property
    def sentences(self) -> Iterable[_Sentence]: ...
    def set_columns(self, *, head: list[int], deprel: list[str]) -> object: ...

class _Segmenter(Protocol):
    # Сегментатор anyseg.Segmenter или объект с тем же segment
    def segment(self, text: str, /) -> Document: ...

class _Analyzer(Protocol):
    # Анализатор rumorph.Analyzer или объект с тем же annotate
    def annotate(self, document: Document, /) -> object: ...

@final
class Parser:
    def __new__(cls, path: str | os.PathLike[str] | None = None) -> Parser: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __copy__(self) -> Parser: ...
    def __deepcopy__(self, memo: Any, /) -> Parser: ...
    def __reduce__(self) -> tuple[type[Parser], tuple[Path | None]]: ...
    @property
    def path(self) -> Path | None: ...
    @property
    def relations(self) -> list[str]: ...
    @property
    def sources(self) -> dict[str, Any]: ...
    @overload
    def parse(
        self,
        words: Iterable[str],
        upos: Iterable[str],
        feats: Iterable[_Feats],
        *,
        probs: Literal[False] = False,
    ) -> list[tuple[int, str]]: ...
    @overload
    def parse(
        self,
        words: Iterable[str],
        upos: Iterable[str],
        feats: Iterable[_Feats],
        *,
        probs: Literal[True],
    ) -> list[tuple[int, str, float, float]]: ...
    @overload
    def parse(
        self,
        words: Iterable[str],
        upos: Iterable[str],
        feats: Iterable[_Feats],
        *,
        probs: bool,
    ) -> list[tuple[int, str]] | list[tuple[int, str, float, float]]: ...
    @overload
    def parse_many(
        self,
        sentences: Iterable[tuple[Iterable[str], Iterable[str], Iterable[_Feats]]],
        *,
        probs: Literal[False] = False,
    ) -> list[list[tuple[int, str]]]: ...
    @overload
    def parse_many(
        self,
        sentences: Iterable[tuple[Iterable[str], Iterable[str], Iterable[_Feats]]],
        *,
        probs: Literal[True],
    ) -> list[list[tuple[int, str, float, float]]]: ...
    @overload
    def parse_many(
        self,
        sentences: Iterable[tuple[Iterable[str], Iterable[str], Iterable[_Feats]]],
        *,
        probs: bool,
    ) -> list[list[tuple[int, str]]] | list[list[tuple[int, str, float, float]]]: ...
    def annotate(self, document: _Document) -> None: ...
    def parse_text(
        self, text: str, *, segmenter: _Segmenter | None = None, analyzer: _Analyzer | None = None
    ) -> Document: ...
