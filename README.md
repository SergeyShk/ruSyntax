<p align="center">
  <img src="https://raw.githubusercontent.com/SergeyShk/ruSyntax/master/docs/img/rusyntax.png" alt="ruSyntax" width="480">
</p>

<h1 align="center">ruSyntax</h1>

<p align="center">
  Быстрый синтаксический анализатор русского языка на Rust с API для Python
</p>

<p align="center">
  <a href="https://sergeyshk.github.io/ruSyntax/">Документация</a> ·
  <a href="https://pypi.org/project/rusyntax/">PyPI</a> ·
  <a href="https://crates.io/crates/rusyntax">crates.io</a> ·
  <a href="https://github.com/SergeyShk/ruSyntax/blob/master/README.en.md">English</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/rusyntax/"><img src="https://img.shields.io/pypi/v/rusyntax?logo=pypi&logoColor=FFE873" alt="Версия"></a>
  <a href="https://pypi.org/project/rusyntax/"><img src="https://img.shields.io/pypi/pyversions/rusyntax.svg?logo=python&logoColor=FFE873" alt="Поддерживаемые версии Python"></a>
  <a href="https://crates.io/crates/rusyntax"><img src="https://img.shields.io/crates/v/rusyntax?logo=rust" alt="Крейт"></a>
  <a href="https://github.com/SergeyShk/ruSyntax/actions/workflows/ci.yml"><img src="https://github.com/SergeyShk/ruSyntax/actions/workflows/ci.yml/badge.svg" alt="Сборка"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/SergeyShk/ruSyntax/blob/master/LICENSE.txt"><img src="https://img.shields.io/github/license/sergeyshk/rusyntax.svg" alt="Лицензия"></a>
</p>

---

**ruSyntax** - синтаксический анализатор русского языка: ядро на Rust и API для Python. Для каждого слова предложения анализатор находит вершину и тип синтаксической связи в разметке [Universal Dependencies](https://universaldependencies.org/).

## Установка

Требуется Python 3.11 или новее.

```bash
pip install rusyntax
```

Колеса собираются для Linux, macOS и Windows; на других платформах для сборки из исходников нужен [Rust](https://www.rust-lang.org/tools/install). Подробнее - на странице [Установка](https://sergeyshk.github.io/ruSyntax/installation/).

## Использование

```python
from rusyntax import Parser

parser = Parser()
document = parser.parse_text("Мама мыла раму.")
print([(word.form, word.head, word.deprel) for word in document.words])
# [('Мама', 2, 'nsubj'), ('мыла', 0, 'root'), ('раму', 2, 'obj'), ('.', 2, 'punct')]
```

`parse_text` делит текст на предложения и слова сегментатором [anySeg](https://github.com/SergeyShk/anySeg), ставит словам часть речи и признаки морфологическим анализатором [ruMorph](https://github.com/SergeyShk/ruMorph) и находит вершину (номер слова с 1, 0 - корень) и тип связи каждого слова. Документ anySeg, уже разобранный морфологически, разбирает `parser.annotate(document)`, предложение со своими частями речи и признаками - `parser.parse(words, upos, feats)`, много предложений сразу - `parser.parse_many(sentences)`.

## Качество и скорость

LAS от сырого текста на тестовых частях трибанков UD 2.19 (скрипт CoNLL 2018), %:

| Система | SynTagRus | Taiga | GSD | PUD | Poetry | Слов/с |
| :------ | --------: | ----: | --: | --: | -----: | -----: |
| ruSyntax | 87,8 | **81,5** | **84,9** | 87,8 | **83,3** | 29 526 |
| spaCy ru_core_news_sm | 81,8 | 71,5 | 75,8 | 86,2 | 65,1 | 10 368 |
| Stanza | 91,2 | 77,6 | 78,0 | 88,2 | 73,4 | 1 098 |
| Stanza accurate | **93,1** | 80,5 | 81,2 | **89,5** | 76,4 | 428 |

Модели Stanza обучены на SynTagRus, модель ruSyntax - на Taiga, GSD и Poetry. Скорость - один поток, вместе с морфологией; остальные системы, метрики и условия замера - на странице [Сравнение](https://sergeyshk.github.io/ruSyntax/performance/).

## Разработка

Проект использует [uv](https://docs.astral.sh/uv/) для управления зависимостями, [maturin](https://www.maturin.rs/) для сборки модуля на Rust и [ruff](https://docs.astral.sh/ruff/) для линтинга и форматирования. Нужен [Rust](https://www.rust-lang.org/tools/install) 1.88 или новее.

```bash
git clone https://github.com/SergeyShk/ruSyntax.git
cd ruSyntax

make deps        # создать окружение, установить зависимости и собрать модуль на Rust
make test        # запустить тесты Python, примеры из докстрингов (doctest) и тесты Rust
make lint        # ruff, mypy, rustfmt и clippy
```

Полный список команд - `make help`. Документация двуязычная: русские страницы `docs/*.md`, английские - `docs/*.en.md` рядом с ними; при правке страницы обновляйте обе версии.

Перед отправкой изменений стоит установить хуки, которые прогонят линтеры на коммите и тесты на пуше:

```bash
uv run pre-commit install
```

## Участие в проекте

Баг-репорты, идеи и пул-реквесты приветствуются - [issues](https://github.com/SergeyShk/ruSyntax/issues) открыты. Как устроен процесс и что проверить перед отправкой пул-реквеста - в [CONTRIBUTING.md](https://github.com/SergeyShk/ruSyntax/blob/master/CONTRIBUTING.md); правила общения - в [кодексе поведения](https://github.com/SergeyShk/ruSyntax/blob/master/CODE_OF_CONDUCT.md).

<details>
<summary><b>Структура проекта</b></summary>

<br>

*   **crates/rusyntax** - ядро анализатора на Rust
*   **crates/rusyntax-py** - модуль Python `rusyntax._rusyntax` на PyO3
*   **python/rusyntax** - пакет Python
*   **docs** - документация MkDocs на русском и английском
*   **tests** - тесты пакета Python
*   **data** - модель и файл сверки ее прямого прохода на Rust с PyTorch
*   **scripts** - выгрузка трибанков UD и обучение модели, сравнение с другими анализаторами, картинки документации, проверка установленного пакета и лицензии крейтов модуля

</details>

## Авторы

*   Шкарин Сергей (kouki.sergey@gmail.com)

## Лицензия

Код распространяется под лицензией [MIT](https://github.com/SergeyShk/ruSyntax/blob/master/LICENSE.txt). Модель обучена на трибанках [UD_Russian-Taiga](https://github.com/UniversalDependencies/UD_Russian-Taiga), [UD_Russian-GSD](https://github.com/UniversalDependencies/UD_Russian-GSD) и [UD_Russian-Poetry](https://github.com/UniversalDependencies/UD_Russian-Poetry) и распространяется по лицензии CC BY-SA 4.0 ([LICENSE-MODEL.txt](https://github.com/SergeyShk/ruSyntax/blob/master/LICENSE-MODEL.txt)).
