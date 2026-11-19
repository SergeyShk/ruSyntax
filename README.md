<h1 align="center">ruSyntax</h1>

<p align="center">
  Быстрый синтаксический анализатор русского языка на Rust с API для Python
</p>

<p align="center">
  <a href="https://sergeyshk.github.io/ruSyntax/">Документация</a> ·
  <a href="https://github.com/SergeyShk/ruSyntax/blob/master/README.en.md">English</a>
</p>

<p align="center">
  <a href="https://github.com/SergeyShk/ruSyntax/actions/workflows/ci.yml"><img src="https://github.com/SergeyShk/ruSyntax/actions/workflows/ci.yml/badge.svg" alt="Сборка"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/SergeyShk/ruSyntax/blob/master/LICENSE.txt"><img src="https://img.shields.io/github/license/sergeyshk/rusyntax.svg" alt="Лицензия"></a>
</p>

---

**ruSyntax** - синтаксический анализатор русского языка: ядро на Rust и API для Python. Для каждого слова предложения анализатор находит вершину и тип синтаксической связи в разметке [Universal Dependencies](https://universaldependencies.org/).

> [!WARNING]
> Проект в разработке: пакет пока содержит только каркас, анализатор появится в следующих версиях. Версии `0.1.0.devN` публикуются на TestPyPI.

## Установка

Требуется Python 3.11 или новее.

```bash
pip install rusyntax
```

Колеса собираются для Linux, macOS и Windows; на других платформах для сборки из исходников нужен [Rust](https://www.rust-lang.org/tools/install). Подробнее - на странице [Установка](https://sergeyshk.github.io/ruSyntax/installation/).

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
*   **scripts** - выгрузка трибанков UD и обучение модели, проверка установленного пакета и лицензии крейтов модуля

</details>

## Авторы

*   Шкарин Сергей (kouki.sergey@gmail.com)

## Лицензия

Код распространяется под лицензией [MIT](https://github.com/SergeyShk/ruSyntax/blob/master/LICENSE.txt). Модель обучена на трибанках [UD_Russian-Taiga](https://github.com/UniversalDependencies/UD_Russian-Taiga), [UD_Russian-GSD](https://github.com/UniversalDependencies/UD_Russian-GSD) и [UD_Russian-Poetry](https://github.com/UniversalDependencies/UD_Russian-Poetry) и распространяется по лицензии CC BY-SA 4.0 ([LICENSE-MODEL.txt](https://github.com/SergeyShk/ruSyntax/blob/master/LICENSE-MODEL.txt)).
