# Установка

Требуется Python 3.11 или новее. Вместе с пакетом устанавливаются сегментатор [anySeg](https://github.com/SergeyShk/anySeg) и морфологический анализатор [ruMorph](https://github.com/SergeyShk/ruMorph): они нужны для разбора текста.

``` bash
pip install rusyntax
```

## Готовые колеса

На PyPI публикуются колеса для Linux (x86_64 и aarch64, manylinux и musllinux), macOS (Intel и Apple Silicon) и Windows (x64 и arm64). Одно колесо подходит для всех версий Python начиная с 3.11, отдельные колеса собираются для Python 3.14t и 3.15t без GIL; Rust для установки не нужен. Python 3.13t без GIL не поддерживается.

Предварительные версии `X.Y.Z.devN` публикуются только на TestPyPI:

``` bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --pre rusyntax
```

uv берет пакет из первого индекса, где он есть, поэтому ему нужен другой порядок индексов:

``` bash
uv pip install --extra-index-url https://test.pypi.org/simple/ --index-strategy unsafe-best-match --pre rusyntax
```

## Сборка из исходников

Для других платформ pip соберет пакет из исходников; для этого нужен [Rust](https://www.rust-lang.org/tools/install) 1.88 или новее:

``` bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install rusyntax
```

Окружение для разработки описано в [CONTRIBUTING.md](https://github.com/SergeyShk/ruSyntax/blob/master/CONTRIBUTING.md).

## Rust { #rust }

Ядро анализатора публикуется на crates.io крейтом [`rusyntax`](https://crates.io/crates/rusyntax), документация API - на [docs.rs](https://docs.rs/rusyntax). Модель в крейт не входит: файл [`data/ru-syntax.bin`](https://github.com/SergeyShk/ruSyntax/raw/0.2.0/data/ru-syntax.bin) репозитория на теге той же версии, что и крейт (лицензия модели - CC BY-SA 4.0), нужно скачать и передать путь к нему в `Parser::from_path`.

``` bash
cargo add rusyntax
```
