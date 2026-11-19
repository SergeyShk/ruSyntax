# Установка

Требуется Python 3.11 или новее. Вместе с пакетом устанавливаются сегментатор [anySeg](https://github.com/SergeyShk/anySeg) и морфологический анализатор [ruMorph](https://github.com/SergeyShk/ruMorph): они нужны для разбора текста.

``` bash
pip install rusyntax
```

## Готовые колеса

На PyPI публикуются колеса для Linux (x86_64 и aarch64, manylinux и musllinux), macOS (Intel и Apple Silicon) и Windows (x64 и arm64). Одно колесо подходит для всех версий Python начиная с 3.11, отдельные колеса собираются для Python 3.14t и 3.15t без GIL; Rust для установки не нужен. Python 3.13t без GIL не поддерживается.

Предварительные версии `0.1.0.devN` публикуются только на TestPyPI:

``` bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "rusyntax>=0.1.0.dev0"
```

## Сборка из исходников

Для других платформ pip соберет пакет из исходников; для этого нужен [Rust](https://www.rust-lang.org/tools/install) 1.88 или новее:

``` bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install rusyntax
```

Окружение для разработки описано в [CONTRIBUTING.md](https://github.com/SergeyShk/ruSyntax/blob/master/CONTRIBUTING.md).
