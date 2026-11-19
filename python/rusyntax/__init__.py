# Синтаксический анализатор русского языка (ruSyntax)
#
# Copyright (C) 2026
# Автор: Шкарин Сергей <kouki.sergey@gmail.com>
# URL: <https://github.com/SergeyShk/ruSyntax>

from ._rusyntax import Parser, __version__

__description__ = """Быстрый синтаксический анализатор русского языка на Rust с API для Python.
Требует версию Python 3.11 и выше"""
__author__ = "Шкарин Сергей"
__author_email__ = "kouki.sergey@gmail.com"

__all__ = ["Parser", "__version__"]
