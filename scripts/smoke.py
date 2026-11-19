"""
Проверка установленного пакета перед публикацией

Запуск: `python scripts/smoke.py [версия]`; из каталога `scripts` пакет берется установленный,
а не исходники из репозитория.
"""

import sys
import sysconfig

import rusyntax

assert "site-packages" in rusyntax.__file__, rusyntax.__file__
# В Python без GIL импорт модуля не должен включать GIL обратно
if sysconfig.get_config_var("Py_GIL_DISABLED"):
    assert not sys._is_gil_enabled()
if len(sys.argv) > 1:
    assert rusyntax.__version__ == sys.argv[1], (rusyntax.__version__, sys.argv[1])
# Модель пакета и путь от текста через установленные anyseg и rumorph
document = rusyntax.Parser().parse_text("Мама мыла раму.")
parse = [(word.head, word.deprel) for word in document.words]
assert parse == [(2, "nsubj"), (0, "root"), (2, "obj"), (2, "punct")], parse
if sysconfig.get_config_var("Py_GIL_DISABLED"):
    assert not sys._is_gil_enabled()
print(f"rusyntax {rusyntax.__version__}")
