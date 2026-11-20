"""
Проверка установленного пакета перед публикацией: разбор текста и команда rusyntax

Запуск: `python scripts/smoke.py [версия]`; из каталога `scripts` пакет берется установленный,
а не исходники из репозитория.
"""

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

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
# Команда: модулем и консольным скриптом рядом с интерпретатором
result = subprocess.run(
    [sys.executable, "-m", "rusyntax"],
    input="Мама мыла раму.\n",
    capture_output=True,
    encoding="utf-8",
    check=True,
)
assert "2\tмыла\tмыть\tVERB" in result.stdout and "\t0\troot\t" in result.stdout, result
script = shutil.which("rusyntax", path=str(Path(sys.executable).parent))
assert script is not None, Path(sys.executable).parent
result = subprocess.run([script, "-V"], capture_output=True, encoding="utf-8", check=True)
assert result.stdout.startswith(f"rusyntax {rusyntax.__version__} ("), result
print(f"rusyntax {rusyntax.__version__}")
