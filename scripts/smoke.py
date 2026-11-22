"""
Проверка установленного пакета перед публикацией: разбор текста, теги ruMorph и команда rusyntax

Запуск: `python scripts/smoke.py [версия]`; из каталога `scripts` пакет берется установленный,
а не исходники из репозитория.
"""

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import rumorph

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
# Теги установленного ruMorph - те, на которых обучена модель: в data/check.conllu - теги обучения.
# Пакет принимает новые 0.x ruMorph, а тест репозитория видит только закрепленный в uv.lock
sentences: list[list[list[str]]] = [[]]
check = Path(__file__).parents[1] / "data" / "check.conllu"
for line in check.read_text(encoding="utf-8").splitlines():
    if line:
        sentences[-1].append(line.split("\t"))
    elif sentences[-1]:
        sentences.append([])
analyzer = rumorph.Analyzer()
total = upos = feats = 0
for sentence in filter(None, sentences):
    best = [a[0] for a in analyzer.analyze_sentence([word[1] for word in sentence])]
    for word, analysis in zip(sentence, best, strict=True):
        pairs = sorted(analysis.feats.items(), key=lambda pair: pair[0].lower())
        total += 1
        upos += word[3] != analysis.pos
        feats += word[5] != ("|".join(f"{name}={value}" for name, value in pairs) or "_")
assert total > 4000 and max(upos, feats) <= total * 0.005, (
    rumorph.__version__,
    total,
    upos,
    feats,
)
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
