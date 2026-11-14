import importlib
import importlib.metadata
import re
import shutil
import subprocess
import sys
import sysconfig
import tomllib
import zipfile
from pathlib import Path

import pytest

import rusyntax

ROOT = Path(__file__).parents[1]


def test_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", rusyntax.__version__)
    assert rusyntax.__version__ != "0.0.0"


def test_metadata():
    assert "синтаксический анализатор" in rusyntax.__description__
    assert "__version__" in rusyntax.__all__
    assert rusyntax.__all__ == sorted(rusyntax.__all__)


def test_version_matches_metadata():
    # Версия берется из модуля расширения, без медленного importlib.metadata при импорте
    assert rusyntax.__version__ == importlib.metadata.version("rusyntax")


def test_crate_version():
    with (ROOT / "Cargo.toml").open("rb") as file:
        crate = tomllib.load(file)["workspace"]["package"]["version"]
    with (ROOT / "pyproject.toml").open("rb") as file:
        package = tomllib.load(file)["project"]["version"]
    assert crate == re.sub(r"\.dev(\d+)$", r"-dev.\1", package)


def test_extension():
    extension = importlib.import_module("rusyntax._rusyntax")
    assert Path(extension.__file__).parent == Path(rusyntax.__file__).parent


def test_package_data():
    assert (Path(rusyntax.__file__).parent / "py.typed").is_file()


@pytest.mark.skipif(shutil.which("uv") is None, reason="колесо собирает uv")
def test_wheel_contents(tmp_path):
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--python",
            sys.executable,
            "--out-dir",
            str(tmp_path),
            str(ROOT),
        ],
        capture_output=True,
        check=True,
    )
    (wheel,) = tmp_path.glob("*.whl")
    assert ("-abi3-" in wheel.name) != bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "rusyntax/py.typed" in names
    modules = [
        name for name in names if re.fullmatch(r"rusyntax/_rusyntax(\.[\w-]+)?\.(so|pyd)", name)
    ]
    assert len(modules) == 1
    assert all(name.startswith(("rusyntax/", "rusyntax-")) for name in names)
    licenses = {name.rsplit("/", 1)[1] for name in names if "/licenses/" in name}
    assert licenses == {"LICENSE.txt", "THIRD-PARTY-NOTICES.txt"}


@pytest.mark.skipif(shutil.which("cargo") is None, reason="крейты читает cargo")
def test_third_party_notices(tmp_path):
    # Лицензии крейтов модуля следуют Cargo.lock: после его изменения - python scripts/notices.py
    path = tmp_path / "notices.txt"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "notices.py"), str(path)],
        capture_output=True,
        check=True,
    )
    expected = (ROOT / "THIRD-PARTY-NOTICES.txt").read_text(encoding="utf-8")
    assert path.read_text(encoding="utf-8") == expected, "python scripts/notices.py"
