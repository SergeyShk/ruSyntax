<h1 align="center">ruSyntax</h1>

<p align="center">
  A fast Russian dependency parser written in Rust with a Python API
</p>

<p align="center">
  <a href="https://sergeyshk.github.io/ruSyntax/en/">Documentation</a> ·
  <a href="https://github.com/SergeyShk/ruSyntax/blob/master/README.md">Русский</a>
</p>

<p align="center">
  <a href="https://github.com/SergeyShk/ruSyntax/actions/workflows/ci.yml"><img src="https://github.com/SergeyShk/ruSyntax/actions/workflows/ci.yml/badge.svg" alt="Build"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/SergeyShk/ruSyntax/blob/master/LICENSE.txt"><img src="https://img.shields.io/github/license/sergeyshk/rusyntax.svg" alt="License"></a>
</p>

---

**ruSyntax** is a dependency parser of Russian: a Rust core with a Python API. For every word of a sentence the parser finds its head and the relation between them in the [Universal Dependencies](https://universaldependencies.org/) annotation.

> [!WARNING]
> Work in progress: the API may change until version 0.1.0. Versions `0.1.0.devN` are published on TestPyPI.

## Installation

Python 3.11 or newer is required.

```bash
pip install rusyntax
```

Wheels are built for Linux, macOS and Windows; on other platforms building from source requires [Rust](https://www.rust-lang.org/tools/install). See the [Installation](https://sergeyshk.github.io/ruSyntax/en/installation/) page.

## Usage

```python
from rusyntax import Parser

parser = Parser()
document = parser.parse_text("Мама мыла раму.")
print([(word.form, word.head, word.deprel) for word in document.words])
# [('Мама', 2, 'nsubj'), ('мыла', 0, 'root'), ('раму', 2, 'obj'), ('.', 2, 'punct')]
```

`parse_text` splits the text into sentences and words with the [anySeg](https://github.com/SergeyShk/anySeg) segmenter, tags the words with the part of speech and features of the [ruMorph](https://github.com/SergeyShk/ruMorph) morphological analyzer and finds the head (the word number from 1, 0 for the root) and the relation of every word. `parser.annotate(document)` parses an anySeg document that already has its morphology, `parser.parse(words, upos, feats)` parses a sentence with your own parts of speech and features, and `parser.parse_many(sentences)` parses many sentences at once.

## Development

The project uses [uv](https://docs.astral.sh/uv/) to manage dependencies, [maturin](https://www.maturin.rs/) to build the Rust module and [ruff](https://docs.astral.sh/ruff/) for linting and formatting. [Rust](https://www.rust-lang.org/tools/install) 1.88 or newer is required.

```bash
git clone https://github.com/SergeyShk/ruSyntax.git
cd ruSyntax

make deps        # create the environment, install the dependencies and build the Rust module
make test        # run the Python tests, the docstring examples (doctest) and the Rust tests
make lint        # ruff, mypy, rustfmt and clippy
```

The full list of commands is `make help`. The documentation is bilingual: Russian pages `docs/*.md` with English `docs/*.en.md` next to them; update both versions when editing a page.

Before submitting changes, install the hooks that run the linters on commit and the tests on push:

```bash
uv run pre-commit install
```

## Contributing

Bug reports, ideas and pull requests are welcome - the [issues](https://github.com/SergeyShk/ruSyntax/issues) are open. The workflow and the checks to run before submitting a pull request are in [CONTRIBUTING.md](https://github.com/SergeyShk/ruSyntax/blob/master/CONTRIBUTING.md); the rules of conduct are in the [code of conduct](https://github.com/SergeyShk/ruSyntax/blob/master/CODE_OF_CONDUCT.md).

<details>
<summary><b>Project structure</b></summary>

<br>

*   **crates/rusyntax** - the analyzer core in Rust
*   **crates/rusyntax-py** - the Python module `rusyntax._rusyntax` built with PyO3
*   **python/rusyntax** - the Python package
*   **docs** - MkDocs documentation in Russian and English
*   **tests** - tests of the Python package
*   **data** - the model and the check of its Rust forward pass against PyTorch
*   **scripts** - the export of the UD treebanks and the training of the model, the check of the installed package and the licenses of the crates of the module

</details>

## Authors

*   Sergey Shkarin (kouki.sergey@gmail.com)

## License

The code is distributed under the [MIT](https://github.com/SergeyShk/ruSyntax/blob/master/LICENSE.txt) license. The model is trained on the treebanks [UD_Russian-Taiga](https://github.com/UniversalDependencies/UD_Russian-Taiga), [UD_Russian-GSD](https://github.com/UniversalDependencies/UD_Russian-GSD) and [UD_Russian-Poetry](https://github.com/UniversalDependencies/UD_Russian-Poetry) and is distributed under the CC BY-SA 4.0 license ([LICENSE-MODEL.txt](https://github.com/SergeyShk/ruSyntax/blob/master/LICENSE-MODEL.txt)).
