# ruSyntax

**ruSyntax** is a dependency parser of Russian: a Rust core with a Python API. For every word of a sentence the parser finds its head and the relation between them in the [Universal Dependencies](https://universaldependencies.org/) annotation.

!!! warning "Work in progress"
    So far the package contains only the scaffolding; the analyzer comes in the next versions. Versions `0.1.0.devN` are published on TestPyPI.

## Installation

Python 3.11 or newer is required.

``` bash
pip install rusyntax
```

Prebuilt wheels and building from source are described on the [Installation](installation.md) page.
