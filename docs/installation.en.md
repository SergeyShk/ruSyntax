# Installation

Python 3.11 or newer is required. The [anySeg](https://github.com/SergeyShk/anySeg) segmenter and the [ruMorph](https://github.com/SergeyShk/ruMorph) morphological analyzer are installed together with the package: parsing a text needs them.

``` bash
pip install rusyntax
```

## Prebuilt wheels

PyPI hosts wheels for Linux (x86_64 and aarch64, manylinux and musllinux), macOS (Intel and Apple Silicon) and Windows (x64 and arm64). One wheel serves every Python version starting with 3.11, and separate wheels are built for the free-threaded Python 3.14t and 3.15t; Rust is not needed to install them. The free-threaded Python 3.13t is not supported.

Pre-release versions `X.Y.Z.devN` are published only on TestPyPI:

``` bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --pre rusyntax
```

uv takes a package from the first index that has it, so it needs another order of indexes:

``` bash
uv pip install --extra-index-url https://test.pypi.org/simple/ --index-strategy unsafe-best-match --pre rusyntax
```

## Building from source

On other platforms pip builds the package from source, which requires [Rust](https://www.rust-lang.org/tools/install) 1.88 or newer:

``` bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install rusyntax
```

The development environment is described in [CONTRIBUTING.md](https://github.com/SergeyShk/ruSyntax/blob/master/CONTRIBUTING.md).

## Rust { #rust }

The parser core is published on crates.io as the [`rusyntax`](https://crates.io/crates/rusyntax) crate, and the API documentation is on [docs.rs](https://docs.rs/rusyntax). The model is not part of the crate: download the [`data/ru-syntax.bin`](https://github.com/SergeyShk/ruSyntax/raw/0.2.0/data/ru-syntax.bin) file of the repository at the tag of the crate version (the model license is CC BY-SA 4.0) and pass its path to `Parser::from_path`.

``` bash
cargo add rusyntax
```
