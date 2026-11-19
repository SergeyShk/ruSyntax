# Installation

Python 3.11 or newer is required. The [anySeg](https://github.com/SergeyShk/anySeg) segmenter and the [ruMorph](https://github.com/SergeyShk/ruMorph) morphological analyzer are installed together with the package: parsing a text needs them.

``` bash
pip install rusyntax
```

## Prebuilt wheels

PyPI hosts wheels for Linux (x86_64 and aarch64, manylinux and musllinux), macOS (Intel and Apple Silicon) and Windows (x64 and arm64). One wheel serves every Python version starting with 3.11, and separate wheels are built for the free-threaded Python 3.14t and 3.15t; Rust is not needed to install them. The free-threaded Python 3.13t is not supported.

Pre-release versions `0.1.0.devN` are published only on TestPyPI:

``` bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "rusyntax>=0.1.0.dev0"
```

## Building from source

On other platforms pip builds the package from source, which requires [Rust](https://www.rust-lang.org/tools/install) 1.88 or newer:

``` bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install rusyntax
```

The development environment is described in [CONTRIBUTING.md](https://github.com/SergeyShk/ruSyntax/blob/master/CONTRIBUTING.md).
