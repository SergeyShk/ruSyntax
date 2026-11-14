//! Модуль Python `rusyntax._rusyntax`

use pyo3::prelude::*;

/// Версия крейта в записи PEP 440: `0.1.0-dev.12` - это `0.1.0.dev12`
fn package_version() -> String {
    env!("CARGO_PKG_VERSION").replace("-dev.", ".dev")
}

#[pymodule(gil_used = false)]
fn _rusyntax(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", package_version())?;
    Ok(())
}
