use std::fmt;

/// Ошибка загрузки модели
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Ошибка ввода-вывода при чтении файла модели
    Io(std::io::Error),
    /// Данные не являются моделью ruSyntax или повреждены
    Format(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(err) => write!(f, "{err}"),
            Error::Format(msg) => write!(f, "неверный формат модели: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            Error::Format(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}
