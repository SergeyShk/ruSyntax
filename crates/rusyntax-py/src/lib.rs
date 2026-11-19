//! Модуль Python `rusyntax._rusyntax`

use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use pyo3::exceptions::{PyOSError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};
use pyo3::{IntoPyObjectExt, intern};

/// Модель пакета: веса `data/ru-syntax.bin` (CC BY-SA 4.0, LICENSE-MODEL.txt) вшиты в модуль
static MODEL: &[u8] = include_bytes!("../../../data/ru-syntax.bin");
/// Предложений, которые разбираются без GIL между проверками Ctrl+C
const CHUNK: usize = 256;

/// Версия крейта в записи PEP 440: `0.1.0-dev.12` - это `0.1.0.dev12`
fn package_version() -> String {
    env!("CARGO_PKG_VERSION").replace("-dev.", ".dev")
}

/// Модель пакета загружается один раз на процесс и делится всеми анализаторами
fn bundled() -> PyResult<Arc<rusyntax::Parser>> {
    static PARSER: OnceLock<Arc<rusyntax::Parser>> = OnceLock::new();
    if let Some(parser) = PARSER.get() {
        return Ok(parser.clone());
    }
    let parser = rusyntax::Parser::from_bytes(MODEL)
        .map_err(|err| PyValueError::new_err(format!("модель пакета: {err}")))?;
    let parser = Arc::new(parser);
    Ok(PARSER.get_or_init(|| parser).clone())
}

/// Ошибка загрузки модели с путем к файлу: OSError с errno и именем файла, как у open
fn load_error(path: &Path, err: rusyntax::Error) -> PyErr {
    let rusyntax::Error::Io(err) = err else {
        return PyValueError::new_err(format!("{}: {err}", path.display()));
    };
    let message = match err.kind() {
        ErrorKind::NotFound => "Нет файла модели".to_owned(),
        ErrorKind::PermissionDenied => "Нет доступа к файлу модели".to_owned(),
        ErrorKind::IsADirectory => "Путь к модели указывает на каталог".to_owned(),
        _ => err.to_string(),
    };
    let filename = path.display().to_string();
    match err.raw_os_error() {
        #[cfg(windows)]
        Some(code) => PyOSError::new_err((0, message, filename, code)),
        #[cfg(not(windows))]
        Some(errno) => PyOSError::new_err((errno, message, filename)),
        None => PyOSError::new_err(format!("{message}: {filename}")),
    }
}

/// Строка Rust из строки Python; непарные суррогаты заменяются на U+FFFD
fn lossy(text: &Bound<'_, PyString>) -> PyResult<String> {
    if let Ok(text) = text.to_str() {
        return Ok(text.to_owned());
    }
    let encoded = text.call_method1("encode", ("utf-16-le", "surrogatepass"))?;
    let bytes = encoded.extract::<Vec<u8>>()?;
    let units: Vec<u16> = (bytes.as_chunks::<2>().0.iter())
        .map(|&pair| u16::from_le_bytes(pair))
        .collect();
    Ok(String::from_utf16_lossy(&units))
}

/// TypeError с типом полученного значения: «Слова должны быть набором, а не int»
fn type_error(message: &str, value: &Bound<'_, PyAny>) -> PyErr {
    let name = value
        .get_type()
        .name()
        .map_or_else(|_| "?".into(), |name| name.to_string());
    PyTypeError::new_err(format!("{message}, а не {name}"))
}

/// Строки набора; `none` - чем заменить `None` (если `None` допустим)
fn strings(value: &Bound<'_, PyAny>, what: &str, none: Option<&str>) -> PyResult<Vec<String>> {
    if value.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(format!(
            "{what}: ожидается набор строк, а не строка - передайте список"
        )));
    }
    let items = value
        .try_iter()
        .map_err(|_| type_error(&format!("{what}: ожидается набор строк"), value))?;
    items
        .enumerate()
        .map(|(number, item)| {
            let item = item?;
            if item.is_none()
                && let Some(none) = none
            {
                return Ok(none.to_owned());
            }
            let item = item.cast::<PyString>().map_err(|_| {
                type_error(
                    &format!("{what}: элемент {number} должен быть строкой"),
                    &item,
                )
            })?;
            lossy(item)
        })
        .collect()
}

/// Слова одного предложения: формы, части речи и признаки одной длины
type Sentence = (Vec<String>, Vec<String>, Vec<String>);

fn sentence(
    words: &Bound<'_, PyAny>,
    upos: &Bound<'_, PyAny>,
    feats: &Bound<'_, PyAny>,
) -> PyResult<Sentence> {
    let words = strings(words, "words", None)?;
    let upos = strings(upos, "upos", None)?;
    let feats = strings(feats, "feats", Some("_"))?;
    if upos.len() != words.len() || feats.len() != words.len() {
        return Err(PyValueError::new_err(format!(
            "У каждого слова должны быть часть речи и признаки: слов {}, частей речи {}, признаков {}",
            words.len(),
            upos.len(),
            feats.len()
        )));
    }
    Ok((words, upos, feats))
}

/// Разбор предложений без GIL кусками по `CHUNK`: между ними срабатывает Ctrl+C
fn parse_all(
    py: Python<'_>,
    parser: &rusyntax::Parser,
    sentences: &[Sentence],
) -> PyResult<Vec<Vec<(usize, String)>>> {
    let mut out = Vec::with_capacity(sentences.len());
    for chunk in sentences.chunks(CHUNK) {
        let parsed = py.detach(|| {
            let words: Vec<Vec<rusyntax::Word<'_>>> = chunk
                .iter()
                .map(|(forms, upos, feats)| {
                    (forms.iter().zip(upos).zip(feats))
                        .map(|((form, upos), feats)| rusyntax::Word::new(form, upos, feats))
                        .collect()
                })
                .collect();
            (parser.parse_many(&words).into_iter())
                .map(|relations| {
                    (relations.into_iter())
                        .map(|relation| (relation.head, relation.deprel.to_owned()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });
        out.extend(parsed);
        py.check_signals()?;
    }
    Ok(out)
}

/// Синтаксический анализатор русского языка
///
/// Описание:
///     Для каждого слова предложения находит вершину (номер слова с 1, 0 - корень) и тип связи
///     Universal Dependencies. На входе - слова с частью речи и признаками UD (как их ставит
///     ruMorph); путь от текста - parse_text (anySeg и ruMorph). Без path - модель пакета,
///     обученная на трибанках UD Taiga, GSD и Poetry (CC BY-SA 4.0). Анализатор не меняется
///     при разборе: его можно разделять между потоками
///
/// Аргументы:
///     path (str | os.PathLike | None): Файл модели ruSyntax; None - модель пакета
///
/// Исключения:
///     OSError: Если файл модели не прочитать
///     ValueError: Если файл - не модель ruSyntax, другой версии формата или поврежден
///
/// Пример использования:
///     >>> from rusyntax import Parser
///     >>> parser = Parser()
///     >>> parser.parse(["Мама", "мыла", "раму"], ["NOUN", "VERB", "NOUN"],
///     ...              ["Case=Nom|Gender=Fem|Number=Sing", "Gender=Fem|Tense=Past", "Case=Acc"])
///     [(2, 'nsubj'), (0, 'root'), (2, 'obj')]
#[pyclass(frozen, module = "rusyntax")]
struct Parser {
    parser: Arc<rusyntax::Parser>,
    path: Option<PathBuf>,
    /// Сегментатор anySeg и анализатор ruMorph для parse_text: создаются при первом вызове
    tools: OnceLock<(Py<PyAny>, Py<PyAny>)>,
}

impl Parser {
    fn tools<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        if let Some((segmenter, analyzer)) = self.tools.get() {
            return Ok((segmenter.bind(py).clone(), analyzer.bind(py).clone()));
        }
        let segmenter = py.import("anyseg")?.getattr("Segmenter")?.call1(("ru",))?;
        let analyzer = py.import("rumorph")?.getattr("Analyzer")?.call0()?;
        let (segmenter, analyzer) = self
            .tools
            .get_or_init(|| (segmenter.unbind(), analyzer.unbind()));
        Ok((segmenter.bind(py).clone(), analyzer.bind(py).clone()))
    }
}

#[pymethods]
impl Parser {
    #[new]
    #[pyo3(signature = (path = None))]
    fn new(py: Python<'_>, path: Option<PathBuf>) -> PyResult<Self> {
        let parser = match &path {
            None => bundled()?,
            Some(path) => Arc::new(
                py.detach(|| rusyntax::Parser::from_path(path))
                    .map_err(|err| load_error(path, err))?,
            ),
        };
        Ok(Parser {
            parser,
            path,
            tools: OnceLock::new(),
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        match &self.path {
            None => Ok("Parser()".to_owned()),
            Some(path) => Ok(format!(
                "Parser({})",
                PyString::new(py, &path.to_string_lossy()).repr()?
            )),
        }
    }

    fn __copy__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __deepcopy__<'py>(slf: Bound<'py, Self>, _memo: &Bound<'py, PyAny>) -> Bound<'py, Self> {
        slf
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let args = PyTuple::new(py, [slf.get().path.clone().into_bound_py_any(py)?])?;
        Ok((slf.get_type().into_any(), args))
    }

    /// Путь к файлу модели; None - модель пакета
    #[getter]
    fn path(&self) -> Option<PathBuf> {
        self.path.clone()
    }

    /// Типы связей, которые выдает модель
    #[getter]
    fn relations(&self) -> Vec<String> {
        self.parser.relations().to_vec()
    }

    /// Описание обучения модели: релиз UD, трибанки, LAS на dev и т. д.; None - если его нет
    #[getter]
    fn sources<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.parser
            .sources()
            .map(|text| py.import("json")?.call_method1("loads", (text,)))
            .transpose()
    }

    /// Разбор предложения
    ///
    /// Описание:
    ///     Вершина и связь каждого слова в том же порядке; у пустого предложения - пустой
    ///     список. Части речи и признаки - разметки UD, как у ruMorph; незнакомые модели
    ///     значения разбираются как неизвестные. Разбор идет без GIL
    ///
    /// Аргументы:
    ///     words (Sequence[str]): Формы слов
    ///     upos (Sequence[str]): Части речи UD
    ///     feats (Sequence[str | None]): Признаки в записи CoNLL-U (Case=Nom|Number=Sing);
    ///         `_`, пустая строка и None - без признаков
    ///
    /// Вывод:
    ///     list[tuple[int, str]]: Номер вершины (0 - корень) и тип связи каждого слова
    ///
    /// Исключения:
    ///     ValueError: Если длины words, upos и feats не совпадают
    ///     TypeError: Если вход - не набор строк
    ///
    /// Пример использования:
    ///     >>> Parser().parse(["Спасибо", "!"], ["INTJ", "PUNCT"], [None, None])
    ///     [(0, 'root'), (1, 'punct')]
    fn parse(
        &self,
        py: Python<'_>,
        words: &Bound<'_, PyAny>,
        upos: &Bound<'_, PyAny>,
        feats: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<(usize, String)>> {
        let sentence = sentence(words, upos, feats)?;
        Ok(parse_all(py, &self.parser, &[sentence])?
            .pop()
            .unwrap_or_default())
    }

    /// Разбор нескольких предложений
    ///
    /// Описание:
    ///     То же, что parse для каждого предложения, но быстрее: предложения близкой длины
    ///     разбираются по четыре сразу. Разбор идет без GIL
    ///
    /// Аргументы:
    ///     sentences (Iterable[tuple[Sequence[str], Sequence[str], Sequence[str | None]]]):
    ///         Предложения - тройки (формы, части речи, признаки)
    ///
    /// Вывод:
    ///     list[list[tuple[int, str]]]: Вершины и связи каждого предложения
    ///
    /// Исключения:
    ///     ValueError: Если у предложения не три набора или их длины не совпадают
    ///     TypeError: Если вход - не набор строк
    ///
    /// Пример использования:
    ///     >>> Parser().parse_many([(["Спасибо", "!"], ["INTJ", "PUNCT"], ["_", "_"])])
    ///     [[(0, 'root'), (1, 'punct')]]
    fn parse_many(
        &self,
        py: Python<'_>,
        sentences: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Vec<(usize, String)>>> {
        let items = sentences
            .try_iter()
            .map_err(|_| type_error("Ожидается набор предложений", sentences))?;
        let mut parsed = Vec::new();
        for (number, item) in items.enumerate() {
            let item = item?;
            let parts: Vec<Bound<'_, PyAny>> = match item.try_iter() {
                Ok(parts) if !item.is_instance_of::<PyString>() => {
                    parts.collect::<PyResult<_>>()?
                }
                _ => {
                    return Err(type_error(
                        &format!("Предложение {number} должно быть тройкой"),
                        &item,
                    ));
                }
            };
            let [words, upos, feats] = parts.as_slice() else {
                return Err(PyValueError::new_err(format!(
                    "Предложение {number}: ожидается тройка (формы, части речи, признаки), получено элементов: {}",
                    parts.len()
                )));
            };
            parsed.push(sentence(words, upos, feats)?);
        }
        parse_all(py, &self.parser, &parsed)
    }

    /// Разбор документа сегментатора anySeg
    ///
    /// Описание:
    ///     Заполняет столбцы head и deprel слов документа; прежние значения заменяются. Части
    ///     речи и признаки берутся из столбцов upos и feats: документ должен быть разобран
    ///     морфологически, например rumorph.Analyzer().annotate(document). Документ -
    ///     anyseg.Document или объект с теми же column, sentences (у предложения - words,
    ///     у слова - index) и set_columns. Разбор идет без GIL
    ///
    /// Аргументы:
    ///     document (anyseg.Document): Документ
    ///
    /// Исключения:
    ///     ValueError: Если у слова нет части речи или предложения документа не идут подряд
    ///         по всем его словам
    ///     TypeError: Если столбцы документа не строки или номер слова не целое число
    ///
    /// Пример использования:
    ///     >>> from anyseg import Segmenter
    ///     >>> from rumorph import Analyzer
    ///     >>> document = Segmenter("ru").segment("Мама мыла раму.")
    ///     >>> Analyzer().annotate(document)
    ///     >>> Parser().annotate(document)
    ///     >>> [(word.form, word.head, word.deprel) for word in document.words]
    ///     [('Мама', 2, 'nsubj'), ('мыла', 0, 'root'), ('раму', 2, 'obj'), ('.', 2, 'punct')]
    fn annotate(&self, py: Python<'_>, document: &Bound<'_, PyAny>) -> PyResult<()> {
        let column = |name: &str| document.call_method1(intern!(py, "column"), (name,));
        let forms = strings(&column("form")?, "form", None)?;
        // Столбец читается один раз: column может вернуть генератор
        let upos = strings(&column("upos")?, "upos", Some(""))?;
        if upos.iter().any(String::is_empty) {
            return Err(PyValueError::new_err(
                "У слов документа нет частей речи: сначала разберите его морфологически, \
                 например rumorph.Analyzer().annotate(document)",
            ));
        }
        let feats = strings(&column("feats")?, "feats", Some("_"))?;
        if upos.len() != forms.len() || feats.len() != forms.len() {
            return Err(PyValueError::new_err("столбцы документа разной длины"));
        }
        // Предложения идут подряд и покрывают все слова
        let unordered =
            || PyValueError::new_err("предложения документа не идут подряд по всем его словам");
        let mut sentences = Vec::new();
        let mut next = 0;
        for sentence in document.getattr(intern!(py, "sentences"))?.try_iter()? {
            let words = sentence?.getattr(intern!(py, "words"))?;
            let words = words
                .try_iter()
                .map_err(|_| type_error("Слова предложения должны быть набором", &words))?;
            let (mut start, mut count) = (next, 0);
            for word in words {
                if count == 0 {
                    let index = word?.getattr(intern!(py, "index"))?;
                    // Целое вне диапазона - номер за пределами документа, а не ошибка типа
                    start = match index.extract::<isize>() {
                        Ok(index) => usize::try_from(index).map_err(|_| unordered())?,
                        Err(err) if err.is_instance_of::<PyOverflowError>(py) => {
                            return Err(unordered());
                        }
                        Err(_) => {
                            return Err(type_error("Номер слова должен быть целым числом", &index));
                        }
                    };
                }
                count += 1;
            }
            if start != next || start + count > forms.len() {
                return Err(unordered());
            }
            let range = start..start + count;
            sentences.push((
                forms[range.clone()].to_vec(),
                upos[range.clone()].to_vec(),
                feats[range].to_vec(),
            ));
            next = start + count;
        }
        if next != forms.len() {
            return Err(unordered());
        }
        let (mut heads, mut deprels) = (Vec::new(), Vec::new());
        for relations in parse_all(py, &self.parser, &sentences)? {
            for (head, deprel) in relations {
                heads.push(head);
                deprels.push(deprel);
            }
        }
        let columns = PyDict::new(py);
        columns.set_item("head", heads)?;
        columns.set_item("deprel", deprels)?;
        document.call_method(intern!(py, "set_columns"), (), Some(&columns))?;
        Ok(())
    }

    /// Разбор текста
    ///
    /// Описание:
    ///     Текст делится на предложения и слова сегментатором anySeg, слова получают лемму, часть
    ///     речи и признаки от ruMorph с контекстом, затем - вершину и тип связи (annotate).
    ///     Сегментатор и анализатор создаются при первом вызове и дальше переиспользуются
    ///
    /// Аргументы:
    ///     text (str): Текст
    ///
    /// Вывод:
    ///     anyseg.Document: Документ со столбцами lemma, upos, feats, head и deprel
    ///
    /// Исключения:
    ///     TypeError: Если текст не строка
    ///
    /// Пример использования:
    ///     >>> document = Parser().parse_text("Мама мыла раму.")
    ///     >>> [(word.form, word.upos, word.head, word.deprel) for word in document.words]
    ///     [('Мама', 'NOUN', 2, 'nsubj'), ('мыла', 'VERB', 0, 'root'), ('раму', 'NOUN', 2, 'obj'), ('.', 'PUNCT', 2, 'punct')]
    fn parse_text<'py>(
        &self,
        py: Python<'py>,
        text: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !text.is_instance_of::<PyString>() {
            return Err(type_error("Текст должен быть строкой", text));
        }
        let (segmenter, analyzer) = self.tools(py)?;
        let document = segmenter.call_method1(intern!(py, "segment"), (text,))?;
        analyzer.call_method1(intern!(py, "annotate"), (&document,))?;
        self.annotate(py, &document)?;
        Ok(document)
    }
}

#[pymodule(gil_used = false)]
fn _rusyntax(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", package_version())?;
    m.add_class::<Parser>()?;
    Ok(())
}
