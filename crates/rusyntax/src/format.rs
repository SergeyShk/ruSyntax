//! Формат файла модели: заголовок (сигнатура, версия формата, CRC32 остальной части файла) и записи по именам.
//!
//! Записи - как у модели контекста ruMorph: имя (u16 длины и UTF-8), вид и данные, числа - little-endian.
//! Виды: 0 - строки, 1 - матрица f32, 2 - матрица int8 по строкам (множитель f32 на строку), 3 - числа u32.

use std::collections::HashMap;

use crate::Error;
use crate::kernels::Matrix;

/// Сигнатура файла модели
pub(crate) const MAGIC: &[u8; 8] = b"RUSYNTAX";
/// Версия формата, которую читает крейт
pub(crate) const FORMAT: u32 = 1;
/// Длина заголовка: сигнатура, версия, CRC32
pub(crate) const HEADER: usize = 16;
const MAX_DIM: usize = 4096;
const MAX_ROWS: usize = 1 << 22;
const MAX_ENTRIES: usize = 256;

pub(crate) fn invalid(message: &str) -> Error {
    Error::Format(message.to_owned())
}

/// Значение записи
pub(crate) enum Value<'a> {
    Strings(Vec<&'a str>),
    Matrix(Matrix),
    Numbers(Vec<u32>),
}

/// Чтение с проверкой границ
struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], Error> {
        let end = self
            .at
            .checked_add(count)
            .filter(|&end| end <= self.data.len())
            .ok_or_else(|| invalid("файл модели обрезан"))?;
        let bytes = &self.data[self.at..end];
        self.at = end;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32, Error> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn count(&mut self, limit: usize) -> Result<usize, Error> {
        let count = self.u32()? as usize;
        if count > limit {
            return Err(invalid("размер записи больше допустимого"));
        }
        Ok(count)
    }

    fn f32s(&mut self, count: usize) -> Result<Vec<f32>, Error> {
        let size = count
            .checked_mul(4)
            .ok_or_else(|| invalid("размер записи"))?;
        let bytes = self.take(size)?;
        let values: Vec<f32> = (bytes.as_chunks::<4>().0.iter())
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid("в весах есть NaN или бесконечность"));
        }
        Ok(values)
    }

    fn value(&mut self) -> Result<Value<'a>, Error> {
        let kind = self.take(1)?[0];
        match kind {
            0 => {
                let count = self.count(MAX_ROWS)?;
                let mut strings = Vec::with_capacity(count.min(1 << 16));
                for _ in 0..count {
                    let len = self.count(1 << 16)?;
                    let text = std::str::from_utf8(self.take(len)?)
                        .map_err(|_| invalid("строка не в UTF-8"))?;
                    strings.push(text);
                }
                Ok(Value::Strings(strings))
            }
            1 | 2 => {
                let (rows, cols) = (self.count(MAX_ROWS)?, self.count(MAX_DIM)?);
                let size = rows
                    .checked_mul(cols)
                    .ok_or_else(|| invalid("размер записи"))?;
                let data = if kind == 1 {
                    self.f32s(size)?
                } else {
                    let scales = self.f32s(rows)?;
                    let values = self.take(size)?;
                    (values.chunks_exact(cols.max(1)).zip(&scales))
                        .flat_map(|(row, &scale)| {
                            row.iter().map(move |&q| f32::from(q as i8) * scale)
                        })
                        .collect()
                };
                Ok(Value::Matrix(Matrix { rows, cols, data }))
            }
            3 => {
                let count = self.count(64)?;
                (0..count)
                    .map(|_| self.u32())
                    .collect::<Result<_, _>>()
                    .map(Value::Numbers)
            }
            _ => Err(invalid(&format!("неизвестный вид записи {kind}"))),
        }
    }
}

/// Записи файла по именам
pub(crate) struct Entries<'a>(HashMap<&'a str, Value<'a>>);

impl<'a> Entries<'a> {
    fn take(&mut self, name: &str) -> Result<Value<'a>, Error> {
        self.0
            .remove(name)
            .ok_or_else(|| invalid(&format!("нет записи {name}")))
    }

    pub(crate) fn matrix(&mut self, name: &str, rows: usize, cols: usize) -> Result<Matrix, Error> {
        match self.take(name)? {
            Value::Matrix(matrix) if matrix.rows == rows && matrix.cols == cols => Ok(matrix),
            Value::Matrix(matrix) => Err(invalid(&format!(
                "{name}: размер {}×{}, нужен {rows}×{cols}",
                matrix.rows, matrix.cols
            ))),
            _ => Err(invalid(&format!("{name} - не матрица"))),
        }
    }

    pub(crate) fn vector(&mut self, name: &str, size: usize) -> Result<Vec<f32>, Error> {
        Ok(self.matrix(name, 1, size)?.data)
    }

    pub(crate) fn numbers(&mut self, name: &str) -> Result<Vec<u32>, Error> {
        match self.take(name)? {
            Value::Numbers(numbers) => Ok(numbers),
            _ => Err(invalid(&format!("{name} - не числа"))),
        }
    }

    pub(crate) fn strings(&mut self, name: &str) -> Result<Vec<&'a str>, Error> {
        match self.take(name)? {
            Value::Strings(strings) => Ok(strings),
            _ => Err(invalid(&format!("{name} - не строки"))),
        }
    }

    /// Необязательная запись из одной строки
    pub(crate) fn text(&mut self, name: &str) -> Result<Option<&'a str>, Error> {
        match self.0.remove(name) {
            None => Ok(None),
            Some(Value::Strings(strings)) if strings.len() == 1 => Ok(Some(strings[0])),
            Some(_) => Err(invalid(&format!("{name} - не одна строка"))),
        }
    }

    /// Номера строк таблицы по строкам записи `name` с номера `first`; повтор строки - ошибка:
    /// иначе номер вышел бы за таблицу
    pub(crate) fn index(&mut self, name: &str, first: u32) -> Result<HashMap<String, u32>, Error> {
        let strings = self.strings(name)?;
        let index: HashMap<String, u32> = (strings.iter().enumerate())
            .map(|(number, text)| ((*text).to_owned(), number as u32 + first))
            .collect();
        if index.len() != strings.len() {
            return Err(invalid(&format!("в {name} повторяются строки")));
        }
        Ok(index)
    }

    /// Записи, которые модель не прочитала: лишняя запись - признак файла от другой версии обучения
    pub(crate) fn finish(self) -> Result<(), Error> {
        match self.0.keys().min() {
            None => Ok(()),
            Some(name) => Err(invalid(&format!("лишняя запись {name}"))),
        }
    }
}

/// Контрольная сумма из заголовка файла, уже проверенного `read`
pub(crate) fn checksum(data: &[u8]) -> u32 {
    (data.get(MAGIC.len() + 4..HEADER))
        .and_then(|bytes| bytes.try_into().ok())
        .map_or(0, u32::from_le_bytes)
}

/// Записи файла: проверяются сигнатура, версия формата, контрольная сумма и границы записей
pub(crate) fn read(data: &[u8]) -> Result<Entries<'_>, Error> {
    if data.get(..MAGIC.len()) != Some(MAGIC.as_slice()) {
        return Err(invalid("нет сигнатуры файла модели ruSyntax"));
    }
    if data.len() < HEADER {
        return Err(invalid("файл модели обрезан"));
    }
    let mut reader = Reader {
        data,
        at: MAGIC.len(),
    };
    let format = reader.u32()?;
    if format != FORMAT {
        return Err(invalid(&format!(
            "версия формата {format}, поддерживается {FORMAT}"
        )));
    }
    let checksum = reader.u32()?;
    if crc32fast::hash(&data[HEADER..]) != checksum {
        return Err(invalid("контрольная сумма не совпадает: файл поврежден"));
    }
    let count = reader.count(MAX_ENTRIES)?;
    let mut entries = HashMap::new();
    for _ in 0..count {
        let len = reader.take(2)?;
        let len = usize::from(u16::from_le_bytes([len[0], len[1]]));
        let name =
            std::str::from_utf8(reader.take(len)?).map_err(|_| invalid("имя записи не в UTF-8"))?;
        if entries.insert(name, reader.value()?).is_some() {
            return Err(invalid(&format!("запись {name} повторяется")));
        }
    }
    if reader.at != data.len() {
        return Err(invalid("лишние данные после записей"));
    }
    Ok(Entries(entries))
}

/// Запись файла модели: для тестов крейта (обучение пишет тот же формат на Python)
#[cfg(test)]
pub(crate) mod write {
    use super::{FORMAT, HEADER, MAGIC};

    pub(crate) enum Item<'a> {
        Strings(Vec<&'a str>),
        F32(usize, usize, Vec<f32>),
        Int8(usize, usize, Vec<f32>),
        Numbers(Vec<u32>),
    }

    pub(crate) fn file(items: &[(&str, Item<'_>)]) -> Vec<u8> {
        let mut body = (items.len() as u32).to_le_bytes().to_vec();
        for (name, item) in items {
            body.extend((name.len() as u16).to_le_bytes());
            body.extend(name.as_bytes());
            match item {
                Item::Strings(strings) => {
                    body.push(0);
                    body.extend((strings.len() as u32).to_le_bytes());
                    for text in strings {
                        body.extend((text.len() as u32).to_le_bytes());
                        body.extend(text.as_bytes());
                    }
                }
                Item::F32(rows, cols, data) => {
                    body.push(1);
                    body.extend((*rows as u32).to_le_bytes());
                    body.extend((*cols as u32).to_le_bytes());
                    data.iter()
                        .for_each(|value| body.extend(value.to_le_bytes()));
                }
                Item::Int8(rows, cols, data) => {
                    body.push(2);
                    body.extend((*rows as u32).to_le_bytes());
                    body.extend((*cols as u32).to_le_bytes());
                    let scales: Vec<f32> = (data.chunks_exact((*cols).max(1)))
                        .map(|row| {
                            row.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-12) / 127.0
                        })
                        .collect();
                    scales
                        .iter()
                        .for_each(|value| body.extend(value.to_le_bytes()));
                    for (row, scale) in data.chunks_exact((*cols).max(1)).zip(&scales) {
                        body.extend(
                            row.iter()
                                .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8 as u8),
                        );
                    }
                }
                Item::Numbers(numbers) => {
                    body.push(3);
                    body.extend((numbers.len() as u32).to_le_bytes());
                    numbers
                        .iter()
                        .for_each(|value| body.extend(value.to_le_bytes()));
                }
            }
        }
        let mut out = Vec::with_capacity(HEADER + body.len());
        out.extend(MAGIC);
        out.extend(FORMAT.to_le_bytes());
        out.extend(crc32fast::hash(&body).to_le_bytes());
        out.extend(body);
        out
    }
}
