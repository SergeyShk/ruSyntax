//! Модель: веса из файла и прямой проход - BiLSTM над словами с тегами, биаффинная оценка вершин,
//! линейная оценка связей по паре «зависимое - вершина».

use std::collections::HashMap;

use crate::Error;
use crate::format::{self, Entries, invalid};
use crate::kernels::{BATCH, Lstm, Matrix, Tiled, bilstm, bilstm_many, dot};
use crate::tree;

/// Число окончаний слова на входе: длиной 1..=4 с меткой начала «^»
pub(crate) const SUFFIXES: usize = 4;
/// Строка таблицы для неизвестной строки (строка 0 - пустая, в суммах не участвует)
const UNKNOWN: usize = 1;

/// Слой `out = LeakyReLU(W · in + b)` с наклоном 0,1
#[derive(Debug)]
struct Dense {
    weight: Tiled,
    bias: Vec<f32>,
    rows: usize,
}

impl Dense {
    fn read(
        entries: &mut Entries<'_>,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self, Error> {
        Ok(Dense {
            weight: Tiled::new(&entries.matrix(&format!("{name}.weight"), rows, cols)?),
            bias: entries.vector(&format!("{name}.bias"), rows)?,
            rows,
        })
    }

    fn apply(&self, inputs: &[f32], count: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; count * self.rows];
        self.weight.apply_all(inputs, &self.bias, &mut out);
        // Без ветвления цикл векторизуется
        for value in &mut out {
            *value = if *value < 0.0 { *value * 0.1 } else { *value };
        }
        out
    }
}

/// Таблица вложений: номер строки по ключу
#[derive(Debug)]
struct Table {
    index: HashMap<String, u32>,
    rows: Matrix,
}

impl Table {
    fn read(
        entries: &mut Entries<'_>,
        strings: &str,
        matrix: &str,
        cols: usize,
    ) -> Result<Self, Error> {
        let index = entries.index(strings, 2)?;
        let rows = entries.matrix(matrix, index.len() + 2, cols)?;
        Ok(Table { index, rows })
    }

    fn number(&self, key: &str) -> usize {
        self.number_or(key, UNKNOWN)
    }

    fn number_or(&self, key: &str, unknown: usize) -> usize {
        self.index
            .get(key)
            .map_or(unknown, |&number| number as usize)
    }

    fn row(&self, number: usize) -> &[f32] {
        &self.rows.data[number * self.rows.cols..(number + 1) * self.rows.cols]
    }
}

/// Слово на входе модели
pub(crate) struct Input<'a> {
    pub form: &'a str,
    pub upos: &'a str,
    pub feats: &'a str,
}

/// Связь слова: вершина (0 - корень), номер связи и их вероятности
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Link {
    pub head: usize,
    pub label: usize,
    pub head_prob: f32,
    pub label_prob: f32,
}

/// Связи слов предложения
pub(crate) type Parse = Vec<Link>;

#[derive(Debug)]
pub(crate) struct Model {
    hidden: usize,
    max_feats: usize,
    words: Option<Table>,
    suffixes: Table,
    upos: Table,
    /// Строка незнакомой части речи у слова: `X`, если он есть в словаре (строку неизвестного модель
    /// видит только у корня)
    unknown_upos: usize,
    feats: Table,
    lstm: Vec<[Lstm; 2]>,
    arc_dep: Dense,
    arc_head: Dense,
    rel_dep: Dense,
    rel_head: Dense,
    /// Биаффинная оценка вершин: `u = W·dep + b`, оценка вершины `j` - `u · head_j`
    arc: Tiled,
    arc_bias: Vec<f32>,
    arc_dim: usize,
    label: Tiled,
    label_bias: Vec<f32>,
    rel_dim: usize,
    /// Номер связи `root`: она только у слова с вершиной 0, и у него только она
    root: Option<usize>,
    pub relations: Vec<String>,
    pub sources: Option<String>,
    /// CRC32 файла после заголовка
    pub checksum: u32,
}

impl Model {
    /// Веса из файла модели: проверяются формат, контрольная сумма и размеры всех матриц
    pub fn parse(data: &[u8]) -> Result<Self, Error> {
        let mut entries = format::read(data)?;
        let config = entries.numbers("config")?;
        let &[
            hidden,
            layers,
            word_dim,
            suffix_dim,
            tag_dim,
            arc_dim,
            rel_dim,
            max_feats,
        ] = config.as_slice()
        else {
            return Err(invalid("в config не 8 чисел"));
        };
        let [
            hidden,
            layers,
            word_dim,
            suffix_dim,
            tag_dim,
            arc_dim,
            rel_dim,
            max_feats,
        ] = [
            hidden, layers, word_dim, suffix_dim, tag_dim, arc_dim, rel_dim, max_feats,
        ]
        .map(|value| value as usize);
        if [hidden, suffix_dim, tag_dim, arc_dim, rel_dim]
            .iter()
            .any(|&value| value == 0 || value > 1024)
            || word_dim > 1024
            || !(1..=4).contains(&layers)
            || !(1..=64).contains(&max_feats)
        {
            return Err(invalid("неверные размеры в config"));
        }
        let words = if word_dim > 0 {
            Some(Table::read(
                &mut entries,
                "vocab.words",
                "embed.word",
                word_dim,
            )?)
        } else {
            None
        };
        let suffixes = Table::read(&mut entries, "vocab.suffixes", "embed.suffix", suffix_dim)?;
        let upos = Table::read(&mut entries, "vocab.upos", "embed.upos", tag_dim)?;
        let unknown_upos = upos.number("X");
        let feats = Table::read(&mut entries, "vocab.feats", "embed.feats", tag_dim)?;
        let mut lstm = Vec::with_capacity(layers);
        let mut din = word_dim + SUFFIXES * suffix_dim + 2 * tag_dim;
        for layer in 0..layers {
            let mut cell = |direction: &str| -> Result<Lstm, Error> {
                let prefix = format!("lstm.{layer}.{direction}");
                Ok(Lstm {
                    input: Tiled::new(&entries.matrix(&format!("{prefix}.ih"), 4 * hidden, din)?),
                    hidden: Tiled::new(&entries.matrix(
                        &format!("{prefix}.hh"),
                        4 * hidden,
                        hidden,
                    )?),
                    bias: entries.vector(&format!("{prefix}.bias"), 4 * hidden)?,
                })
            };
            lstm.push([cell("forward")?, cell("backward")?]);
            din = 2 * hidden;
        }
        let arc_dep = Dense::read(&mut entries, "arc_dep", arc_dim, 2 * hidden)?;
        let arc_head = Dense::read(&mut entries, "arc_head", arc_dim, 2 * hidden)?;
        let rel_dep = Dense::read(&mut entries, "rel_dep", rel_dim, 2 * hidden)?;
        let rel_head = Dense::read(&mut entries, "rel_head", rel_dim, 2 * hidden)?;
        // В файле - матрица биаффинного слоя PyTorch (a + 1) × a: последняя строка - смещение зависимого
        let biaffine = entries.matrix("arc.weight", arc_dim + 1, arc_dim)?;
        let mut transposed = vec![0.0f32; arc_dim * arc_dim];
        for row in 0..arc_dim {
            for col in 0..arc_dim {
                transposed[col * arc_dim + row] = biaffine.data[row * arc_dim + col];
            }
        }
        let arc_bias = biaffine.data[arc_dim * arc_dim..].to_vec();
        let arc = Tiled::new(&Matrix {
            rows: arc_dim,
            cols: arc_dim,
            data: transposed,
        });
        let relations: Vec<String> = entries
            .strings("relations")?
            .iter()
            .map(|text| (*text).to_owned())
            .collect();
        if relations.is_empty() {
            return Err(invalid("пустой список связей"));
        }
        let label = Tiled::new(&entries.matrix("label.weight", relations.len(), 2 * rel_dim)?);
        let label_bias = entries.vector("label.bias", relations.len())?;
        let root = relations.iter().position(|relation| relation == "root");
        let sources = entries.text("sources")?.map(str::to_owned);
        entries.finish()?;
        Ok(Model {
            hidden,
            max_feats,
            words,
            suffixes,
            upos,
            unknown_upos,
            feats,
            lstm,
            arc_dep,
            arc_head,
            rel_dep,
            rel_head,
            arc,
            arc_bias,
            arc_dim,
            label,
            label_bias,
            rel_dim,
            root,
            relations,
            sources,
            checksum: format::checksum(data),
        })
    }

    fn input_dim(&self) -> usize {
        self.words.as_ref().map_or(0, |table| table.rows.cols)
            + SUFFIXES * self.suffixes.rows.cols
            + self.upos.rows.cols
            + self.feats.rows.cols
    }

    /// Вход BiLSTM: корень (строки неизвестного, без признаков) и слова - `(n + 1) × din`
    pub(crate) fn embed(&self, words: &[Input<'_>]) -> Vec<f32> {
        let din = self.input_dim();
        let mut x = vec![0.0f32; (words.len() + 1) * din];
        let mut key = String::new();
        let mut row = |t: usize, word: Option<&Input<'_>>| {
            let mut out = &mut x[t * din..(t + 1) * din];
            let normal = word.map(|word| normalize(word.form));
            if let Some(table) = &self.words {
                let number = normal.as_deref().map_or(UNKNOWN, |form| table.number(form));
                let (head, rest) = std::mem::take(&mut out).split_at_mut(table.rows.cols);
                head.copy_from_slice(table.row(number));
                out = rest;
            }
            let mut numbers = [UNKNOWN; SUFFIXES];
            if let Some(form) = normal.as_deref() {
                suffix_keys(form, &mut key, |k, key| {
                    numbers[k] = self.suffixes.number(key)
                });
            }
            for number in numbers {
                let (head, rest) = std::mem::take(&mut out).split_at_mut(self.suffixes.rows.cols);
                head.copy_from_slice(self.suffixes.row(number));
                out = rest;
            }
            let (head, rest) = std::mem::take(&mut out).split_at_mut(self.upos.rows.cols);
            head.copy_from_slice(self.upos.row(word.map_or(UNKNOWN, |word| {
                self.upos.number_or(word.upos, self.unknown_upos)
            })));
            if let Some(word) = word {
                for pair in feat_pairs(word.feats).take(self.max_feats) {
                    for (value, add) in rest.iter_mut().zip(self.feats.row(self.feats.number(pair)))
                    {
                        *value += add;
                    }
                }
            }
        };
        row(0, None);
        for (t, word) in words.iter().enumerate() {
            row(t + 1, Some(word));
        }
        x
    }

    /// Разбор до `BATCH` непустых предложений сразу
    pub fn parse_group(&self, sentences: &[&[Input<'_>]]) -> Vec<Parse> {
        debug_assert!(!sentences.is_empty() && sentences.len() <= BATCH);
        let d = 2 * self.hidden;
        let lens: Vec<usize> = sentences.iter().map(|words| words.len() + 1).collect();
        let mut states: Vec<Vec<f32>> = sentences.iter().map(|words| self.embed(words)).collect();
        for layer in &self.lstm {
            states = if states.len() == 1 {
                vec![bilstm(layer, &states[0], lens[0], d)]
            } else {
                let inputs: Vec<&[f32]> = states.iter().map(Vec::as_slice).collect();
                bilstm_many(layer, &inputs, &lens, d)
            };
        }
        states
            .iter()
            .zip(&lens)
            .map(|(state, &n)| self.decode(state, n))
            .collect()
    }

    /// Вершины и связи одного предложения по выходу BiLSTM (`n` позиций, 0 - корень)
    fn decode(&self, state: &[f32], n: usize) -> Parse {
        let (a, r) = (self.arc_dim, self.rel_dim);
        let dep = self.arc_dep.apply(state, n);
        let head = self.arc_head.apply(state, n);
        let mut u = vec![0.0f32; n * a];
        self.arc.apply_all(&dep, &self.arc_bias, &mut u);
        let score = |i: usize, j: usize| dot(&u[i * a..(i + 1) * a], &head[j * a..(j + 1) * a]);
        let (heads, head_probs) = tree::heads(n, &score);
        let rel_dep = self.rel_dep.apply(state, n);
        let rel_head = self.rel_head.apply(state, n);
        let mut pairs = vec![0.0f32; (n - 1) * 2 * r];
        for (i, pair) in pairs.chunks_exact_mut(2 * r).enumerate() {
            pair[..r].copy_from_slice(&rel_dep[(i + 1) * r..(i + 2) * r]);
            let h = heads[i + 1];
            pair[r..].copy_from_slice(&rel_head[h * r..(h + 1) * r]);
        }
        let labels = self.relations.len();
        let mut scores = vec![0.0f32; (n - 1) * labels];
        self.label.apply_all(&pairs, &self.label_bias, &mut scores);
        (1..n)
            .zip(scores.chunks_exact(labels))
            .map(|(i, row)| {
                // У корня связь root - по правилу, а не по оценке
                let (label, label_prob) = match self.root {
                    Some(root) if heads[i] == 0 => (root, 1.0),
                    root => {
                        let label = best_label(row, root);
                        (label, label_prob(row, label, root))
                    }
                };
                Link {
                    head: heads[i],
                    label,
                    head_prob: head_probs[i],
                    label_prob,
                }
            })
            .collect()
    }
}

/// Вероятность связи `label` - softmax оценок связей без `skip` (связи корня у слова с вершиной не корень)
fn label_prob(row: &[f32], label: usize, skip: Option<usize>) -> f32 {
    let values = || {
        (row.iter().enumerate())
            .filter(move |&(other, _)| Some(other) != skip || other == label)
            .map(|(_, &value)| f64::from(value))
    };
    let max = values().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = values().map(|value| (value - max).exp()).sum();
    ((f64::from(row[label]) - max).exp() / sum) as f32
}

/// Связь с лучшей оценкой, кроме `skip` (если других связей нет - `skip`)
fn best_label(row: &[f32], skip: Option<usize>) -> usize {
    let mut best: Option<usize> = None;
    for (label, &value) in row.iter().enumerate() {
        if Some(label) != skip && best.is_none_or(|best| value > row[best]) {
            best = Some(label);
        }
    }
    best.or(skip).unwrap_or(0)
}

/// Форма для словаря модели: строчные буквы, «ё» как «е»
fn normalize(form: &str) -> String {
    let lower = form.to_lowercase();
    if lower.contains('ё') {
        lower.replace('ё', "е")
    } else {
        lower
    }
}

/// Ключи окончаний `k:окончание` длиной `k` = 1..=4 у формы с меткой начала «^» (у коротких форм - вся
/// форма с меткой) по очереди в буфере `key`: `each(k - 1, ключ)`
fn suffix_keys(normal: &str, key: &mut String, mut each: impl FnMut(usize, &str)) {
    let mut starts = [None; SUFFIXES];
    for (start, (at, _)) in starts.iter_mut().zip(normal.char_indices().rev()) {
        *start = Some(at);
    }
    for (k, start) in starts.into_iter().enumerate() {
        key.clear();
        key.push(char::from(b'1' + k as u8));
        key.push(':');
        match start {
            Some(at) => key.push_str(&normal[at..]),
            None => {
                key.push('^');
                key.push_str(normal);
            }
        }
        each(k, key);
    }
}

/// Пары «признак=значение» из записи CoNLL-U `A=B|C=D`; `_` и пустая строка - без признаков
fn feat_pairs(feats: &str) -> impl Iterator<Item = &str> {
    feats
        .split('|')
        .filter(|pair| !pair.is_empty() && *pair != "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(normal: &str) -> Vec<String> {
        let mut out = Vec::new();
        suffix_keys(normal, &mut String::new(), |k, key| {
            assert_eq!(k, out.len());
            out.push(key.to_owned());
        });
        out
    }

    #[test]
    fn suffixes_mark_word_start() {
        assert_eq!(keys("я"), ["1:я", "2:^я", "3:^я", "4:^я"]);
        assert_eq!(keys("дома"), ["1:а", "2:ма", "3:ома", "4:дома"]);
        assert_eq!(keys("домами"), ["1:и", "2:ми", "3:ами", "4:мами"]);
        assert_eq!(keys(""), ["1:^", "2:^", "3:^", "4:^"]);
    }

    #[test]
    fn root_label_only_at_root() {
        let row = [0.0, 3.0, 1.0];
        assert_eq!(best_label(&row, None), 1);
        assert_eq!(best_label(&row, Some(1)), 2);
        assert_eq!(best_label(&[5.0], Some(0)), 0);
    }

    #[test]
    fn normalize_lowercases_and_drops_yo() {
        assert_eq!(normalize("ЁЛКА"), "елка");
        assert_eq!(normalize("Её"), "ее");
    }

    #[test]
    fn feats_skip_empty() {
        assert_eq!(feat_pairs("_").count(), 0);
        assert_eq!(feat_pairs("").count(), 0);
        assert_eq!(
            feat_pairs("Case=Nom|Number=Sing").collect::<Vec<_>>(),
            ["Case=Nom", "Number=Sing"]
        );
    }
}
