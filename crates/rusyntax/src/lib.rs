//! Синтаксический анализатор русского языка: для каждого слова предложения - вершина и тип связи
//! Universal Dependencies.
//!
//! На входе - слова предложения с частью речи и признаками UD (например, от морфологического анализатора
//! ruMorph), на выходе - номер вершины (0 - корень) и связь каждого слова. Модель - BiLSTM над словами
//! и их тегами, вершины - биаффинная оценка и дерево максимального веса с одним корнем (у вырожденных
//! и очень длинных предложений - быстрая починка лучших вершин; у предложений длиннее 1024 слов вершина
//! ищется среди 64 слов с каждой стороны), связи - по паре «зависимое - вершина».
//! Модель читается из файла формата ruSyntax ([`Parser::from_bytes`]).
//!
//! ```no_run
//! use rusyntax::{Parser, Word};
//!
//! let parser = Parser::from_path("ru-syntax.bin")?;
//! let words = [
//!     Word::new("Мама", "NOUN", "Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing"),
//!     Word::new("мыла", "VERB", "Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act"),
//!     Word::new("раму", "NOUN", "Animacy=Inan|Case=Acc|Gender=Fem|Number=Sing"),
//!     Word::new(".", "PUNCT", "_"),
//! ];
//! for (word, relation) in words.iter().zip(parser.parse(&words)) {
//!     println!("{} -> {} {}", word.form, relation.head, relation.deprel);
//! }
//! # Ok::<(), rusyntax::Error>(())
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod error;
mod format;
mod kernels;
mod model;
mod tree;

use std::path::Path;

pub use error::Error;

use kernels::BATCH;
use model::{Input, Model};

/// Слово предложения с морфологией UD; создается через [`Word::new`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Word<'a> {
    /// Форма слова, как в тексте
    pub form: &'a str,
    /// Часть речи UD (`NOUN`, `VERB`, ...) - одна из 17, заглавными буквами; незнакомая модели часть речи
    /// разбирается как `X`
    pub upos: &'a str,
    /// Признаки UD в записи CoNLL-U: `Case=Nom|Number=Sing`; `_` или пустая строка - без признаков
    pub feats: &'a str,
}

impl<'a> Word<'a> {
    /// Слово из формы, части речи и признаков
    #[must_use]
    pub fn new(form: &'a str, upos: &'a str, feats: &'a str) -> Self {
        Word { form, upos, feats }
    }
}

/// Связь слова с вершиной
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Relation<'p> {
    /// Номер вершины среди слов предложения с 1; 0 - корень предложения
    pub head: usize,
    /// Тип связи UD с подтипом, если он есть (`nsubj`, `obl:tmod`, `root`, ...); `root` - только у слова
    /// с вершиной 0
    pub deprel: &'p str,
    /// Номер типа связи в [`Parser::relations`]
    pub label: usize,
    /// Вероятность вершины по модели: softmax оценок всех кандидатов в вершины слова (у предложений
    /// длиннее 1024 слов - кандидатов в окне)
    pub head_prob: f32,
    /// Вероятность типа связи при этой вершине: softmax оценок связей; у связей, которые ставит правило, -
    /// `root` корня и `vocative` упоминания `@имя` в начале предложения - 1
    pub deprel_prob: f32,
}

/// Синтаксический анализатор: модель в памяти. Разбор не меняет анализатор, поэтому один анализатор
/// можно разделять между потоками
pub struct Parser {
    model: Model,
}

impl std::fmt::Debug for Parser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parser")
            .field("relations", &self.model.relations.len())
            .field("sources", &self.model.sources)
            .finish_non_exhaustive()
    }
}

impl Parser {
    /// Анализатор из файла модели
    ///
    /// # Errors
    ///
    /// [`Error::Io`], если файл не прочитать, и [`Error::Format`], если это не модель ruSyntax, версия формата
    /// другая, контрольная сумма не совпадает или размеры весов не сходятся
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Анализатор из содержимого файла модели
    ///
    /// # Errors
    ///
    /// [`Error::Format`], если это не модель ruSyntax, версия формата другая, контрольная сумма не совпадает
    /// или размеры весов не сходятся
    pub fn from_bytes(data: &[u8]) -> Result<Self, Error> {
        Ok(Parser {
            model: Model::parse(data)?,
        })
    }

    /// Типы связей, которые выдает модель
    #[must_use]
    pub fn relations(&self) -> &[String] {
        &self.model.relations
    }

    /// Описание обучения модели (запись `sources` файла, JSON), если оно есть
    #[must_use]
    pub fn sources(&self) -> Option<&str> {
        self.model.sources.as_deref()
    }

    /// Контрольная сумма модели - CRC32 файла после заголовка: отличает одну модель от другой
    #[must_use]
    pub fn checksum(&self) -> u32 {
        self.model.checksum
    }

    /// Разбор предложения: связь каждого слова в том же порядке; у пустого предложения - пустой список
    #[must_use]
    pub fn parse(&self, words: &[Word<'_>]) -> Vec<Relation<'_>> {
        self.parse_many(&[words]).pop().unwrap_or_default()
    }

    /// Разбор нескольких предложений: то же, что [`Parser::parse`] для каждого, но предложения близкой
    /// длины разбираются по четыре сразу - это быстрее
    #[must_use]
    pub fn parse_many<'w, S: AsRef<[Word<'w>]>>(&self, sentences: &[S]) -> Vec<Vec<Relation<'_>>> {
        let mut out: Vec<Vec<Relation<'_>>> = vec![Vec::new(); sentences.len()];
        let mut order: Vec<usize> = (0..sentences.len())
            .filter(|&i| !sentences[i].as_ref().is_empty())
            .collect();
        order.sort_by_key(|&i| std::cmp::Reverse(sentences[i].as_ref().len()));
        let inputs: Vec<Vec<Input<'_>>> = sentences
            .iter()
            .map(|words| {
                (words.as_ref().iter())
                    .map(|word| Input {
                        form: word.form,
                        upos: word.upos,
                        feats: word.feats,
                    })
                    .collect()
            })
            .collect();
        for group in order.chunks(BATCH) {
            let batch: Vec<&[Input<'_>]> = group.iter().map(|&i| inputs[i].as_slice()).collect();
            for (&i, parse) in group.iter().zip(self.model.parse_group(&batch)) {
                out[i] = (parse.into_iter())
                    .map(|link| Relation {
                        head: link.head,
                        deprel: &self.model.relations[link.label],
                        label: link.label,
                        head_prob: link.head_prob,
                        deprel_prob: link.label_prob,
                    })
                    .collect();
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::write::{Item, file};

    struct Rng(u64);

    impl Rng {
        fn values(&mut self, count: usize) -> Vec<f32> {
            (0..count)
                .map(|_| {
                    self.0 ^= self.0 << 13;
                    self.0 ^= self.0 >> 7;
                    self.0 ^= self.0 << 17;
                    ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.6
                })
                .collect()
        }
    }

    const RELATIONS: [&str; 5] = ["root", "nsubj", "obj", "punct", "obl:tmod"];

    /// Записи модели со случайными весами: `int8` - матрицы int8 по строкам, `word_dim` 0 - без слов
    fn items(layers: usize, word_dim: usize, int8: bool) -> Vec<(&'static str, Item<'static>)> {
        let (hidden, suffix_dim, tag_dim, arc, rel) = (8, 3, 4, 6, 5);
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let mut matrix = |rows: usize, cols: usize| {
            let data = rng.values(rows * cols);
            if int8 {
                Item::Int8(rows, cols, data)
            } else {
                Item::F32(rows, cols, data)
            }
        };
        let name = |text: String| -> &'static str { Box::leak(text.into_boxed_str()) };
        let mut out = vec![
            (
                "config",
                Item::Numbers(vec![
                    hidden as u32,
                    layers as u32,
                    word_dim as u32,
                    3,
                    4,
                    6,
                    5,
                    12,
                ]),
            ),
            ("sources", Item::Strings(vec![r#"{"train": ["test"]}"#])),
            (
                "vocab.suffixes",
                Item::Strings(vec!["1:а", "2:ма", "1:у", "1:.", "2:^."]),
            ),
            ("embed.suffix", matrix(7, suffix_dim)),
            (
                "vocab.upos",
                Item::Strings(vec!["NOUN", "VERB", "PUNCT", "X"]),
            ),
            ("embed.upos", matrix(6, tag_dim)),
            (
                "vocab.feats",
                Item::Strings(vec!["Case=Nom", "Case=Acc", "Number=Sing", "Gender=Fem"]),
            ),
            ("embed.feats", matrix(6, tag_dim)),
            ("relations", Item::Strings(RELATIONS.to_vec())),
        ];
        if word_dim > 0 {
            out.push((
                "vocab.words",
                Item::Strings(vec!["мама", "мыть", "мыла", "рама", "раму", "."]),
            ));
            out.push(("embed.word", matrix(8, word_dim)));
        }
        let mut din = word_dim + model::SUFFIXES * suffix_dim + 2 * tag_dim;
        for layer in 0..layers {
            for direction in ["forward", "backward"] {
                let prefix = format!("lstm.{layer}.{direction}");
                out.push((name(format!("{prefix}.ih")), matrix(4 * hidden, din)));
                out.push((name(format!("{prefix}.hh")), matrix(4 * hidden, hidden)));
                out.push((name(format!("{prefix}.bias")), matrix(1, 4 * hidden)));
            }
            din = 2 * hidden;
        }
        for (layer, rows) in [
            ("arc_dep", arc),
            ("arc_head", arc),
            ("rel_dep", rel),
            ("rel_head", rel),
        ] {
            out.push((name(format!("{layer}.weight")), matrix(rows, 2 * hidden)));
            out.push((name(format!("{layer}.bias")), matrix(1, rows)));
        }
        out.push(("arc.weight", matrix(arc + 1, arc)));
        out.push(("label.weight", matrix(RELATIONS.len(), 2 * rel)));
        out.push(("label.bias", matrix(1, RELATIONS.len())));
        out
    }

    fn model(layers: usize, word_dim: usize, int8: bool) -> Vec<u8> {
        file(&items(layers, word_dim, int8))
    }

    fn sentences() -> Vec<Vec<Word<'static>>> {
        let w = Word::new;
        vec![
            vec![
                w("Мама", "NOUN", "Case=Nom|Gender=Fem|Number=Sing"),
                w("мыла", "VERB", "Gender=Fem"),
                w("раму", "NOUN", "Case=Acc|Number=Sing"),
                w(".", "PUNCT", "_"),
            ],
            vec![w("Ёлка", "PROPN", "Case=Nom|Foo=Bar"), w("", "X", "")],
            vec![w("я", "PRON", "Case=Nom")],
            vec![],
            (0..37)
                .map(|i| w(if i % 3 == 0 { "мама" } else { "дом" }, "NOUN", "Case=Nom"))
                .collect(),
            vec![
                w("раму", "NOUN", "Case=Acc"),
                w("мыла", "VERB", "_"),
                w("мама", "NOUN", "Case=Nom"),
            ],
        ]
    }

    fn is_tree(relations: &[Relation<'_>]) -> bool {
        let n = relations.len();
        if relations.iter().filter(|r| r.head == 0).count() != 1
            || relations.iter().any(|r| r.head > n)
        {
            return false;
        }
        (1..=n).all(|start| {
            let (mut v, mut steps) = (start, 0);
            while v != 0 && steps <= n {
                v = relations[v - 1].head;
                steps += 1;
            }
            v == 0
        })
    }

    #[test]
    fn every_word_gets_one_relation_in_a_tree() {
        for (layers, word_dim, int8) in [(1, 5, false), (2, 5, true), (2, 0, false)] {
            let parser = Parser::from_bytes(&model(layers, word_dim, int8)).unwrap();
            assert_eq!(parser.relations(), RELATIONS);
            assert_eq!(parser.sources(), Some(r#"{"train": ["test"]}"#));
            for words in sentences() {
                let relations = parser.parse(&words);
                assert_eq!(relations.len(), words.len());
                assert!(words.is_empty() || is_tree(&relations), "{relations:?}");
                assert!(relations.iter().all(|r| RELATIONS[r.label] == r.deprel));
                assert!(
                    relations
                        .iter()
                        .all(|r| (r.head == 0) == (r.deprel == "root"))
                );
                // Вероятности от 0 до 1, у корня связь root - 1
                assert!(relations.iter().all(|r| {
                    r.head_prob > 0.0
                        && r.head_prob <= 1.0
                        && r.deprel_prob > 0.0
                        && r.deprel_prob <= 1.0
                        && (r.head != 0 || r.deprel_prob == 1.0)
                }));
            }
        }
    }

    #[test]
    fn mention_at_start_is_vocative() {
        // Модель со связью vocative вместо obl:tmod: упоминание в начале - корень или обращение к корню
        let mut with_vocative = items(1, 5, false);
        with_vocative.retain(|(name, _)| *name != "relations");
        with_vocative.push((
            "relations",
            Item::Strings(vec!["root", "nsubj", "obj", "punct", "vocative"]),
        ));
        let parser = Parser::from_bytes(&file(&with_vocative)).unwrap();
        let w = Word::new;
        let (mut applied, mut root) = (0, 0);
        for verb in ["мыла", "мыть", "мама", "раму", "дом"] {
            for first in ["@мама", "@m", "@рама_2026"] {
                let words = [
                    w(first, "PROPN", "_"),
                    w(verb, "VERB", "_"),
                    w(".", "PUNCT", "_"),
                ];
                let relations = parser.parse(&words);
                assert!(is_tree(&relations), "{relations:?}");
                let top = relations.iter().position(|r| r.head == 0).unwrap() + 1;
                if relations[0].head == 0 {
                    root += 1;
                    continue;
                }
                applied += 1;
                assert_eq!((relations[0].head, relations[0].deprel), (top, "vocative"));
                assert!(relations[0].head_prob > 0.0 && relations[0].deprel_prob == 1.0);
            }
        }
        assert!(applied > 0, "правило ни разу не сработало, корней {root}");
    }

    #[test]
    fn checksum_is_crc_of_body() {
        let data = model(1, 5, false);
        let parser = Parser::from_bytes(&data).unwrap();
        assert_eq!(parser.checksum(), crc32fast::hash(&data[format::HEADER..]));
    }

    #[test]
    fn many_sentences_match_one_by_one() {
        let parser = Parser::from_bytes(&model(2, 5, false)).unwrap();
        let all = sentences();
        let many = parser.parse_many(&all);
        let one: Vec<_> = all.iter().map(|words| parser.parse(words)).collect();
        assert_eq!(many, one);
        assert!(parser.parse_many::<Vec<Word<'_>>>(&[]).is_empty());
    }

    #[test]
    fn parse_is_deterministic() {
        let parser = Parser::from_bytes(&model(2, 5, true)).unwrap();
        let words = &sentences()[0];
        assert_eq!(parser.parse(words), parser.parse(words));
    }

    #[test]
    fn debug_is_short() {
        let parser = Parser::from_bytes(&model(1, 5, false)).unwrap();
        assert_eq!(
            format!("{parser:?}"),
            r#"Parser { relations: 5, sources: Some("{\"train\": [\"test\"]}"), .. }"#
        );
    }

    #[test]
    fn parser_is_shared_between_threads() {
        fn shared<T: Send + Sync>() {}
        shared::<Parser>();
    }

    fn format_error(data: &[u8]) -> String {
        match Parser::from_bytes(data) {
            Err(Error::Format(message)) => message,
            other => panic!("ожидалась ошибка формата: {other:?}"),
        }
    }

    /// Файл с пересчитанной контрольной суммой: порча, которую сумма не ловит
    fn resum(mut data: Vec<u8>) -> Vec<u8> {
        let checksum = crc32fast::hash(&data[format::HEADER..]);
        data[12..16].copy_from_slice(&checksum.to_le_bytes());
        data
    }

    #[test]
    fn damaged_files_are_rejected() {
        let good = model(1, 5, false);
        assert!(format_error(b"").contains("сигнатуры"));
        assert!(format_error(b"RUSYNTAX").contains("обрезан"));
        let mut version = good.clone();
        version[8] = 2;
        assert!(format_error(&version).contains("версия формата 2"));
        let mut flipped = good.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 1;
        assert!(format_error(&flipped).contains("контрольная сумма"));
        assert!(format_error(&resum(good[..good.len() - 3].to_vec())).contains("обрезан"));
        let mut extra = good.clone();
        extra.push(0);
        assert!(format_error(&resum(extra)).contains("лишние данные"));
    }

    #[test]
    fn inconsistent_models_are_rejected() {
        let without = |skip: &str| {
            let rest: Vec<_> = items(1, 5, false)
                .into_iter()
                .filter(|(name, _)| *name != skip)
                .collect();
            format_error(&file(&rest))
        };
        assert!(without("label.bias").contains("нет записи label.bias"));
        assert!(without("vocab.words").contains("нет записи vocab.words"));
        let mut extra = items(1, 5, false);
        extra.push(("lstm.1.forward.ih", Item::F32(1, 1, vec![0.0])));
        assert!(format_error(&file(&extra)).contains("лишняя запись lstm.1.forward.ih"));
        let mut duplicate = items(1, 5, false);
        duplicate.push(("sources", Item::Strings(vec!["{}"])));
        assert!(format_error(&file(&duplicate)).contains("запись sources повторяется"));
        let mut wrong = items(1, 5, false);
        wrong.retain(|(name, _)| *name != "arc.weight");
        wrong.push(("arc.weight", Item::F32(6, 6, vec![0.0; 36])));
        assert!(format_error(&file(&wrong)).contains("arc.weight: размер 6×6, нужен 7×6"));
        let mut repeated = items(1, 5, false);
        repeated.retain(|(name, _)| *name != "vocab.upos");
        repeated.push(("vocab.upos", Item::Strings(vec!["NOUN", "NOUN", "PUNCT"])));
        assert!(format_error(&file(&repeated)).contains("в vocab.upos повторяются строки"));
        let mut nan = items(1, 5, false);
        nan.retain(|(name, _)| *name != "label.bias");
        nan.push(("label.bias", Item::F32(1, 5, vec![f32::NAN; 5])));
        assert!(format_error(&file(&nan)).contains("NaN"));
        let mut config = items(1, 5, false);
        config[0] = ("config", Item::Numbers(vec![8, 9, 5, 3, 4, 6, 5, 12]));
        assert!(format_error(&file(&config)).contains("config"));
    }

    #[test]
    fn missing_file_is_io_error() {
        let err = Parser::from_path("/нет/такого/файла.bin").unwrap_err();
        assert!(matches!(err, Error::Io(_)));
        assert!(
            err.to_string()
                .starts_with("не удалось прочитать файл модели: ")
        );
    }

    #[test]
    fn unknown_upos_is_parsed_as_x() {
        let parser = Parser::from_bytes(&model(1, 5, false)).unwrap();
        let embed = |upos| {
            (parser.model).embed(&[model::Input {
                form: "мама",
                upos,
                feats: "_",
            }])
        };
        assert_eq!(embed("noun"), embed("X"));
        assert_eq!(embed("ADJ"), embed("X"));
        assert_ne!(embed("NOUN"), embed("X"));
    }
}
