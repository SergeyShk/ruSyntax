//! Сверка прямого прохода с PyTorch: модель репозитория (`data/ru-syntax.bin`) разбирает предложения
//! `data/check.conllu` так же, как квантованная модель на PyTorch при обучении (`scripts/train.py`).
//! Допуск - 0,2 % слов: последние биты сумм на разных процессорах (FMA) различаются, и равные почти
//! оценки могут поменяться местами.

use std::path::PathBuf;

use rusyntax::{Parser, Word};

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data")
        .join(name)
}

/// Слова предложения (форма, UPOS, признаки) и вершины со связями PyTorch
type Sentence = (Vec<(String, String, String)>, Vec<(usize, String)>);

/// Предложения файла сверки
fn check() -> Vec<Sentence> {
    let text = std::fs::read_to_string(data("check.conllu")).expect("data/check.conllu");
    let mut out = Vec::new();
    let (mut words, mut parse) = (Vec::new(), Vec::new());
    for line in text.lines().chain([""]) {
        if line.is_empty() {
            if !words.is_empty() {
                out.push((std::mem::take(&mut words), std::mem::take(&mut parse)));
            }
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        words.push((cols[1].to_owned(), cols[3].to_owned(), cols[5].to_owned()));
        parse.push((cols[6].parse().expect("вершина"), cols[7].to_owned()));
    }
    out
}

#[test]
fn model_matches_pytorch() {
    // В опубликованном крейте нет каталога data: без файла сверки тест пропускается, без модели - падает
    if !data("check.conllu").exists() {
        eprintln!("нет data/check.conllu: сверка пропущена");
        return;
    }
    let parser = Parser::from_path(data("ru-syntax.bin")).expect("data/ru-syntax.bin");
    let sentences = check();
    let words: Vec<Vec<Word<'_>>> = sentences
        .iter()
        .map(|(words, _)| {
            words
                .iter()
                .map(|(form, upos, feats)| Word::new(form, upos, feats))
                .collect()
        })
        .collect();
    let many = parser.parse_many(&words);
    let (mut total, mut differ) = (0, 0);
    for ((words, (_, expected)), parse) in words.iter().zip(&sentences).zip(&many) {
        assert_eq!(parser.parse(words), *parse);
        for ((head, deprel), relation) in expected.iter().zip(parse) {
            total += 1;
            differ += usize::from(*head != relation.head || deprel != relation.deprel);
        }
    }
    assert!(total > 4000, "в файле сверки {total} слов");
    assert!(
        differ * 500 <= total,
        "разбор расходится с PyTorch у {differ} слов из {total}"
    );
}
