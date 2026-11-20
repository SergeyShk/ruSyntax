# rusyntax

Быстрый синтаксический анализатор русского языка на Rust - ядро библиотеки
[ruSyntax](https://github.com/SergeyShk/ruSyntax) для Python: для каждого слова предложения - вершина и тип
связи [Universal Dependencies](https://universaldependencies.org/).

- Вход - слова предложения с частью речи и признаками UD (например, от морфологического анализатора
  [ruMorph](https://github.com/SergeyShk/ruMorph)), выход - номер вершины (0 - корень) и связь каждого слова.
- Модель - BiLSTM над словами, окончаниями и тегами; вершины - биаффинная оценка и дерево максимального веса
  с одним корнем (непроективные деревья тоже), связи - по паре «зависимое - вершина». У вырожденных
  предложений (больше 16 циклов или 8 кандидатов в корень у лучших вершин) и предложений длиннее 1024 слов
  дерево собирается быстрее: лишние корни и узлы циклов подвешиваются к лучшему корню; у длинных предложений
  вершина ищется среди 64 слов с каждой стороны и корня, и время растет с длиной линейно.
- Часть речи - одна из 17 частей речи UD заглавными буквами (незнакомая разбирается как `X`), признаки -
  запись CoNLL-U; у `Relation` - вершина, тип связи, его номер в `Parser::relations` и вероятности
  вершины и связи по модели (`head_prob`, `deprel_prob`); `Parser::checksum` - CRC32 модели.
- Предложения близкой длины разбираются по четыре сразу: `Parser::parse_many`.

## Установка

```toml
[dependencies]
rusyntax = "0.1"
```

Модель в крейт не входит: анализатор читает ее из файла. Модель версии 0.1 - файл
[`data/ru-syntax.bin`](https://github.com/SergeyShk/ruSyntax/raw/0.1.0/data/ru-syntax.bin) репозитория на теге 0.1.0
(CC BY-SA 4.0, [LICENSE-MODEL.txt](https://github.com/SergeyShk/ruSyntax/blob/0.1.0/LICENSE-MODEL.txt)).

## Пример

```rust,no_run
use rusyntax::{Parser, Word};

fn main() -> Result<(), rusyntax::Error> {
    let parser = Parser::from_path("ru-syntax.bin")?;
    let words = [
        Word::new("Мама", "NOUN", "Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing"),
        Word::new("мыла", "VERB", "Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act"),
        Word::new("раму", "NOUN", "Animacy=Inan|Case=Acc|Gender=Fem|Number=Sing"),
        Word::new(".", "PUNCT", "_"),
    ];
    for (word, relation) in words.iter().zip(parser.parse(&words)) {
        println!("{} -> {} {}", word.form, relation.head, relation.deprel);
    }
    Ok(())
}
```

Модель загружается и проверяется при каждом вызове `Parser::from_path` и `Parser::from_bytes`: создайте
анализатор один раз и разбирайте им весь текст, в том числе из нескольких потоков.

## Файл модели

Заголовок - сигнатура `RUSYNTAX`, версия формата и CRC32 остальной части файла; дальше - записи по именам:
списки строк, матрицы f32 или int8 по строкам с множителем на строку, числа. Файл другой версии формата,
поврежденный или с несходящимися размерами весов не загружается: `Error::Format`.

## Лицензия

Код распространяется под лицензией [MIT](https://github.com/SergeyShk/ruSyntax/blob/master/LICENSE.txt).
