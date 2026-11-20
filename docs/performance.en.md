# Comparison

From raw text ruSyntax is the most accurate of the compared parsers on the Taiga, GSD and Poetry treebanks and on par with Stanza on PUD; on SynTagRus Stanza is ahead - its models are trained on SynTagRus, which ruSyntax has not seen ([model](model.md)). At the same time ruSyntax parses about 30 thousand words per second in one thread together with the ruMorph morphology - 2.8 times faster than spaCy, 27 times faster than Stanza and 69 times faster than Stanza with a transformer - and loads in 0.08 s.

## Quality { #quality }

The gold standard is the test parts of the treebanks [UD_Russian-SynTagRus](https://github.com/UniversalDependencies/UD_Russian-SynTagRus), [UD_Russian-Taiga](https://github.com/UniversalDependencies/UD_Russian-Taiga), [UD_Russian-GSD](https://github.com/UniversalDependencies/UD_Russian-GSD), [UD_Russian-PUD](https://github.com/UniversalDependencies/UD_Russian-PUD) and [UD_Russian-Poetry](https://github.com/UniversalDependencies/UD_Russian-Poetry) of the UD 2.19 release. The metrics are those of the official CoNLL 2018 script: UAS is the share of words with the right head, LAS with the right head and relation (without the subtype); punctuation counts. The treebanks follow different annotation conventions, so compare systems on one treebank rather than treebanks with each other.

What the models are trained on: ruSyntax - Taiga, GSD and Poetry; Stanza and Stanza accurate (with the rubert-base-cased transformer) - SynTagRus of UD 2.18, so its test part is in-domain for them; spaCy ru_core_news_sm and Slovnet - Nerus, automatically annotated news; UDPipe 1 - SynTagRus of UD 2.5, whose relation conventions have changed since, so its LAS drops more than its UAS.

### Raw text { #raw }

Every system splits the text into sentences and words itself: this is how a parser works for the user. ruSyntax uses `parse_text` (anySeg and ruMorph). UAS / LAS, %:

| System | SynTagRus | Taiga | GSD | PUD | Poetry |
| :----- | --------: | ----: | --: | --: | -----: |
| ruSyntax | 90.6 / 87.8 | 84.8 / **81.5** | 88.8 / **84.9** | 92.2 / 87.8 | 86.7 / **83.3** |
| spaCy ru_core_news_sm | 85.7 / 81.8 | 76.6 / 71.5 | 79.5 / 75.8 | 90.1 / 86.2 | 71.3 / 65.1 |
| Stanza | 93.0 / 91.2 | 81.5 / 77.6 | 81.4 / 78.0 | 92.7 / 88.2 | 77.9 / 73.4 |
| Stanza accurate | 94.6 / **93.1** | 84.0 / 80.5 | 84.6 / 81.2 | 93.8 / **89.5** | 80.1 / 76.4 |
| Slovnet (Natasha) | 79.7 / 75.8 | 68.2 / 63.0 | 72.9 / 68.8 | 86.5 / 82.1 | 59.5 / 53.0 |
| UDPipe 1 | 86.4 / 80.8 | 72.9 / 66.8 | 75.6 / 68.2 | 84.9 / 78.3 | 67.9 / 61.0 |

An error in splitting words and sentences brings parse errors with it: on Taiga and GSD, with many abbreviations, numbers and social media texts, the other parsers lose more here than ruSyntax.

??? note "Splitting into words and sentences"

    Token / sentence F1, %:

    | System | SynTagRus | Taiga | GSD | PUD | Poetry |
    | :----- | --------: | ----: | --: | --: | -----: |
    | ruSyntax (anySeg) | 99.6 / 98.9 | 99.1 / 89.9 | 99.1 / 96.3 | 99.8 / 99.2 | 99.6 / 95.9 |
    | spaCy ru_core_news_sm | 98.4 / 95.4 | 96.8 / 80.4 | 94.7 / 85.8 | 98.5 / 96.5 | 98.3 / 94.2 |
    | Stanza | 99.8 / 98.6 | 95.5 / 88.7 | 91.6 / 95.5 | 99.5 / 99.2 | 95.2 / 88.2 |
    | Stanza accurate | 99.7 / 98.8 | 95.8 / 87.8 | 92.8 / 96.0 | 99.6 / 99.2 | 95.5 / 74.8 |
    | Slovnet (Natasha, razdel) | 98.7 / 98.4 | 95.8 / 89.8 | 92.6 / 91.3 | 99.3 / 99.0 | 99.2 / 92.2 |
    | UDPipe 1 | 99.7 / 98.5 | 94.0 / 87.0 | 92.5 / 95.0 | 97.3 / 98.8 | 94.5 / 88.6 |

### Gold tokens { #tokens }

The systems get the gold words and sentence boundaries and add the morphology themselves (ruSyntax - the ruMorph tags): this shows the parsing quality without splitting errors. UAS / LAS, %:

| System | SynTagRus | Taiga | GSD | PUD | Poetry |
| :----- | --------: | ----: | --: | --: | -----: |
| ruSyntax | 91.0 / 88.2 | 86.5 / 83.1 | 90.1 / 86.0 | 92.6 / 88.1 | 87.7 / **84.3** |
| spaCy ru_core_news_sm | 87.0 / 82.9 | 80.1 / 74.3 | 84.5 / 80.1 | 91.5 / 87.3 | 73.3 / 66.7 |
| Stanza | 93.4 / 91.6 | 85.8 / 81.5 | 87.0 / 82.8 | 93.5 / 88.9 | 83.0 / 78.5 |
| Stanza accurate | 95.1 / **93.6** | 88.5 / **84.5** | 90.2 / **86.2** | 94.5 / **90.1** | 87.8 / 84.0 |
| Slovnet (Natasha) | 80.8 / 76.7 | 72.4 / 66.2 | 78.8 / 74.3 | 87.2 / 82.6 | 60.2 / 53.5 |
| UDPipe 1 | 87.0 / 81.3 | 78.1 / 71.0 | 80.0 / 71.9 | 87.3 / 80.4 | 71.5 / 64.5 |

Slovnet picks the head of every word separately and does not always give a tree: in 29 % of the SynTagRus sentences the extra roots and cycles are attached to the root before scoring.

## Speed { #speed }

Parsing the gold tokens of SynTagRus test (157,706 words) in one thread, the best of two runs without loading; ruSyntax includes the ruMorph tags. Loading is the import and the creation of the parser with the parse of the first sentence; memory is the peak of the process.

| System | Words/s | Loading, s | Memory, MB |
| :----- | ------: | ---------: | ---------: |
| ruSyntax | 29,526 | **0.08** | **160** |
| spaCy ru_core_news_sm | 10,368 | 2.00 | 1,007 |
| UDPipe 1 | 6,318 | 1.73 | 467 |
| Stanza | 1,098 | 2.84 | 6,638 |
| Stanza accurate | 428 | 4.66 | 5,993 |
| Slovnet (Natasha) | 58,583 | 0.27 | 360 |

Slovnet is faster, but picks heads without a tree, gives no morphology and is 5 to 31 points behind in LAS. Stanza accurate was measured on the first 20,006 words. The memory peak depends on the amount of text per call: every system except Stanza accurate got all of SynTagRus test in one call, so its peak is not comparable with the others. Measured on an Apple M3 Pro, Python 3.11, load average 2-3.

## Reproducing { #reproduce }

The `scripts/compare.py` script downloads the test parts of the treebanks, the CoNLL 2018 evaluation script and the models of the other parsers with a SHA-256 check (the Stanza models with its downloader and a check of the file sums), parses the treebanks with every system and computes the metrics, measures the speed, loading and memory in a separate process per system, and writes the tables. The package versions are pinned in `scripts/compare.py.lock`; ruSyntax comes from a wheel built from the repository, and ruMorph and anySeg are pinned in the command (their versions are written to the results):

``` bash
rm -rf target/compare/wheels
uvx maturin@1.15.0 build --release --locked -o target/compare/wheels
uvx uv@0.13.5 run --with target/compare/wheels/rusyntax-*.whl --with rumorph==0.6.0 --with anyseg==0.4.0 \
    scripts/compare.py target/compare
```
