# The rusyntax command

``` bash
rusyntax [options] [file ...]
```

The command parses text from files or the standard input and writes [CoNLL-U](https://universaldependencies.org/format.html): for every word, its lemma, part of speech and features, its head and its [Universal Dependencies](https://universaldependencies.org/) relation. It comes with the package (`pip install rusyntax`) and also runs as `python -m rusyntax`. The [anySeg](https://github.com/SergeyShk/anySeg) segmenter splits the text into sentences and words, and the [ruMorph](https://github.com/SergeyShk/ruMorph) morphological analyzer gives the parts of speech and features - the ones installed with the package - so the result is that of [`parse_text`](parser.md#parse_text), and so is the speed: one thread, about 29 thousand words per second with the morphology ([Comparison](performance.md)).

``` bash
echo "Мама мыла раму." | rusyntax
```

```
# newdoc
# newpar
# sent_id = 1
# text = Мама мыла раму.
1	Мама	мама	NOUN	_	Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing	2	nsubj	_	_
2	мыла	мыть	VERB	_	Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act	0	root	_	_
3	раму	рама	NOUN	_	Animacy=Inan|Case=Acc|Gender=Fem|Number=Sing	2	obj	_	SpaceAfter=No
4	.	.	PUNCT	_	_	2	punct	_	_
```

## Options { #options }

| Option | Default | Description |
| :----- | :------ | :---------- |
| `-i`, `--input` | `text` | [Input](#input): `text`, `conllu` |
| `-m`, `--model` | the model of the package | A ruSyntax model file, as in `Parser(path)` |
| `-o`, `--output` | the standard output | The result file; `-` is the standard output |
| `-h`, `--help` | | Show the help (in Russian) |
| `-V`, `--version` | | Show the versions of ruSyntax, anySeg and ruMorph |

Files are read one after another; without files and in place of `-` the standard input is read. The input is in UTF-8: a file in another encoding (cp1251, UTF-16) is not parsed, and a single line with wrong bytes is parsed with U+FFFD in their place. An option value is written separately or attached: `-i conllu`, `-iconllu`, `--input=conllu`; a file name starting with a hyphen goes after `--`. Text is parsed paragraph by paragraph and CoNLL-U sentence by sentence: the result is written as soon as a blank line or the end of the input closes the paragraph or the sentence - to a terminal at once, and to a pipe or a file whenever the command waits for the next line of input; it can be kept open and fed text paragraph by paragraph.

In Windows PowerShell 5.1 the `>` redirection and a pipeline (`Get-Content text.txt | rusyntax`) corrupt Cyrillic: pass the file as an argument and write the result with the `-o` option (in PowerShell 7 they work too). Text passed with `echo` in cmd comes in the console encoding: run `chcp 65001` first. Neither cmd nor PowerShell expands file patterns (`*.txt`).

## Input { #input }

- `text` - text: a blank line ends a paragraph. The anySeg segmenter finds the tokens and the sentences by the Universal Dependencies conventions the ruMorph and ruSyntax models are trained on; a sentence may go across a line break. A paragraph without blank lines longer than 256 KB (a file with a sentence per line, a single long line) is cut by anySeg into pieces at sentence boundaries: the memory does not grow with the input, and the sentence numbers and the paragraph go on across the pieces; a sentence longer than 256 KB is cut.
- `conllu` - [CoNLL-U](https://universaldependencies.org/format.html) with parts of speech (UPOS) and features (FEATS): the HEAD and DEPREL columns are filled, and the other columns, the comments, multiword tokens and empty nodes stay as they are (the DEPS column does not change either). A part of speech is one of the 17 UD parts of speech in capitals. A sentence where no word has a part of speech goes through ruMorph first: LEMMA, UPOS and FEATS are filled, as by the `rumorph -i conllu` command; a part of speech on some words of a sentence but not on all is an error. The file is read as by `anyseg.read_conllu`: the word IDs go in order, a multiword token has all its words, the forms agree with the comment `text` if there is one, and a sentence has words - this is stricter than `rumorph -i conllu`. At a wrong line parsing of the file stops: the sentences before it are written, and the next files are parsed.

``` bash
printf '1\tМама\t_\tNOUN\t_\tCase=Nom\t_\t_\t_\t_\n2\tспит\t_\tVERB\t_\t_\t_\t_\t_\t_\n' | rusyntax -i conllu
```

```
1	Мама	_	NOUN	_	Case=Nom	2	nsubj	_	_
2	спит	_	VERB	_	_	0	root	_	_
```

The model is trained on the ruMorph annotation: with gold parts of speech and features LAS is about 1 point higher, and with the tags of other taggers lower (on test: Stanza 0.7, UDPipe 1 4, spaCy 6 points lower); the features should be complete. To replace the morphology of a CoNLL-U file with the ruMorph annotation and parse it, pipe the commands:

``` bash
rumorph -i conllu -f conllu gold.conllu | rusyntax -i conllu > parsed.conllu
```

## Output { #output }

CoNLL-U as anySeg writes it. For the `text` input it is that of the `rumorph -f conllu` command with HEAD and DEPREL filled: a file starts with the comment `newdoc`, a paragraph with `newpar`, a sentence has the comments `sent_id` (numbered across all the files) and `text`; MISC has `SpaceAfter=No` on a token with no space after it in its paragraph and `SpacesAfter` where the space is not a single one; the indent of a paragraph is `SpacesBefore` of its first token (the rumorph 0.7 command does not write the indent). For the `conllu` input a sentence is written as it was read, with the new heads and relations. Lines end with LF, the text is in UTF-8.

## Exit codes { #exit-codes }

| Code | When |
| :--- | :--- |
| 0 | Success, also when the output is closed before the end (`rusyntax text.txt \| head`) |
| 1 | An input file is not found, cannot be read or is not in UTF-8, a line is not in UTF-8 (it is parsed with U+FFFD), a CoNLL-U line is wrong or a part of speech is unknown; the sentences before the error are parsed, and so are the other files |
| 1 | The model cannot be read, the output file cannot be created or is an input file or the model file; nothing is parsed |
| 1 | A write error; parsing stops |
| 2 | Wrong options |

Error messages, in Russian, go to the standard error.
