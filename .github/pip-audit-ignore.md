# pip-audit ignore list: reachability analysis

Scope: `nltk==3.10.3`. Of the six nltk advisories that the advisory database
lists as "no fixed version" against 3.9.2 (PYSEC-2026-98, -97, -597, -3740,
-3955, -4318), the installed 3.10.3 clears five; `pip-audit` on the installed
wheel reports only the one below. The analysis of the other five is kept here
for the record, because the database entry for a version range can change.

pamola-core uses NLTK only through `pamola_core/utils/nlp/` and only for:
`nltk.tokenize.word_tokenize`, `nltk.stem` (WordNet lemmatizer, Snowball),
`nltk.corpus.stopwords`, `nltk.collocations`, and `nltk.data.find` with
**fixed literal** resource paths (`tokenizers/punkt`, `corpora/wordnet`,
`corpora/stopwords`). Re-verify with `grep -rn nltk pamola_core` when touching
the NLP package.

## Ignored (listed in `pip-audit-ignore.txt`)

| ID | Component | Needs | Reached by pamola-core? |
| --- | --- | --- | --- |
| PYSEC-2026-3740 (CVE-2026-81726) | model-artifact save/load helpers (`TransitionParser.train/parse`, `AveragedPerceptron.save/load`, `PerceptronTagger.save_to_json`, `save_maxent_params`) use raw `open()` and bypass the NLTK path sandbox | calling one of those helpers with a caller-controlled model path | No. None of them is imported or called anywhere in pamola-core. |

## Fixed by 3.10.3 (no ignore entry; analysis kept for the record)

| ID | Component | Reached by pamola-core? |
| --- | --- | --- |
| PYSEC-2026-98 (CVE-2026-0847) | `CorpusReader` path traversal | No: no `CorpusReader` is instantiated with a caller path; `nltk.corpus.stopwords` uses NLTK's own loader with a fixed corpus name. |
| PYSEC-2026-97 (CVE-2026-0846) | `nltk.util.filestring()` | No: never imported or called. |
| PYSEC-2026-597 (CVE-2026-12243) | `nltk.data.load/find`, `..%2f` bypass | No for data content: every `nltk.data.find` uses a literal path. The tokenizer `language` reaches `word_tokenize(language=...)` (resolves `tokenizers/punkt_tab/<language>`); it comes from operator configuration or `langdetect` ISO codes, never from dataset values. |
| PYSEC-2026-3955 (CVE-2026-12372) | `validate_network_url` misses `100.64.0.0/10` | No: pamola-core never passes NLTK a URL. |
| PYSEC-2026-4318 (CVE-2026-12259) | downloader extracts before checksum check | Only on explicit operator opt-in: since SEC-01 the NLP package is offline by default (`remote_downloads_allowed()` false), downloads use literal names, and the CI job installs corpora out of band. |

Conclusion: the single ignored advisory is not reachable from pamola-core. Revisit on every nltk release.
