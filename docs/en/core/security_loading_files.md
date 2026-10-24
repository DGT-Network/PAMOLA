# Loading files: security policy

PAMOLA-CORE reads files that come from outside the process: datasets, mapping files, configuration. A file format that can carry executable content turns "read this file" into "run this code". This page states which formats CORE loads, which it refuses by default, and what you must do when you accept the risk.

## Policy

1. **Formats that cannot execute code** are loaded without restriction: CSV, JSON, Parquet, Excel, and the encrypted PAMOLA mapping format (authenticated AES-256-GCM frames; a wrong key or a modified file fails closed with `MappingIntegrityError`).
2. **Pickle is never loaded implicitly.** Unpickling runs code contained in the file. Every library call that would unpickle requires `allow_unsafe_pickle=True`; without it the call raises `UnsafeDeserializationError` before the file is opened. With it, the call logs a warning and emits a `RuntimeWarning`.
3. **Pickle is for files you produced yourself or fully trust.** A file that came from a user, a customer, a download or a shared folder is not trusted. Convert it to a safe format at the source.
4. **Writing pickle is not restricted**, but the written file inherits the same rule on the way back in. Pickle files hold source values in clear text; they are not a protected artifact (see the artifact policy).

## Where the rule applies

| Call | Behaviour |
|---|---|
| `pamola_core.utils.io.read_dataframe(path)` with `file_format="pickle"` or a `.pkl`/`.pickle` path | raises `UnsafeDeserializationError` unless `allow_unsafe_pickle=True` |
| `MappingStore.load(path)` with a `.pkl`/`.pickle` path or `format="pickle"` | raises unless `allow_unsafe_pickle=True` |
| `MappingStore.load_pickle(path)` | raises unless `allow_unsafe_pickle=True` |
| `ConsistentMappingPseudonymizationOperation` mapping files | encrypted PAMOLA format only; no pickle path |

The rest of the package contains no `pickle.load`, `joblib.load`, `torch.load` or `numpy.load(allow_pickle=True)` on a user-supplied path. This is checked by a repository grep recorded in the release notes; keep it that way in new code.

## Expressions in configuration

A few operations evaluate an expression string given in their configuration (for example `AddOrModifyFieldsOperation` expressions, aggregation expressions, the MVF parser of correlation analysis). Treat operation configuration as trusted code: do not build it from untrusted input, and do not accept configuration files from untrusted sources.
