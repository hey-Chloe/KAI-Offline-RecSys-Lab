# Criteo CTR complete pinned-shard scale V2

Status: `COMPLETE`; public offline only; online claim: `false`.

## Protocol

- Complete pinned parquet shard rows: 766,864.
- Seeds: 3407, 6502, 9109.
- Same rows, 13 numeric + 26 categorical fields, train-fitted preprocessing, and dev checkpoint rule.
- Source-order split; parquet conversion order is not claimed chronological.

## Test metrics — mean ± population std

| Model | ROC-AUC | PR-AUC | LogLoss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| LR | 0.672563 ± 0.000358 | 0.064850 ± 0.000053 | 0.143133 ± 0.000021 | 0.030878 ± 0.000001 | 0.029273 ± 0.000025 |
| DeepFM | 0.730783 ± 0.001799 | 0.102801 ± 0.000900 | 0.128356 ± 0.000225 | 0.029408 ± 0.000027 | 0.003539 ± 0.000774 |
| DCNv2 | 0.709836 ± 0.002239 | 0.084471 ± 0.000690 | 0.131085 ± 0.000038 | 0.029712 ± 0.000030 | 0.003795 ± 0.001308 |

## Limitations

- This run uses every row in one pinned official parquet shard, not the complete 1TB or 24-day corpus.
- The parquet conversion order was not independently audited, so source-order blocks are not described as chronological.
- Criteo states positive and negative examples were subsampled at different rates; raw probabilities are not population CTR without correction.
- Hashed categorical feature semantics are undisclosed.
- Offline public-data metrics do not establish online CTR lift, revenue lift, or production performance.
- CC BY-NC-SA 4.0 restricts use to noncommercial research under its terms.
