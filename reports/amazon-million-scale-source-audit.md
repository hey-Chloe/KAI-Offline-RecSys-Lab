# Amazon Reviews'23 Million-Item Source Audit

## Decision

The scale corpus uses two unmodified public metadata archives published by McAuley Lab: `Industrial_and_Scientific` and `Amazon_Fashion`. Local streaming inspection found **1,253,672 rows and 1,253,672 unique `parent_asin` values**. No malformed rows, missing IDs, cross-file duplicates, copied vectors, or synthetic expansion were observed.

## Immutable local evidence

| Category | Rows / unique items | Compressed bytes | SHA-256 |
|---|---:|---:|---|
| Industrial_and_Scientific | 427,564 | 281,523,932 | `0beb251cec166347a3ec3ef23e55ec89f7fb27a6e8e9a0737d6b34cdc184ebcb` |
| Amazon_Fashion | 826,108 | 224,299,124 | `0b121c7494b0216ba3bf80adce9c79286fe08f14086966f841fe6716e1a24b73` |
| Combined | 1,253,672 | 505,823,056 | catalog-order digest `eafac62d05f6019fae758ef718d7fb60b369daf92b7d4d4a418d80a305a009a2` |

The official project page describes `parent_asin`, title, main category, and category hierarchy and publishes direct metadata download links. It also notes that reported item statistics are review-derived and that some reviewed items can lack metadata. This experiment counts the downloaded metadata rows directly rather than copying the website's headline counts.

## License and use boundary

The provider has explicitly stated that it is not in a position to assign a dataset license or dictate usage terms, and that the dataset is made available primarily for research. Therefore:

- use is local, research-only, and non-commercial;
- applicable legal and ethical obligations remain with the user;
- raw metadata rows are not committed or redistributed;
- generated embeddings, checkpoint, and HNSW index remain ignored local artifacts;
- the public report contains only aggregate counts, cryptographic evidence, system metrics, and limitations.

Official documentation: <https://amazon-reviews-2023.github.io/main.html>

Provider license clarification: <https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1>

## Model compatibility audit

The frozen V2 item tower concatenates an item-ID embedding with title and category features. Its trained ID vocabulary contains only the original 25,754-item Industrial_and_Scientific training catalog, so it cannot assign genuine learned IDs to the million-item corpus.

The scale pipeline therefore gives every new product the same UNK item ID and allows only title/category metadata to vary. This is a valid cold-start embedding path for measuring encoding and ANN index scalability, but it does **not** establish recommendation quality for the million-item catalog. The frozen checkpoint is loaded by SHA-256 and is not retrained.
