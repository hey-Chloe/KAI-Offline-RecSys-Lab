# Criteo Attribution CVR / ESMM source audit

Audit status: `ELIGIBLE_AND_EXECUTED_FOR_NONCOMMERCIAL_OFFLINE_RESEARCH`

## Publisher and artifact identity

- Publisher page: <https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/>
- Publisher-owned repository: <https://huggingface.co/datasets/criteo/criteo-attribution-dataset>
- Pinned repository revision: `904188a63cbad78bee43cd26ff5ee4ac77903986`
- File: `criteo_attribution_dataset.tsv.gz`
- Publisher repository SHA-256: `94ac7a465564349bc7ba008602211d5990a3c53cc133abc0aadef61ea2391a98`
- Locally downloaded file SHA-256: `94ac7a465564349bc7ba008602211d5990a3c53cc133abc0aadef61ea2391a98`
- Local compressed bytes: `653015824`
- License identified by the publisher repository: `CC-BY-NC-SA-4.0`; use is restricted here to attributed, noncommercial offline research.

The historical `go.criteo.net` download URL currently returns HTTP 404. The
replacement artifact was downloaded from the `criteo` publisher organization,
not a third-party mirror, and pinned by repository revision plus file hash.

## Schema gate

The publisher documents one timestamp-sorted row per displayed impression. The
same row includes `click`, `conversion`, campaign/context features and the
30-day conversion definition. This passes the impression-funnel gate that the
clicked-only Sponsored Search Conversion Log failed.

Full-file streaming audit:

- impressions: `16,468,027`;
- click-positive rows: `5,947,563`;
- raw conversion-positive rows: `806,196`;
- distinct non-negative `conversion_id` values among those rows: `435,810`;
- `click × conversion` positive rows: `806,196`;
- raw conversion with click=0: `0` in this published artifact.

The conversion-positive total above is a row count, not a count of distinct
conversion events; the distinction is retained because multiple impression
rows may reference a conversion timeline.

The V1 run uses the first `600,000` physical timestamp-sorted rows without
label-conditioned sampling. The raw file remains excluded from Git.

## Frozen label definition

- CTR label: `click` on all impressions.
- CTCVR label: `click × conversion` on all impressions.
- Post-click CVR label: `conversion` evaluated only where `click=1`.

The publisher says conversion means an outcome within 30 days after an
impression, independently of last-click attribution. Even though this artifact
contains no `conversion=1, click=0` rows, the pipeline still constructs CTCVR
explicitly and records excluded view-through counts rather than assuming the
raw label is universally post-click.

## Feature leakage audit

Allowed model features:

- `campaign`, `cat1`–`cat9`;
- transformed `cost`;
- transformed `time_since_last_click` plus an explicit prior-click indicator.

Excluded from model features:

- identity shortcut: `uid`;
- labels: `click`, `conversion`;
- outcome-derived fields: `conversion_timestamp`, `conversion_id`,
  `attribution`, `click_pos`, `click_nb`, `cpo`.

Preprocessing is fitted on train only. Splits are fixed contiguous timestamp
blocks. Dev selects checkpoints; test is not used for model or epoch selection.

## Claim boundary

This audit authorizes only the reported public offline comparison. It does not
establish production CVR, online lift, revenue impact, a real advertising
system, or eligibility for commercial use.
