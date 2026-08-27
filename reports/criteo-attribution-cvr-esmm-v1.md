# Criteo Attribution CVR / ESMM public benchmark V1

Status: `COMPLETE`; data origin: `public`; online claim: `false`.

## Frozen protocol

- Rows: 420,000 train / 90,000 dev / 90,000 test, in timestamp order.
- Test labels: 32,821 clicks; 4,488 click-and-conversion rows; 0 non-click conversion rows excluded from CTCVR.
- Naive baseline: independent CTR on all impressions plus CVR on clicked impressions only.
- ESMM: joint entire-space CTR and CTCVR supervision; inferred CVR is evaluated only on clicked test rows.
- Preprocessing is fitted on train only; model/epoch selection uses dev only; test is scored once by the frozen protocol.
- Seeds: 3407, 6502, 9109.
- Config SHA-256: `2660bf2f3dd74cf3bcc0c28d447bd97f1da9a55e8916ab7c506ecf012c385367`.
- Split SHA-256: `356683bb0c004f4400bc97f41c0507416a654109ff8514d839c3b11b95bfdd65`.
- Result SHA-256: `1ae667d2e7b2d0652a7c832234b128f04a88a78c307bf8b85afcf83a486242b0`.

## Test metrics — mean ± population std

| Model | Task | ROC-AUC | PR-AUC | LogLoss | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| Naive independent | CTR | 0.712914 ± 0.000424 | 0.590558 ± 0.000743 | 0.589334 ± 0.000148 | 0.201500 ± 0.000057 | 0.023787 ± 0.004292 |
| Naive independent | CTCVR | 0.825770 ± 0.005550 | 0.292083 ± 0.002668 | 0.156545 ± 0.000936 | 0.040531 ± 0.000071 | 0.002967 ± 0.000719 |
| Naive independent | Post-click CVR | 0.813043 ± 0.007833 | 0.513684 ± 0.008337 | 0.304217 ± 0.002922 | 0.088549 ± 0.000601 | 0.015130 ± 0.002443 |
| ESMM | CTR | 0.711015 ± 0.001137 | 0.588003 ± 0.001657 | 0.590571 ± 0.001033 | 0.202015 ± 0.000408 | 0.023788 ± 0.005041 |
| ESMM | CTCVR | 0.841554 ± 0.002555 | 0.310462 ± 0.004102 | 0.152730 ± 0.000693 | 0.039921 ± 0.000136 | 0.002489 ± 0.000486 |
| ESMM | Post-click CVR | 0.834856 ± 0.002509 | 0.535438 ± 0.004471 | 0.294407 ± 0.001031 | 0.086227 ± 0.000286 | 0.013049 ± 0.000621 |

## Observed comparison

- CTCVR: ESMM changes ROC-AUC by `+0.015784`, PR-AUC by `+0.018380`, and LogLoss by `-0.003815` versus the independent baseline.
- Post-click CVR: ESMM changes ROC-AUC by `+0.021812`, PR-AUC by `+0.021754`, and LogLoss by `-0.009810`.
- CTR: ESMM changes ROC-AUC by `-0.001899`, PR-AUC by `-0.002555`, and LogLoss by `+0.001237`; the CTR task does not improve in this run.

The conversion-task gains and CTR regression are both retained. These multi-seed means are descriptive; this benchmark does not claim statistical significance or causal online lift.

These are descriptive public offline results. A metric improvement, if any, is not an online CTR/CVR or revenue claim.

## Label boundary

The publisher's raw `conversion` label can include a conversion within 30 days even when the current impression was not clicked. This experiment defines CTCVR as `click × conversion`; it does not relabel view-through conversions as post-click outcomes.

## Limitations

- This is a fixed 600000-row prefix, not the full 16468027-row public dataset.
- The publisher's conversion label means a conversion within 30 days after an impression independently of last-click attribution; this benchmark defines CTCVR explicitly as click multiplied by conversion.
- The fixed prefix contains repeated campaign/user timelines and is not an IID random sample; the split is deliberately temporal and may expose distribution shift.
- User identity and all outcome-derived fields are excluded, but anonymized categorical meanings are undisclosed.
- The independent baseline and ESMM are compact CPU/MPS-scale research implementations, not production serving models.
- Offline public-data metrics do not establish KAI production, online lift, conversion lift, revenue, or business performance.
- CC BY-NC-SA 4.0 restricts use to noncommercial purposes under its terms.
