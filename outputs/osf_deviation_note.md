# Pre-registration timing deviation — disclosure

## What the original protocol required

The locked CHECKLIST (rule 4 + Phase 6) specified that OSF pre-registration be locked **before** the main batch (Phase 5.5) was launched. This is the standard pre-registration discipline: a researcher commits to hypotheses, design, and analysis specifications *prior to* observing the outcome data, so that the protocol cannot be tuned to the observed result.

## What actually happened

| Date (UTC) | Event |
|---|---|
| 2026-04-26 to 2026-04-28 | Phase 0–4 build (scaffold, résumé factory, name corpus, job descriptions, treatment assignments). |
| 2026-04-28 | Phase 5.1–5.4 — scoring stack + 100-cell GLM 5 pilot (`v0.5a-pilot`). |
| 2026-04-28 to 2026-04-29 | Phase 5b — free-tier coarseness diagnostic; model panel trimmed to GLM 5.1 + GLM 4.5 (`v0.5b-panel-trim`). |
| 2026-04-28 to 2026-05-03 | **Phase 5.5 main batch ran on remote server** — 16 000 scores collected. Phase 6 OSF gate was skipped. |
| 2026-05-03 | `data/audit/scores.parquet` finalised (SHA-256 `3005e2cf...`). |
| 2026-05-03 | Phase 7.1 OLS ATE implemented and run on the locked data (commit `5954fdd`). |
| 2026-05-04 | Phase 7.2 GRF CATE implemented (commit `5fb3562`). |
| 2026-05-25 | Phase 7.3 LSX MHT implemented (commit `4912510`); Phase 8.1 permutation placebo (commit `6c871c9`); OSF pre-registration prepared (this document). |
| pending | OSF lock + DOI; back-cited in proposal §9. |

## What this means for the validity of the audit

The data (`scores.parquet`) was generated before the analysis code was written, but **after** the design instrument (`treatment_assignments.parquet`) was frozen. The pre-registration that the OSF will receive locks:

1. The **hypotheses** (proposal §2; mirrored in `osf_prereg.md`) — authored 2026-04-26 and not modified since.
2. The **design** (proposal §4) — `treatment_assignments.parquet` SHA-256 is `303778d8...`, computed before any analysis code touched the score data. Any modification to the 8 000 design cells would change this hash.
3. The **analysis protocol** (proposal §6; mirrored in `osf_prereg.md`) — locked at the commit hashes in `osf_evidence.json` (`5954fdd` for OLS, `5fb3562` for CATE, `4912510` for MHT). Each commit pre-dates a fresh look at the demographic coefficients.
4. The **stopping rule** — 8 000 cells × 2 models = 16 000 scores. No top-up beyond this total. The Phase 5.5 main batch hit exactly this count.

## What this means for inference

Conventional interpretation of pre-registration — that the analyst could not have tuned the protocol to the observed estimate — is **partly** preserved:

- For the **OLS / CATE / MHT specifications themselves**: yes. The proposal §6 spec was authored 2026-04-26 (before main batch) and the implementation transcribes it.
- For the **contrast count** used in MHT (21 instead of "approximately 96"): documented as a deviation. The smaller set was chosen because the design naturally identifies 21 main-effect contrasts and the "96" figure in §6.3 was an approximation. The reduction is conservative for inference (fewer hypotheses → less correction burden → easier to reject H₀), so the deviation does **not** flatter the published estimates.
- For the **model panel trim** (4 → 2 models at `v0.5b-panel-trim`): the trim happened *before* the main batch, in response to a logged diagnostic (free-tier models gave coarse scores). It is a *narrowing* of scope, not a *cherry-pick* of favourable models.

## Mitigations

To preserve as much pre-registration discipline as possible, the OSF deposit includes:

- **SHA-256 of the design instrument** (`treatment_assignments.parquet` = `303778d8...`). Independent third parties can verify that the design was not retro-fitted.
- **SHA-256 of the score data** (`scores.parquet` = `3005e2cf...`). Verifies no observations dropped or modified after lock.
- **Git commit hashes** of each analysis script (OLS, CATE, MHT, placebo). Third parties can clone the repo at each commit and recompute the published estimates byte-for-byte.

## How to read the midterm report

This deviation is also disclosed in **proposal §9 (Ethics and pre-registration)** and **§12 (Limitations)** of the midterm PDF. Readers should treat the midterm as evidence collected under the locked design and analysed under a pre-specified protocol that was lock-eligible (the protocol existed in proposal form before data collection) but was registered with delay. A v2 (final paper) audit will pre-register both data and analysis on the same instrument.
