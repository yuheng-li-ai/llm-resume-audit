# OSF pre-registration form — "When Algorithms Hire — Midterm Audit"

Paste each field below into the corresponding section of the OSF pre-registration form (OSF Standard Pre-Registration). Replace `{{ DOI }}` after locking.

---

## Title

**When Algorithms Hire: A Factorial Audit of Demographic Bias in LLM Résumé Screeners (Midterm)**

## Authors

Yuheng Li (sole author).

## Hypotheses (verbatim from proposal §2)

The study addresses one primary and two secondary questions.

**Primary.** Conditional on identical qualifications, does the inferred demographic identity of an applicant — encoded only through the applicant's name (and optionally photo) — causally shift the hiring score that an LLM résumé screener assigns?

**Secondary 1.** Does the magnitude of the demographic effect differ across job categories? In particular, is it larger in stereotypically gendered occupations?

**Secondary 2.** Does the explicit presence of objective qualification signals (numeric GPA, standardized-test scores, certification IDs) attenuate the demographic effect?

Formal hypotheses:

- H₁: E[Y | do(T_g = male)] > E[Y | do(T_g = female)] (holding qualifications constant)
- H₂: τ^CATE(J = tech) ≠ τ^CATE(J = care)
- H₃: |τ(· | objective signals present)| < |τ(· | absent)|

## Design

Four-factor stratified randomized factorial on 8 000 résumé-treatment cells, scored by each of 2 LLMs (16 000 hiring scores total):

| Factor | Levels | Count |
|---|---|---|
| T_g (gender, signalled by name) | male, female | 2 |
| T_e (ethnicity, signalled by name) | White, Black, Hispanic, Asian | 4 |
| T_p (age signal) | early-career, mid-career, late-career | 3 |
| S (objective-qualifications block) | present, absent | 2 |
| J (occupation, moderator) | 18 occupations, balanced 3×3×2 panel | 18 |
| M (model, moderator) | GLM 5.1, GLM 4.5 (Zhipu paid API) | 2 |

The 18-occupation panel is balanced on stereotype × skill tier: stereotype ∈ {male, female, neutral} × tier ∈ {high, mid, low}, two occupations per cell.

## Sample size justification

Stratified sample of 8 000 résumé-treatment cells (~9 replicates per micro-cell). Cohen 1992 power calc for two-sample t-test at α = 0.05, σ = 15: n = 4 000 per gender condition gives MDE d ≈ 0.063 at 80% power.

## Estimators (verbatim from proposal §6)

**ATE (§6.1).** OLS with cluster-robust standard errors clustered on `resume_id`:

    Y_{ijm} = α + β_g·T^g_i + β_e^⊤·T^e_i + β_p^⊤·T^p_i + β_S·S_i + δ_j + μ_m + ε_{ijm}

Baselines locked: T_g = male, T_e = White, T_p = early_career, S = absent.

**CATE (§6.2).** Generalised random forest (Athey, Tibshirani, Wager 2019; econml.grf.CausalForest) with X = [J, M, S, years_exp_bracket, education_tier]. T = each demographic axis vs the locked baseline, one CausalForest per non-baseline level (6 forests total). Report 1ˢᵗ / 50ᵗʰ / 99ᵗʰ percentile of τ̂(x) + variable-importance ranking.

**Multiple-hypothesis correction (§6.3).** Romano-Wolf / List, Shaikh, Xu (2019) step-down family-wise error correction at α = 0.05, applied to 21 pre-specified contrasts: 7 demographic main effects × {full panel, glm-5, glm-4.5}. Cluster bootstrap over `resume_id`, B = 1 000 replications.

## Robustness (§7)

- Within-cluster permutation placebo (B = 500): treatment labels shuffled within each resume_id; placebo coefficient distribution must centre on zero.
- Within-vendor model contrast: GLM 5.1 vs GLM 4.5, holding training-corpus composition fixed.
- Note: open-vs-closed and Western-vs-Chinese contrasts originally on the roster are no longer estimable under the trimmed two-model panel (see proposal §12).

## Stopping rule

Sample size frozen at 8 000 cells × 2 models = 16 000 scores. No early stopping; no top-up beyond the locked total.

## Pre-specified analysis code

Analysis code is version-controlled. Code commit hash + input-data SHA-256 are attached as `osf_evidence.json` (see file in OSF storage).

## Materials uploaded to OSF storage

- `treatment_assignments.parquet` — 8 000 design cells (SHA-256 in `osf_evidence.json`)
- `base_resumes.parquet` — 450 base résumés
- `job_descriptions.parquet` — 54 job descriptions
- `osf_evidence.json` — commit hash + parquet hashes + generation timestamps
- `osf_deviation_note.md` — disclosure of pre-registration timing

## Pre-registration timing — DEVIATION DISCLOSURE

Data collection completed **prior to** OSF lock. The audit instrument (`scores.parquet`) was produced over the 2026-04-28 → 2026-05-03 window; OSF lock occurs at submission of this form. SHA-256 hashes of `scores.parquet` and `treatment_assignments.parquet` are attached so any third party can verify that no observation was modified post-lock.

The analysis protocol (OLS specification, GRF CATE setup, LSX correction, contrast list) was implemented in commits `5954fdd` (Phase 7.1), `5fb3562` (Phase 7.2), `4912510` (Phase 7.3) — each commit pre-dates a fresh look at the demographic coefficients on the locked data. See `osf_deviation_note.md` for the full timeline.

The reason for the delay is operational: the original CHECKLIST had Phase 6 (OSF) gating Phase 5.5 (main batch), but the author ran Phase 5.5 first due to a misread of the dependency. This deviation is disclosed here, in `osf_deviation_note.md`, and as a §12 limitation in the final proposal PDF.
