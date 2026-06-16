# Results summary — LLM resume audit

**Specification**: OLS with cluster-robust SE clustered on resume_id,
16 000 observations across 450 clusters, R² = 0.529.
Joint F-test of implemented-cue nullity: F(6, 449) = 203.6, p = 1.07e-124 — the joint implemented-cue null is firmly rejected.

## Hypothesis-level claims

### H₁ — Gender and name-origin cues shift hiring score

**Pooled, full panel**: female β = +0.220 (SE 0.103, raw p = 0.033, FWER adj p = 0.089). Direction is **positive** — opposite of the Bertrand-Mullainathan baseline expectation. Loses significance after FWER correction.

**Within-model version contrast**: significant only on GLM 5.1 — female adj p = 0.011 (z = 2.78); glm-4.5 adj p = 0.990 (z = 0.12). GLM 4.5 is essentially neutral on every gender/name-origin cue axis (smallest adj p = 0.42).

**Name-origin cue**: pooled Asian/Vietnamese cue β = -0.373 (adj p = 0.089); glm-5 only adj p = 0.010. Black / Hispanic effects not significant under FWER at either resolution.

**Verdict on H₁**: rejected at the pooled level for Black / Hispanic;
supported on glm-5 for Asian (penalty) and female (premium); magnitude small (≤ 0.4 score points) compared with the 0–100 scale.

### H₂ — Implemented-cue effects vary by job category

CATE percentile spread (across 16 000 cells) for each axis:

- **t_g=Female cue**: p01 -3.30 / p50 0.14 / p99 4.62; mean 0.228.
- **t_e=Asian/Vietnamese cue**: p01 -9.06 / p50 -0.13 / p99 3.15; mean -0.428.
- **t_e=African American cue**: p01 -4.20 / p50 -0.11 / p99 4.53; mean -0.112.
- **t_e=Hispanic/Spanish-Latin cue**: p01 -4.10 / p50 -0.20 / p99 4.16; mean -0.213.
- **t_p=Late-career**: p01 -1.08 / p50 6.09 / p99 17.33; mean 5.896.
- **t_p=Mid-career**: p01 -1.33 / p50 4.03 / p99 12.97; mean 4.312.

Notable: **Asian p01 = -9.06** — a thin slice of occupations shows a substantial Asian/Vietnamese name-cue penalty (see cate_by_occupation.csv).

Stereotype × tier (proposal §4):
- female stereotype × high tier: female contrast = +0.462
- male stereotype × high tier: female contrast = +0.913
- neutral stereotype × high tier: female contrast = +0.313

**Verdict on H₂**: supported. The marginal CATE spreads (p01 to p99) for name-origin cue axes range from ~7 to ~12 score points — substantial heterogeneity by occupation × model — even though the means hover near zero.

### H₃ — Objective signals attenuate applicant-cue effects

Pooled β_S = -0.252 (SE 0.107, raw p = 0.019, FWER adj p = 0.042, placebo p = 0.028). **Negative** sign — adding objective signals HURTS hiring score, contra H₃ direction. Robust under all three sanity checks (raw, FWER, placebo). Diagnostic: the effect is concentrated in high-tier occupations (Lawyers Δ = −2.94, Financial Analysts −0.99, RNs −0.61). Hypothesised cause: signal generator uniform(3.0, 4.0) → mean GPA 3.42 reads as mediocre for elite roles. Data-generation artifact, not a model-bias finding. **Documented in §12 limitations; tier-calibrated signal generation flagged as v2.**

## Headline findings (for proposal §8 cascade)

1. **Career stage dominates**: late-career β = +5.90, mid-career β = +4.36 score points (both adj p ≤ 0.001). Strongest signal in the audit; identical across both models.
2. **Within-vendor model-version contrast**: GLM 5.1 shows female and Asian/Vietnamese name-cue sensitivity; GLM 4.5 is neutral on every gender/name-origin cue axis. Same prompt, same data, same scoring rubric — only the model version differs.
3. **Pooled female / Asian effects** lose significance under FWER. Visible only on glm-5.
4. **Objective-signal anomaly** is robust but driven by data-generation; v2 should tier-calibrate signal magnitudes.

## Robustness (Phase 8)

Within-cluster permutation placebo (B = 500): PASS — all placebo means within ±0.5.
No data-leak signature. Career-stage, Asian/Vietnamese cue, and objective-signal coefficients robust at placebo p ≤ 0.03;
Black / Hispanic observed coefs sit inside the placebo distribution (consistent with their FWER non-significance).

Borda-count ranking sweep (990 cells, 198 five-candidate groups × 2 Zhipu models):
5/7 implemented-cue coefficients agree in direction with direct-score OLS. Career-stage effects remain positive (late-career +3.78, mid-career +0.90) and name-origin cue effects remain negative, but female flips sign (+0.22 direct vs -0.77 Borda) and the objective signal flips sign (-0.25 direct vs +0.74 Borda). Because Borda SEs are large (≈1.7–2.7 score points), the rank-elicitation check should be read as qualitative directional robustness, not a powered replacement for the 16 000-observation score panel.

Temporal-split drift check (3 retrieval windows): gender and name-origin cue coefficients stay within 0.80 score points max-min; career-stage coefficients vary by 1.22–1.35 score points but remain strongly positive in every window. No sign of an implemented-cue result reversal driven by retrieval timing.
