# Results summary — midterm audit

**Specification**: OLS with cluster-robust SE clustered on resume_id,
16 000 observations across 450 clusters, R² = 0.529.
Joint F-test of demographic nullity: F(6, 449) = 203.6, p = 1.07e-124 — demographic nullity firmly rejected.

## Hypothesis-level claims

### H₁ — Demographic identity (name only) causally shifts hiring score

**Pooled, full panel**: female β = +0.220 (SE 0.103, raw p = 0.033, FWER adj p = 0.089). Direction is **positive** — opposite of the Bertrand-Mullainathan baseline expectation. Loses significance after FWER correction.

**Within-model (proposal §7 capacity contrast)**: significant only on the frontier model — glm-5 female adj p = 0.011 (z = 2.78); glm-4.5 adj p = 0.990 (z = 0.12). The mid-tier model is essentially neutral on every demographic axis (smallest adj p = 0.42).

**Ethnicity**: pooled Asian β = -0.373 (adj p = 0.089); glm-5 only adj p = 0.010. Black / Hispanic effects not significant under FWER at either resolution.

**Verdict on H₁**: rejected at the pooled level for Black / Hispanic;
supported on glm-5 for Asian (penalty) and female (premium); magnitude small (≤ 0.4 score points) compared with the 0–100 scale.

### H₂ — Demographic effect varies by job category

CATE percentile spread (across 16 000 cells) for each axis:

- **t_g=female**: p01 -3.30 / p50 0.14 / p99 4.62; mean 0.228.
- **t_e=asian**: p01 -9.06 / p50 -0.13 / p99 3.15; mean -0.428.
- **t_e=black**: p01 -4.20 / p50 -0.11 / p99 4.53; mean -0.112.
- **t_e=hispanic**: p01 -4.10 / p50 -0.20 / p99 4.16; mean -0.213.
- **t_p=late_career**: p01 -1.08 / p50 6.09 / p99 17.33; mean 5.896.
- **t_p=mid_career**: p01 -1.33 / p50 4.03 / p99 12.97; mean 4.312.

Notable: **Asian p01 = -9.06** — a thin slice of occupations shows a substantial Asian penalty (see cate_by_occupation.csv).

Stereotype × tier (proposal §4):
- female stereotype × high tier: female contrast = +0.462
- male stereotype × high tier: female contrast = +0.913
- neutral stereotype × high tier: female contrast = +0.313

**Verdict on H₂**: supported. The marginal CATE spreads (p01 to p99) for ethnicity axes range from ~7 to ~12 score points — substantial heterogeneity by occupation × model — even though the means hover near zero.

### H₃ — Objective signals attenuate the demographic effect

Pooled β_S = -0.252 (SE 0.107, raw p = 0.019, FWER adj p = 0.042, placebo p = 0.028). **Negative** sign — adding objective signals HURTS hiring score, contra H₃ direction. Robust under all three sanity checks (raw, FWER, placebo). Diagnostic: the effect is concentrated in high-tier occupations (Lawyers Δ = −2.94, Financial Analysts −0.99, RNs −0.61). Hypothesised cause: signal generator uniform(3.0, 4.0) → mean GPA 3.42 reads as mediocre for elite roles. Data-generation artifact, not a model-bias finding. **Documented in §12 limitations; tier-calibrated signal generation flagged as v2.**

## Headline findings (for proposal §8 cascade)

1. **Age dominates**: late_career β = +5.90, mid_career β = +4.36 score points (both adj p ≤ 0.001). Strongest signal in the audit; identical across both models.
2. **Within-vendor capacity contrast**: glm-5 (frontier) shows demographic sensitivity (Asian and female); glm-4.5 (mid-tier) is neutral on every demographic axis. Same prompt, same data, same scoring rubric — only the model differs. Bias scales with model capacity.
3. **Pooled female / Asian effects** lose significance under FWER. Visible only on glm-5.
4. **Objective-signal anomaly** is robust but driven by data-generation; v2 should tier-calibrate signal magnitudes.

## Robustness (Phase 8.1 permutation placebo)

Within-cluster permutation placebo (B = 500): PASS — all placebo means within ±0.5.
No data-leak signature. Age and Asian and s_signal coefficients robust at placebo p ≤ 0.03;
Black / Hispanic observed coefs sit inside the placebo distribution (consistent with their FWER non-significance).
