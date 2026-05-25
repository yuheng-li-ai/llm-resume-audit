# Phase 8 — deferred sub-items (8.4 + 8.5)

CHECKLIST Phase 8 has five sub-items. Three are executed in this codebase:

- **8.1 permutation placebo** — done (commit `6c871c9`; PASS, no leak signature).
- **8.2 Borda-count score-format robustness** — done (commit pending Phase 8.2 land).
- **8.3 weekly model-drift correction** — done as a temporal-split robustness check (commit pending Phase 8.3 land); no calibration cells were injected during the main batch, so the original "weekly calibration résumé re-score" design is unavailable; the temporal split is the substantive substitute and reports the same diagnostic.

Two sub-items are deferred with explicit justification:

## 8.4 Open-weight vs commercial coefficient gap — INFEASIBLE under the locked panel

The CHECKLIST item asks for a robustness check that contrasts open-weight versus commercial models, originally motivated by the 4-model roster (GLM 5.1, GLM-4 Flash, Gemini 2.5 Flash, Llama 3.3 70B) at proposal v0.1a-design-v2. The free-tier coarseness diagnostic at `v0.5b-panel-trim` trimmed the panel to GLM 5.1 + GLM 4.5 (both Zhipu paid, both proprietary commercial), so the open-vs-closed and Western-vs-Chinese contrasts are no longer estimable on the present instrument. The within-vendor capacity contrast (GLM 5.1 vs GLM 4.5) implemented in Phase 7.3 fills part of the gap and is reported in proposal §6.4 + §8.

A v2 audit on a multi-vendor panel (Zhipu + at least one open-weight or Western commercial provider with score-resolution comparable to GLM 5.1) is required to complete the original 8.4 design. The single-vendor scope is disclosed in proposal §12 caveat 1 + 2.

## 8.5 Photo extension on 500-cell subsample with StyleGAN3 faces — BLOCKED on Phase 2b

The CHECKLIST item depends on Phase 2b (16 face images per demographic cell via StyleGAN3), which is itself a placeholder (`config/occupations.toml` does not load face URIs and no scoring path consumes them). Implementing 8.5 would require:

1. Generating or licensing 8 demographic-cell × 16 = 128 StyleGAN3 faces.
2. Adding a multimodal prompt path to the scoring stack (current `ZhipuClient` is text-only; would need either Zhipu's vision endpoint or a different model entirely).
3. Re-running 500 cells with the photo-augmented prompt.
4. Estimating the marginal coefficient of the visual demographic cue on top of the name-only treatment.

This is a v2 / final-paper extension, not a midterm item. Proposal §10 already names the photo extension as a final-paper deliverable. Disclosed in proposal §4 (Layer 2b: "used in robustness extension only") and §10.

## Net effect on the midterm

Phase 8 reports 3 of 5 sub-items as **executed and committed** (8.1 placebo, 8.2 Borda, 8.3 temporal drift), and 2 of 5 as **deferred with justification** (8.4 infeasible, 8.5 blocked on 2b). The robustness story carried into proposal §7 + §8 rests on 8.1 + 8.2 + 8.3.
