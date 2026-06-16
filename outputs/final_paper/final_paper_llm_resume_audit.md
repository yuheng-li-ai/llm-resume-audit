---
title: "When Algorithms Hire"
aliases:
  - Final Paper LLM Resume Audit
tags:
  - final-paper
  - llm-audit
  - causal-inference
  - algorithmic-hiring
---

# When Algorithms Hire: A Factorial Audit of Demographic Bias in Large Language Model Resume Screeners

**Author:** Yuheng Li
**Course:** Social Science and Data Mining
**Date:** May 2026
**Format:** Generic two-column LaTeX paper; this Markdown note is a readable companion to the final paper content.

## Abstract

Large language models (LLMs) are increasingly used as first-pass resume screeners, but their demographic sensitivity remains difficult to measure because real hiring data confound applicant identity with qualifications, occupation, and employer choice. I run a randomized factorial audit in which 8,000 synthetic resume-treatment cells are scored by two Zhipu GLM models, producing 16,000 hiring-score observations across an 18-occupation panel. Applicant gender, ethnicity, age signal, and an objective-qualification block are randomized while base qualifications are held fixed. This design identifies demographic average treatment effects directly through random assignment. The strongest effect is age: late-career applicants score 5.90 points higher and mid-career applicants 4.36 points higher than otherwise identical early-career applicants. Pooled gender and ethnicity effects are small: female names receive a 0.22 point premium and Asian names a 0.37 point penalty, but neither survives family-wise correction in the pooled panel. The frontier model, GLM 5.1, shows female and Asian sensitivity after correction, while GLM 4.5 is neutral on every demographic axis. Conditional average treatment effects reveal substantial heterogeneity, including an Asian penalty below -9 points in the lower tail. A placebo test finds no merge-leak signature, a Borda ranking robustness check agrees in direction for five of seven coefficients, and temporal splits show no demographic reversal over the collection window. The audit therefore finds limited race/gender effects in the pooled GLM panel, strong age sensitivity, and evidence that model capacity can change demographic sensitivity even within a single vendor family.

## 1. Introduction

Employers increasingly use automated systems to filter resumes before any human recruiter reads them. New York City's Local Law 144, the EU AI Act, and guidance from the U.S. Equal Employment Opportunity Commission all treat hiring algorithms as objects of audit because small differences at the screening stage can scale into large differences in access to interviews. LLM-based screeners intensify this concern. They can read unstructured resumes, summarize fit, and produce apparently calibrated scores, but they are trained on human language and may reproduce demographic associations embedded in labor-market text.

This paper asks a narrow causal question: holding qualifications fixed, does the demographic identity signaled by an applicant's name and career history causally change the score assigned by an LLM resume screener? The question is deliberately phrased in potential-outcomes terms. Real hiring data rarely separate identity from experience, education, networks, and employer targeting. A randomized audit can. I construct resumes from common templates and O*NET task statements, assign demographic signals by randomization, and submit the resulting applications to two models under a fixed scoring prompt. The design implements the intervention $do(T=t)$ directly: identity cues are manipulated while the underlying resume is held constant.

The experiment makes three contributions. First, it brings the Bertrand and Mullainathan audit logic into a setting where the decision-maker is an LLM rather than a human employer. Second, it covers a balanced 18-occupation panel rather than a narrow set of stereotypically male or female jobs. Third, it compares GLM 5.1 and GLM 4.5, two models from the same vendor family. That within-vendor contrast does not identify all model-market differences, but it holds the provider ecosystem fixed while changing model capacity.

The main findings are mixed. Traditional race/gender concerns are not strongly supported in the pooled panel after multiple-testing correction, although GLM 5.1 alone shows a significant female premium and Asian penalty. The most practically large signal is age: mid- and late-career applicants are scored substantially above otherwise identical early-career applicants. A third result is diagnostic rather than substantive: adding an objective-qualification block lowers scores slightly, especially in high-tier occupations, because the generated GPA and test signals appear mediocre for elite jobs.

## 2. Research Design and Data

The unit of design is a resume-treatment cell. A base resume $i$ is first generated for one of eighteen occupations. The resume is then assigned four randomized factors: gender signal $T_g$, ethnicity signal $T_e$, age/career-stage signal $T_p$, and an indicator $S$ for whether an explicit objective-qualification block is included. The resulting prompt is scored by model $M \in \{\text{GLM 5.1},\text{GLM 4.5}\}$. The outcome $Y$ is a 0-100 hiring score.

The realised design contains 8,000 treatment cells and 16,000 model-score observations. The occupation panel contains eighteen occupations arranged as a $3 \times 3 \times 2$ grid: gender stereotype (male, female, neutral) by skill tier (high, mid, low) by two occupations per cell. This construction avoids a common weakness of resume audits: all inference coming from a few occupations such as software engineering and nursing.

The design separates three ideas that are often conflated: the demographic cue, the qualification content, and the decision environment. The experiment manipulates the cue, controls qualification content within base-resume families, and stratifies the decision environment by occupation and model.

The resumes are synthetic but structured. Base resumes are generated from O*NET task taxonomies and templated with Faker and Jinja2 rather than produced by an LLM. Names are drawn from a curated gender-by-ethnicity corpus based on Census and first-name sources. Age is signaled through graduation years and experience histories. The objective-signal block includes GPA, certification identifiers, and standardized-test percentile information.

| Component | Levels or source | Realised count |
| --- | --- | ---: |
| Gender signal | male, female | 2 |
| Ethnicity signal | White, Black, Hispanic, Asian | 4 |
| Age signal | early-career, mid-career, late-career | 3 |
| Objective-signal block | absent, present | 2 |
| Occupations | balanced stereotype x tier panel | 18 |
| Models | GLM 5.1 and GLM 4.5 via Zhipu paid API | 2 |
| Treatment cells | stratified sample from full factorial | 8,000 |
| Model-score observations | treatment cells x models | 16,000 |

The main panel uses GLM 5.1 and GLM 4.5 through the Zhipu paid API. The original design considered a wider model roster, but pre-flight diagnostics showed that several free-tier alternatives collapsed the score distribution into only a few unique values under the calibrated prompt. Including such models would have made continuous-score regression and CATE estimation difficult to interpret. Details of this panel-trimming step are reported in Appendix A.

The scoring prompt asks the model to act as a first-pass resume screener and return a hiring score. It does not label the demographic treatments or tell the model that the audit concerns bias.

## 3. Identification and Estimation

Let $Y_i(t)$ denote the score that base resume $i$ would receive if assigned demographic treatment value $t$. The estimand for a binary treatment contrast is:

$$
\tau(t,t') = \mathbb{E}[Y_i(t)-Y_i(t')].
$$

For ethnicity and age, the baseline categories are White and early-career respectively. The experiment observes one realised treatment condition for each cell, but random assignment ensures that observed differences in conditional means identify average potential-outcome differences. The formal proof is given in Appendix C; the core argument is that the random number generator, not resume quality, determines treatment.

This gives strong internal validity for the audit instrument. Treatment assignment is independent of latent resume quality, and every treatment level has positive probability in every design stratum. Consistency is also plausible because each treatment is a concrete string manipulation: a name replacement, a career-stage signal, and a qualification block. The main remaining dependence is repeated use of the same base resumes, which motivates cluster-robust standard errors by base resume.

External validity is weaker. A score is not a callback, synthetic resumes are not real applicants, and two Zhipu models are not the entire market for hiring AI. The audit should therefore be read as a controlled test of demographic sensitivity in a specific class of LLM screeners, not as a direct estimate of labor-market discrimination by employers.

This is the core method tradeoff. Compared with observational hiring data, the synthetic audit has stronger internal validity because the demographic cue is randomized. Compared with human callback audits, it is cheaper and more scalable. The cost is weaker ecological validity.

The primary model is:

$$
Y_{ijm} =
\alpha + \beta_g T^g_i + \boldsymbol{\beta}_e^\top T^e_i
 + \boldsymbol{\beta}_p^\top T^p_i + \beta_S S_i
 + \delta_j + \mu_m + \varepsilon_{ijm}.
$$

Occupation fixed effects and model fixed effects account for design strata and precision. Standard errors are clustered at the base-resume level. To estimate heterogeneity, I fit generalized random forests for each non-baseline contrast and report the 1st, 50th, and 99th percentiles of the estimated CATE distribution. Family-wise error is controlled using a Romano-Wolf/List-Shaikh-Xu style step-down correction over 21 contrasts.

## 4. Results

The pooled gender and ethnicity effects are small in score units. Female names receive a 0.220 point premium relative to male names. Asian names receive a 0.373 point penalty relative to White names. Black and Hispanic coefficients are also negative, at -0.197 and -0.146, but are not statistically distinguishable from zero. The female and Asian coefficients have raw $p$-values near 0.03, but their family-wise adjusted $p$-values are both 0.089.

The age effects are much larger. Late-career applicants score 5.903 points above otherwise identical early-career applicants, and mid-career applicants score 4.364 points higher. Both survive family-wise correction easily. In practical terms, the age signal is the largest demographic dimension in the audit.

The age result is not automatically evidence of unlawful discrimination because experience can be job-relevant. It does show, however, that the model converts career stage into a large score premium even under controlled resume generation.

The objective-signal coefficient reverses the preregistered expectation. The presence of a credential block lowers the score by 0.252 points and survives the family-wise correction at 0.042. This should not be interpreted as a general result that objective information harms applicants. The block's generated GPA has a mean around 3.42, which appears mediocre in high-tier occupations such as lawyers and financial analysts.

| Coefficient | Estimate | SE | raw p | FWER p |
| --- | ---: | ---: | ---: | ---: |
| Female | 0.220 | 0.103 | 0.033 | 0.089 |
| Asian | -0.373 | 0.175 | 0.034 | 0.089 |
| Black | -0.197 | 0.143 | 0.168 | 0.418 |
| Hispanic | -0.146 | 0.138 | 0.289 | 0.532 |
| Late-career | 5.903 | 0.174 | <1e-100 | 0.001 |
| Mid-career | 4.364 | 0.173 | <1e-100 | 0.001 |
| Objective signal present | -0.252 | 0.107 | 0.019 | 0.042 |

The pooled results mask an important model contrast. GLM 5.1 shows significant female and Asian sensitivity after correction: the female contrast has adjusted $p=0.011$, and the Asian contrast has adjusted $p=0.010$. GLM 4.5 is neutral on every demographic axis. Since the prompt, resume pool, outcome scale, and vendor family are held fixed, this is a within-vendor capacity contrast rather than a cross-provider comparison.

This matters for audit policy. Certifying one model version does not guarantee that nearby versions in the same vendor family behave identically.

CATE estimates support the second hypothesis: demographic effects vary by occupation and model. For the female contrast, the CATE distribution runs from -3.30 at the 1st percentile to +4.62 at the 99th percentile. The Asian contrast has a mean of -0.43 but a 1st percentile of -9.06, indicating that a small slice of occupation-model cells concentrates a much larger penalty. Age again dominates: the late-career CATE distribution has a median of 6.09 and a 99th percentile of 17.33.

| Axis | Contrast | n | p01 | p50 | p99 | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Gender | Female | 16,000 | -3.30 | 0.14 | 4.62 | 0.23 |
| Ethnicity | Asian | 7,990 | -9.06 | -0.13 | 3.15 | -0.43 |
| Ethnicity | Black | 7,994 | -4.20 | -0.11 | 4.53 | -0.11 |
| Ethnicity | Hispanic | 8,016 | -4.10 | -0.20 | 4.16 | -0.21 |
| Age | Late-career | 10,656 | -1.08 | 6.09 | 17.33 | 5.90 |
| Age | Mid-career | 10,670 | -1.33 | 4.03 | 12.97 | 4.31 |

## 5. Robustness

The first robustness check is a within-cluster permutation placebo. Treatment labels are shuffled within each base-resume cluster and the OLS model is refit 500 times. The test passes this diagnostic: all placebo means are within $\pm 0.5$ score points. Age, Asian, and objective-signal coefficients fall in the tails of their placebo distributions, while Black and Hispanic do not, matching the main inference.

The second robustness check changes the elicitation format. A subsample is scored with a ranking prompt: the model sees five candidates for the same job and ranks them from best to worst. The ranking is converted to a 0-100 Borda score and the OLS model is refit. The realised Borda sweep contains 990 cells, or 198 five-candidate groups evaluated by two Zhipu models. Five of seven coefficient directions match the direct-score design: age remains positive and all ethnicity coefficients remain negative. Female and objective-signal coefficients flip. Because Borda standard errors are large, around 1.7-2.7 score points, this is evidence of partial directional robustness rather than a powered replication.

The third check addresses temporal drift. The 16,000 scored observations are split into three retrieval windows, and the demographic coefficients are re-estimated in each. Female and ethnicity coefficients remain within 0.81 score points max-min across windows. Age coefficients vary by 1.22-1.35 points but remain strongly positive.

Together, the robustness checks support a cautious conclusion. Age is stable across checks. The Asian penalty is directionally supported but marginal in the pooled panel after correction. The female result is more format-sensitive because it is positive under direct scoring and negative under Borda.

| Check | Main conclusion |
| --- | --- |
| Permutation placebo | All placebo means within +/-0.5; no merge-leak signature. |
| Borda ranking sweep | 990 cells; 5/7 signs match direct-score OLS; female and objective signal flip with large SEs. |
| Temporal drift | No demographic-result reversal; age remains positive in every window. |
| Deferred checks | Open-vs-commercial is infeasible under the locked panel; photo extension requires multimodal path and face assets. |

## 6. Discussion

The audit's strongest substantive finding is not a race or gender coefficient, but age. Late-career and mid-career signals raise scores by several points even after holding the resume-generation process fixed. This may reflect a reasonable preference for experience, but it is still relevant to disparate-impact analysis because age is protected in many employment contexts and because years of experience can become a proxy for career stage.

The second finding is that model capacity may matter. GLM 5.1 shows demographic sensitivity that GLM 4.5 does not. This should be interpreted cautiously. A two-model within-vendor contrast cannot establish a universal law that larger models are more biased. It does show, however, that audits should not treat all versions of a vendor's model family as interchangeable.

Compared with observational hiring data, this design has much stronger internal validity: the demographic cue is randomized, the resume content is held fixed, and every coefficient has a clear intervention interpretation. Compared with human callback audits, it is cheaper, scalable, and can test many occupation-model cells quickly. The cost is ecological validity. LLM scores are not human callbacks, and synthetic resumes may miss features of real applicant pools.

For deployment, the synthetic factorial audit should be one layer in a governance pipeline: first test model sensitivity under controlled interventions, then validate with human recruiters or real platform outcomes, then monitor deployed decisions for distribution shift.

## 7. Limitations and Conclusion

Several limitations bound the claims. First, the model panel is restricted to two Zhipu GLM models. Second, the OSF lock was delayed until after the main batch. The design instrument was fixed before scoring and the analysis protocol follows the written proposal, but the registration timing is not ideal. Third, the outcome is a score, not a callback, interview, or offer. Fourth, the objective-signal block was not calibrated tightly enough to occupation tier; the negative signal coefficient is therefore a design lesson rather than a general finding about credentials.

Within those limits, the study demonstrates that a factorial LLM audit can deliver clean causal evidence about demographic sensitivity. In this GLM panel, pooled race/gender effects are small and mostly do not survive family-wise correction, but GLM 5.1 alone shows female and Asian sensitivity. Age effects are large, robust, and practically meaningful. Heterogeneity is substantial enough that pooled averages understate risks in particular occupation-model cells.

## References

See the LaTeX version for the typeset reference list. Key cited works include Bertrand and Mullainathan (2004), Pearl (2009), Imbens and Rubin (2015), Athey, Tibshirani, and Wager (2019), List, Shaikh, and Xu (2019), Salinas et al. (2023), Veldanda et al. (2024), and Wilson and Caliskan (2024).

## Appendix A: Pre-Experiments and Panel Trimming

The original design considered a broader roster, including free-tier or low-cost alternatives. A pilot on GLM 5.1 produced a usable continuous score distribution: 100 pilot calls returned parseable scores, 24 unique score values, mean 81.19, standard deviation 11.10, and range 42-98. Several free-tier candidates produced much coarser distributions under the same calibrated prompt, collapsing to only three or four unique values in diagnostic runs. Such outputs are usable for a coarse binary screen but not for OLS and CATE estimation on a 0-100 scale.

The pre-experiment clarified the cost-quality tradeoff: a free or cheap model is not a useful audit subject if it compresses scores into too few bins. The final panel prioritizes score granularity over vendor breadth.

## Appendix B: Data Construction Details

The base resume factory generated 450 resumes, 25 per occupation. O*NET task statements supplied occupation-specific content. The body text used a template rather than an LLM writer, reducing the risk that the scoring model would detect text generated by a sibling model. The treatment injector replaced name tokens, shifted education and experience histories to represent career stage, and optionally inserted the objective-signal block.

The name corpus was designed to signal gender and ethnicity through first and last names. For White, Hispanic, and Asian surnames, posterior ethnicity probabilities met the planned threshold of 0.90. The Black surname cell used Washington, with posterior probability 0.88, because stricter Census thresholds tended to select recent African or Caribbean immigrant surnames rather than African American surnames in the Bertrand-Mullainathan tradition.

## Appendix C: Identification Proofs

**Proposition 1.** Let $T_i$ be assigned by a random number generator independent of the full vector of potential outcomes $\{Y_i(t):t \in \mathcal{T}\}$. Then $\{Y_i(t):t \in \mathcal{T}\} \perp T_i$.

**Proof.** By construction, the assignment mechanism is generated outside the resume content and outside the model score. For any treatment value $t$ and any potential-outcome event $A$,

$$
P(T_i=t \mid \{Y_i(s)\}_{s\in\mathcal{T}} \in A)=P(T_i=t).
$$

This equality is the definition of statistical independence between treatment assignment and the potential-outcome vector. Therefore the treatment is ignorable in the sense of Imbens and Rubin (2015).

**Proposition 2.** Under ignorability and consistency, $\mathbb{E}[Y_i\mid T_i=t]-\mathbb{E}[Y_i\mid T_i=t'] = \mathbb{E}[Y_i(t)-Y_i(t')]$.

**Proof.** Consistency gives $Y_i=Y_i(t)$ for units assigned $T_i=t$. Therefore:

$$
\mathbb{E}[Y_i\mid T_i=t]=\mathbb{E}[Y_i(t)\mid T_i=t].
$$

By ignorability, $\mathbb{E}[Y_i(t)\mid T_i=t]=\mathbb{E}[Y_i(t)]$. Applying the same argument to $t'$ and subtracting yields the result.

**Proposition 3.** In a balanced randomized factorial design with mutually orthogonal treatment indicators, the population OLS coefficient on a treatment indicator equals the corresponding marginal ATE, conditional on included fixed effects.

**Proof sketch.** By the Frisch-Waugh-Lovell theorem, the coefficient on a treatment indicator can be obtained by residualizing both the outcome and the treatment on the remaining regressors and regressing the residualized outcome on the residualized treatment. Randomization and balance imply that the residualized treatment remains orthogonal to other treatment indicators and fixed-effect strata. Within each stratum, Proposition 2 identifies the treatment contrast as a difference in mean potential outcomes. The OLS coefficient is a weighted average of these stratum-level contrasts.

Backdoor adjustment is not the main identification argument. In observational hiring data, demographic identity can be connected to scores through education, occupation, experience, and unobserved applicant quality. In this audit, treatment is assigned after base resumes are generated. Latent resume quality can affect the score, but it cannot affect treatment assignment.

## Appendix D: Robustness Detail

The Borda comparison estimates are: female direct-score coefficient +0.220 versus Borda -0.767; Asian -0.373 versus -2.700; Black -0.197 versus -2.680; Hispanic -0.146 versus -4.240; late-career +5.903 versus +3.777; mid-career +4.364 versus +0.905; objective signal -0.252 versus +0.738. Signs match for Asian, Black, Hispanic, late-career, and mid-career.

The temporal drift check splits retrievals into three equal windows. Female coefficients range from -0.056 to +0.464. Asian coefficients range from -0.784 to +0.017. Late-career coefficients are +6.691, +5.338, and +5.704; mid-career coefficients are +5.064, +3.844, and +4.191. The age effect persists in every retrieval window.

## Appendix E: Reproducibility and Provenance

The main input files are `data/processed/treatment_assignments.parquet` and `data/audit/scores.parquet`. The realised score file contains 16,000 observations and no non-null errors. Main numerical claims are drawn from `outputs/tables/ols_ate_demographic.csv`, `mht_adjusted.csv`, `cate_percentiles.csv`, `placebo_summary.csv`, `borda_comparison.csv`, and `drift_temporal_split.csv`. The delayed registration disclosure records SHA-256 hashes for the design and score files and commit hashes for the analysis scripts. The OSF lock occurred after the main batch rather than before it; this is reported as a workflow deviation.
