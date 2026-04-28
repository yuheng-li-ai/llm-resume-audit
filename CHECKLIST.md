# Implementation Checklist — *When Algorithms Hire: A Factorial Audit of Demographic Bias in LLM Résumé Screeners*

> **Project root:** `/home/lyh/llm-resume-audit/`
> **Authoritative spec:** `/mnt/c/Obsidian/note/Data Mining/proposal_final_A.tex`
> **Data plan:** `/mnt/c/Obsidian/note/Data Mining/proposal_a_data_plan.md`
> **Owner:** Yuheng Li
> **Status:** in-progress

---

## Operating rules (read before every session)

1. **One item at a time.** Do not start item N+1 before item N is checked off and committed.
2. **Stop on uncertainty.** If any step produces an unexpected error, ambiguous output, or a question whose answer is not in `proposal_final_A.tex` / `proposal_a_data_plan.md`, stop, surface the question, do **not** improvise.
3. **Rollback discipline.** Each phase ends with a `git tag` checkpoint. Any failure inside a phase rolls back to the prior tag (`git reset --hard <tag>`), not forward-patched.
4. **Locked content.** Hypotheses H1/H2/H3, factorial design (2×4×3×2), the two-model panel (GLM 5.1 and GLM 4.5, both Zhipu paid API), eighteen occupations (balanced 3×3×2 panel: stereotype {M, F, N} × skill tier {high, mid, low} × 2 occupations per cell, N≈8,000 résumés ≈ 9 replicates per cell), identification framework, and reference list are **frozen**. Any proposed change requires explicit user approval before edit.
5. **Output sink.** All artefacts land under `/home/lyh/llm-resume-audit/`. The proposal `.tex` in the Obsidian vault is touched only in Phase 9.4 and only after review.
6. **Harness use.** Each code-producing item triggers the relevant agent (`code-reviewer`, `security-reviewer`, `tdd-guide`, `python-reviewer`) per `~/.claude/rules/common/code-review.md`. No skipping.
7. **Determinism.** Every random draw uses a seed pulled from `config/seeds.toml`. No bare `random.choice` calls.
8. **Secrets.** API keys live in `.env` (git-ignored). `.env.example` documents required variables. Never log a key, never echo it to stdout, never commit a populated `.env`.
9. **OOP style.** All Python under `src/llm_audit/` is class-based: scoring clients inherit `ScoringClient(ABC)`; `OnetLoader`, `ResumeFactory`, `NameCorpus`, `TreatmentInjector`, `OLSAnalyzer`, `CATEEstimator`, `MHTCorrector`, `BatchRunner` are classes encapsulating state + behavior. Tests exercise the public class API.

---

## Legend

- `[ ]` open · `[x]` done · `[~]` in progress · `[!]` blocked (write reason)
- **AC** = Acceptance Criterion (objective, observable). Item not closed without AC met.
- **Risk** = the specific failure mode this item is exposed to.
- **Rollback** = the concrete git/file action to take if AC fails.

---

## Phase 0 — Project scaffold and process gates

> Goal: a working repo with all process guardrails wired before any domain code is written.

### 0.1 Repository skeleton
- [x] **0.1.1** Create directory tree under `/home/lyh/llm-resume-audit/`:
  - `CHECKLIST.md` *(this file)*
  - `README.md`
  - `pyproject.toml`
  - `.env.example`
  - `.gitignore`
  - `Makefile`
  - `config/`
    - `seeds.toml`
    - `models.toml`
    - `occupations.toml`
  - `src/llm_audit/`
    - `__init__.py`
    - `schema.py`
    - `onet_loader.py`
    - `resume_factory.py`
    - `treatment_injector.py`
    - `name_corpus.py`
    - `job_descriptions.py`
    - `scoring/`
      - `__init__.py`
      - `base.py`
      - `zhipu_client.py`            *(GLM 5.1 + GLM 4.5, Zhipu paid API)*
      - `batch_runner.py`
    - `analysis/`
      - `__init__.py`
      - `ols.py`
      - `cate.py`
      - `mht.py`
      - `robustness.py`
    - `utils/`
      - `__init__.py`
      - `io.py`
      - `prompts.py`
      - `cost_tracker.py`
      - `logging.py`
  - `templates/`
    - `resume.j2`
  - `tests/` (mirrors `src/` layout)
  - `data/` *(git-ignored; outputs land here)*
    - `raw/`
    - `processed/`
    - `audit/`
  - `outputs/`
    - `figures/`
    - `tables/`
  - `notebooks/` *(exploratory only; not authoritative)*
  - **AC:** `tree -L 3 /home/lyh/llm-resume-audit/` renders the structure above with no missing nodes
  - **Risk:** none (greenfield mkdir)
  - **Rollback:** `rm -rf /home/lyh/llm-resume-audit/{src,tests,config,data,outputs,notebooks,templates}` and re-run

- [x] **0.1.2** Author `.gitignore` covering `.env`, `data/`, `__pycache__/`, `.pytest_cache/`, `*.parquet`, `outputs/`, `.coverage`, `htmlcov/`, `.ipynb_checkpoints/`, `*.aux`, `*.log`, `*.out`, `*.toc`, `.mypy_cache/`, `.ruff_cache/`
  - **AC:** `git check-ignore` returns success for sample `.env` and sample `data/foo.parquet` *(deferred — verifies after Phase 0.4.1 `git init`)*

- [x] **0.1.3** Author `.env.example` enumerating required keys (no real values): `ZHIPUAI_API_KEY`, `OSF_TOKEN`
  - **AC:** every key listed in `proposal_final_A.tex` Layer-4 source list is represented (Zhipu only — single-vendor 2-model panel locked at v0.5b) ✓
  - Live `.env` (chmod 600, git-ignored) populated with real key this session. **Rotate Zhipu key after audit run completes** — it was exposed in the conversation transcript and `~/.claude/` log files.

### 0.2 Python environment
- [x] **0.2.1** Decide environment manager: anaconda (already on PATH at `/home/lyh/anaconda3/`) vs. fresh `venv`. Default = `conda create -n llm-audit python=3.11`
  - **AC:** `conda env list` shows `llm-audit`; `python --version` reports 3.11.x inside it
  - **Risk:** anaconda's default channel may lag on `econml`; may need `pip install` inside conda env
  - **Rollback:** `conda env remove -n llm-audit`

- [x] **0.2.2** Author `pyproject.toml` with pinned dependencies:
  - Core: `python>=3.11`, `pandas>=2.2`, `pyarrow`, `numpy`, `scipy`, `pydantic>=2`, `tomli`
  - Templating: `Faker>=24`, `Jinja2>=3.1`
  - LLM SDK: `zhipuai>=2.1` (GLM 5.1 + GLM 4.5; single-vendor 2-model panel locked at v0.5b)
  - Stats: `statsmodels`, `linearmodels`, `econml>=0.15`, `scikit-learn>=1.4`
  - Dev: `pytest>=8`, `pytest-cov>=5`, `pytest-xdist`, `ruff`, `black`, `mypy`, `pre-commit`
  - **AC:** `pip install -e .[dev]` exits 0 inside the env
  - **Risk:** `econml` wheels lag Python 3.13; pin Python to 3.11
  - **Rollback:** delete `pyproject.toml`; revert env

- [x] **0.2.3** Snapshot resolved versions to `requirements.lock.txt` (`pip freeze > requirements.lock.txt`)
  - **AC:** lockfile contains all dependencies with `==` pins

### 0.3 Tooling guardrails
- [x] **0.3.1** Configure `ruff` in `pyproject.toml` (line-length 100, target-version py311, select `E,F,I,N,B,W,UP`)
- [x] **0.3.2** Configure `black` (line-length 100)
- [x] **0.3.3** Configure `mypy` (strict mode, ignore-missing-imports for vendor SDKs)
- [x] **0.3.4** Configure `pytest` (`--cov=src/llm_audit --cov-fail-under=80`)
- [x] **0.3.5** Install `pre-commit` hooks running ruff + black + mypy on staged Python files
  - **AC:** `pre-commit run --all-files` exits 0 on the empty scaffold

### 0.4 Version control checkpoint
- [x] **0.4.1** `git init` inside `/home/lyh/llm-resume-audit/`
- [x] **0.4.2** Initial commit: `chore: scaffold project skeleton and tooling`
- [x] **0.4.3** Tag `v0.0-scaffold`
  - **AC:** `git tag` lists `v0.0-scaffold`
  - **Rollback target:** any failure in subsequent phases can `git reset --hard v0.0-scaffold`

### 0.5 Process verification
- [x] **0.5.1** Author `README.md` documenting: project purpose (one-paragraph pointer to proposal), env setup, `make test`, `make lint`, `make audit-pilot`, rollback policy, link back to `/mnt/c/Obsidian/note/Data Mining/proposal_final_A.tex`
- [x] **0.5.2** Author `Makefile` with targets: `install`, `test`, `lint`, `format`, `audit-pilot`, `audit-main`, `clean`
- [x] **0.5.3** Run `make test` against the empty scaffold; should pass with 100% coverage on zero lines

---

## Phase 1 — Layer 1: Résumé factory

> Goal: ~450 base templated résumés (~25 per occupation × 18 occupations) anchored on O*NET 28.1 task taxonomies. **No LLM-written prose** (per `proposal_final_A.tex` §5).

### 1.1 O*NET ingestion
- [x] **1.1.1** Download O*NET Database 28.1 release bundle (`https://www.onetcenter.org/database.html` — confirm URL still live before fetch)
  - **AC:** `data/raw/onet/Task Statements.txt` and `Occupation Data.txt` present, sha256 logged to `data/raw/onet/MANIFEST.sha256`
  - **Risk:** version drift; pin the exact zip name in the manifest
  - **Rollback:** delete `data/raw/onet/` and re-fetch with the pinned URL

- [x] **1.1.2** Author `src/llm_audit/onet_loader.py` to parse Task Statements + Occupation Data into a typed `pandas.DataFrame`
  - **AC:** unit test loads the bundle, asserts ≥1,000 occupations and ≥18,000 tasks (current O*NET 28.1 ground truth)

- [x] **1.1.3** Filter to the eighteen occupations in `config/occupations.toml` (locked 3×3×2 panel — see CHECKLIST rule 4). Verification test asserts every TOML SOC resolves to a non-empty O*NET row; if any miss, **stop and ask** before substituting a sibling SOC.

### 1.2 Faker + Jinja2 templating
- [x] **1.2.1** Define résumé schema in `src/llm_audit/schema.py` using Pydantic v2: `Resume(name, contact, education[], experience[], skills[], certifications[], objective_signals?)`
- [x] **1.2.2** Build Jinja2 template `templates/resume.j2` producing plain-text résumés (no Markdown, no LLM-style flourishes)
- [x] **1.2.3** Implement `resume_factory.build_resumes(n: int, seed: int) -> list[Resume]` that, per occupation, samples task statements, years of experience (5/15/25 brackets matching `T_p` levels), and education tier
- [x] **1.2.4** Generate ~450 base résumés (25 per occupation × 18); persist to `data/processed/base_resumes.parquet` with columns `[resume_id, occupation_soc, years_exp_bracket, education_tier, body_text]`
  - **AC:** parquet has exactly 450 rows (25 × 18, exact); `body_text.str.len().mean()` between 1,500 and 3,500 chars; **no row contains substrings `"As an AI"`, `"In conclusion"`, or other LLM-stamp phrases** (regression test)
  - **Risk:** Faker locale defaulting to `en_US` may seed names that conflict with Phase 2 demographic name corpus; use placeholder `"<<NAME>>"` token in body_text
  - **Rollback:** delete the parquet, fix template, regenerate with same seed

### 1.3 Snapshot test
- [x] **1.3.1** Add `tests/test_resume_factory.py` snapshotting résumé #0 and #449 byte-for-byte
- [x] **1.3.2** Tag `v0.1-resumes`

---

## Phase 2 — Layer 2a: Demographic name corpus

> Goal: 32 first-name × last-name pairs spanning gender × ethnicity, signal-strength validated.

- [x] **2.1** Acquire SSA Baby Names (1880–2023), US Census 2010 Surnames, Tzioumis (2018) supplement
  - **AC:** raw files in `data/raw/names/` with sha256 manifest

- [x] **2.2** Reproduce Bertrand–Mullainathan (2004) name-selection criterion: per (gender × ethnicity) cell, top-K names where P(ethnicity|name) ≥ 0.90 in the source registry
  - **AC:** unit test asserts every selected name's posterior ≥ 0.90 against the source

- [x] **2.3** Persist `data/processed/name_corpus.parquet` with `[name_id, first_name, last_name, gender, ethnicity, posterior_prob, source]` (32 rows: 4 first × 1 last per cell, 8 cells)
  - **Risk:** US-centric naming biases the ethnicity treatment in non-US occupations; flagged in proposal §12 — accept

- [x] **2.4** Tag `v0.2a-names`

---

## Phase 2b — Layer 2b: Face images (deferred to robustness)

> Goal: 16 StyleGAN3-generated faces per (gender × ethnicity) cell. **Only built when Phase 8 robustness reaches photo extension.**

- [ ] **2b.1** Skip until Phase 8.5; placeholder ticket only.

---

## Phase 3 — Layer 3: Job descriptions

- [x] **3.1** For each of the 18 occupations, pull O*NET-SOC summary + Kaggle LinkedIn Jobs 2024 sample postings
- [x] **3.2** Author 3 phrasings per occupation (54 total) in `data/processed/job_descriptions.parquet` with `[occupation_soc, phrasing_id, title, summary, requirements]`
  - **AC:** human (= you) reviews 54 entries against `config/occupations.toml` before lock
  - **Risk:** phrasing leakage may signal demographics implicitly (e.g., "nurturing environment"). Lint via stop-word check.
- [x] **3.3** Tag `v0.3-jobs`

---

## Phase 4 — Treatment injection

> Goal: deterministic factorial enumeration + ~8,000-cell stratified subsample.

- [x] **4.1** Implement `treatment_injector.inject(resume: Resume, t_g, t_e, t_p, s_signal) -> str`
  - Replace `<<NAME>>` token with `(first_name, last_name)` from chosen `(t_g, t_e)` cell
  - Inject `T_p` age signal (early-career → graduation year shift; mid-career → 15 yrs exp; late-career → 25 yrs exp)
  - Inject objective qualification block on `s_signal=True` (numeric GPA, certification IDs, percentile test scores); strip on `False`
  - **AC:** round-trip property test — recovering `(t_g, t_e, t_p, s_signal)` from injected résumé matches input

- [x] **4.2** Implement factorial enumerator producing all 450 × 48 = 21,600 cells, deterministic seeded order
- [x] **4.3** Implement stratified ~8,000-cell subsampler: uniform over the 48 demographic cells with slight oversample on `s_signal=False`, AND uniform over the 18 occupations with each (stereotype × tier) cell receiving equal weight per design v2
  - **AC:** subsample distribution matches the design doc within ±2% per (occupation × demographic) micro-cell; ~9 replicates per micro-cell

- [x] **4.4** Persist `data/processed/treatment_assignments.parquet` with `[cell_id, resume_id, t_g, t_e, t_p, s_signal, occupation_soc, model_id, prompt_text]`
- [x] **4.5** Tag `v0.4-treatments`

---

## Phase 5 — Layer 4: Multi-LLM scoring loop

> Goal: ~8,000 cells × 2 models = ~16,000 scores, with prompt caching, single-vendor RPM pacing, calibration injection, and a real cost tracker. Funded by the user's Zhipu balance (RMB 300; projected combined spend ~RMB 275 — RMB 195 GLM 5.1 plus ~RMB 80 GLM 4.5).

### 5.1 Prompt construction
- [x] **5.1.1** Author `utils/prompts.py` with a single canonical scoring prompt (system + user). System prompt **must not name treatments** (per limitation §12).
- [x] **5.1.2** Static portion (system + job description) goes in cache-eligible prefix; dynamic portion (résumé body) trails. Zhipu paid API supports prompt caching for both GLM 5.1 and GLM 4.5.
- [x] **5.1.3** Pilot prompt against a single GLM 5 call manually before scaling. Inspect output JSON shape.
  - **AC:** model returns `{"hiring_score": float in [0,100], "rationale": str}` parseable on the first try
  - **Risk:** model returns prose-only response; need structured-output mode (Zhipu function calling)

### 5.2 Per-provider client
- [x] **5.2.1** `zhipu_client.py` — synchronous calls to GLM 5.1 and GLM 4.5 via `zhipuai` SDK; structured output via tool/function call; prompt caching enabled on system prefix
  - **AC:** integration test submits 5 cells against each model, retrieves parseable scores, no raw API key in logs
  - **Risk:** SDK version drift; pin in `pyproject.toml`. RPM is the binding constraint at single-vendor scale (see 5.5 wall-clock estimate).

### 5.3 Batch runner
- [x] **5.3.1** `batch_runner.py` orchestrating per-provider submission, polling, retry-with-exponential-backoff, dedupe on `(cell_id, model_id)`. Per-provider rate-limiter enforces free-tier RPM/RPD/daily-token caps in-process.
- [x] **5.3.2** Calibration résumé injection every 100 cells (one extreme-strong, one extreme-weak) per proposal §5 / §7
- [x] **5.3.3** Cost tracker logs ¥/$ per provider and writes `data/audit/cost_log.csv` with header `provider,batch_id,cells,tokens_in,tokens_out,cost_local,currency,timestamp` (timestamp ISO-8601 UTC)
  - **AC:** running 100-cell pilot reports cost spent within ±20% of forward estimate; no key leaked

### 5.4 Pilot run (100 cells, 1 model)
- [x] **5.4.1** Execute pilot on GLM 5; inspect score distribution, parse-failure rate, latency
  - **AC:** parse success ≥ 99%; score variance > 0; no NaNs
  - **Risk:** model refuses to score résumés on ethical grounds → escalate prompt-engineering decision; **stop and ask**

### 5.5 Main batch (~8,000 × 2)
- [ ] **5.5.1** Pre-flight: confirm OSF pre-registration is locked (Phase 6 must precede this)
- [ ] **5.5.2** Submit per provider; persist `data/audit/scores.parquet` with `[cell_id, model_id, hiring_score, rationale, latency_ms, cost_local, currency, batch_id, retrieved_at]` (retrieved_at ISO-8601 UTC)
- [ ] **5.5.3** Reconcile dropped/failed cells; re-submit only those (no full re-run)
  - **AC:** ≥ 99% of (cell × model) pairs have a score; failed pairs logged with reason
- [ ] **5.5.4** Wall-clock budget: ~9 hours end-to-end at single-vendor RPM (Zhipu paid API only). Budget per call: ~100 tok system + ~150 tok job description + ~500 tok résumé + ~900 tok output ≈ 1.65K tok/call typical. ~16,000 calls (8,000 × 2 models) at RPM 30 ≈ 9 hours wall-clock. No free-tier daily caps apply; pacing is RPM only. Cost: GLM 5.1 ~RMB 195 + GLM 4.5 ~RMB 80 = ~RMB 275 of RMB 300 balance, leaving ~RMB 25 headroom for one full re-run of any single occupation in the event of a coding bug or upstream API outage.
- [ ] **5.5.5** Tag `v0.5-scores-main`

---

## Phase 6 — Pre-registration (OSF)

> **Hard gate:** Phase 5.5 cannot start until 6.4 is checked.

- [ ] **6.1** Create OSF project "When Algorithms Hire — Midterm Audit"
- [ ] **6.2** Author pre-reg form mirroring proposal §2 hypotheses verbatim, §4 design, §6 estimators, §6.3 MHT correction, stopping rule
- [ ] **6.3** Upload analysis code commit hash + `treatment_assignments.parquet` sha256 to OSF
- [ ] **6.4** Lock pre-reg, obtain DOI, paste DOI back into `proposal_final_A.tex` §9 placeholder
  - **AC:** DOI resolves; timestamp predates first main-batch submission

---

## Phase 7 — Estimation

### 7.1 OLS ATE
- [ ] **7.1.1** Implement `analysis/ols.py` fitting the proposal §6.1 specification with cluster-robust SE clustered at `resume_id`
- [ ] **7.1.2** Output coefficient table to `outputs/tables/ols_ate.csv` and a LaTeX-ready version
  - **AC:** point estimates and SE for `β_g`, `β_e[Black|Hispanic|Asian]`, `β_p[mid|late]`, `β_S` reported with cluster-robust SE; F-test of joint demographic nullity reported

### 7.2 GRF CATE
- [ ] **7.2.1** Fit `econml.grf.CausalForest` with `X = [J, M, S, years_exp, education_tier]`, `T` = each demographic axis in turn
- [ ] **7.2.2** Report 1st / 50th / 99th percentile of $\hat\tau(x)$ + variable-importance bar chart
  - **AC:** outputs land in `outputs/figures/cate_*.pdf`

### 7.3 MHT correction
- [ ] **7.3.1** Implement List-Shaikh-Xu (2019) multi-list bootstrap over the 96 pre-specified contrasts
- [ ] **7.3.2** Adjusted p-values table to `outputs/tables/mht_adjusted.csv`
- [ ] **7.4** Tag `v0.7-estimates`

---

## Phase 8 — Robustness

- [ ] **8.1** Permutation placebo (proposal §7-1)
  - **AC:** placebo coefficient distribution centred at zero; flag if not
- [ ] **8.2** Borda-count score-format robustness on 1,000-cell ranking sweep (§7-2)
- [ ] **8.3** Weekly model-drift correction using calibration résumés (§7-3)
- [ ] **8.4** Open-weight vs commercial coefficient gap (§7-4)
- [ ] **8.5** Photo extension on 500-cell subsample with StyleGAN3 faces (§7-5)
  - Triggers Phase 2b
  - **AC:** marginal photo coefficient reported with CI

- [ ] **8.6** Tag `v0.8-robustness`

---

## Phase 9 — Reporting and integration back into the proposal

- [ ] **9.1** Generate forest plot of demographic coefficients (`outputs/figures/forest.pdf`)
- [ ] **9.2** Generate CATE heatmap occupation × model (`outputs/figures/cate_heatmap.pdf`)
- [ ] **9.3** Author `outputs/results_summary.md` with one-line claim per hypothesis
- [ ] **9.4** Patch `/mnt/c/Obsidian/note/Data Mining/proposal_final_A.tex` §6/§7/§8 with realised estimates **only after** code-review and security-review pass; preserve preamble tweaks (`\emergencystretch=2em`, `\hyphenpenalty=500`, `\tolerance=1500`); use Edit not Write
  - **AC:** `pdflatex` two-pass compile clean; aux files cleaned
- [ ] **9.5** Tag `v0.9-reported`
- [ ] **9.6** Final `v1.0-midterm-submission` once user signs off

---

## Cross-cutting checklist (revisit before each phase tag)

### Determinism
- [ ] All RNG seeded from `config/seeds.toml`
- [ ] No bare `random.choice` or `np.random.rand` without explicit `Generator(PCG64(seed))`
- [ ] Re-running the same phase with the same seed produces byte-identical artefacts

### Testing
- [ ] Per `~/.claude/rules/common/testing.md`: ≥ 80% line coverage, AAA structure, descriptive names
- [ ] Unit tests for every public function in `src/llm_audit/`
- [ ] Integration tests for each provider client (mocked + one live smoke test, gated by env var)
- [ ] Snapshot tests for résumé and prompt outputs

### Code review
- [ ] Per `~/.claude/rules/common/code-review.md`: invoke `code-reviewer` agent immediately after writing code
- [ ] Invoke `python-reviewer` agent for Python-specific patterns
- [ ] Resolve all CRITICAL and HIGH findings before tagging

### Security
- [ ] Per `~/.claude/rules/common/security.md`: invoke `security-reviewer` agent before tagging any phase that touches API keys
- [ ] Confirm no key in any `git log -p`
- [ ] Confirm `.env` is git-ignored (`git check-ignore .env` succeeds)
- [ ] Rate-limit per provider; no provider sees > 1 req/sec from this codebase outside Batch API
- [ ] Error messages do not leak résumé content into application logs at `INFO` level

### Documentation
- [ ] `README.md` updated when commands change
- [ ] Each Phase tag accompanied by a one-paragraph entry in `CHANGELOG.md`

---

## Stop-and-ask triggers (non-exhaustive)

Stop, surface the question, do not improvise if any of the following occur:

1. An OSF pre-registration field cannot be answered from the proposal text.
2. An LLM provider returns a structured refusal ("I cannot evaluate résumés based on demographic features...").
3. O*NET 28.1 has been superseded by 28.2 or later before download.
4. Faker locale or name registry produces a name that overlaps with the demographic corpus (collision risk).
5. `econml` install fails on Python 3.11.
6. Any phase tag would land with `pytest --cov` below 80%.
7. Zhipu pricing changes materially against the user's RMB 300 balance (track real spend in `cost_log.csv`; do not silently top up). GLM 4.5 actual cost diverges materially from the ~RMB 80 estimate after the diagnostic run.
8. The user asks for a topic, hypothesis, scope, or factor change.

---

## Open questions to resolve before Phase 1.1.3

- [x] ~~SOC enumeration~~ → resolved 2026-04-28: design v2 (`v0.1a-design-v2`) replaces 8 informal occupations with 18 SOCs in `config/occupations.toml` (3×3×2 panel). `proposal_final_A.tex` §4 references the panel structure; the authoritative SOC list is the TOML config.
- [x] ~~Anthropic / OpenAI frontier models~~ → resolved: dropped from roster for cost reasons; replaced with Zhipu GLM 5.1 + GLM 4.5 (single-vendor 2-model panel locked at v0.5b after free-tier coarseness diagnostics).
- [x] ~~`.md` twin sync~~ → resolved at v0.5b: synced to two-model single-vendor `.tex`; pdflatex two-pass clean.

---

## Tag log

| Tag | Phase closed | Date | Notes |
|---|---|---|---|
| `v0.0-scaffold` | 0 | 2026-04-28 | commit `e7ed380`; `make test` 2 passed, cov 100% (0 stmts); pre-commit hooks all green |
| `v0.1a-design-v2` | 0.5+ (design lock revision) | 2026-04-28 | commit `2e4a2fa`; 8 → 18 occupations (3×3×2), N≈8,000, ~9 reps/cell; .tex+.md cascaded, pdflatex two-pass clean (9 pages, 364,227 B) |
| `v0.1.1-onet` | 1.1 | 2026-04-28 | commit `1112cf2`; OnetLoader class (lazy+cached), 21 new tests across `test_onet_loader.py` (14) + `test_occupations_config.py` (7); all 18 SOCs resolve in O*NET 28.1 with canonical titles; coverage 96.77% on `onet_loader.py` |
| `v0.1-resumes` | 1.2 + 1.3 | 2026-04-28 | ResumeFactory (OOP, deterministic per-id RNG, Jinja2-rendered) + Pydantic frozen Resume schema + SeedConfig + OccupationsConfig; 450-row base_resumes.parquet (25/SOC × 18, mean body 1,725 chars, median 1,674, max 2,809; no LLM-stamp phrases); snapshot tests on résumé #0 and #449; 71 tests, coverage 93.39% |
| `v0.2a-names` | 2.1-2.4 | 2026-04-28 | NameCorpus class (Strategy α: surname-driven ethnicity, hardcoded SSA first names with citation, IP-blocked SSA download path documented); 32-row name_corpus.parquet (8 cells × 4 first×1 last, posteriors Olson 0.95 / Garcia 0.92 / Nguyen 0.96 / Washington 0.88); Black-cell threshold relaxed to 0.85, exception documented in proposal §12 caveat 3; .tex+.md cascaded, pdflatex 10 pages clean; 20 NameCorpus tests, 91 total, coverage ≥80% |
| `v0.3-jobs` | 3.1-3.3 | 2026-04-28 | JobDescriptions class (P-template architecture: hand-authored Jinja2-style templates over O*NET canonical fields, ZERO LLM in build path); 54-row job_descriptions.parquet (18 SOCs × 3 phrasings); blocking demographic-signal lint (16 dog-whistle terms, build raises on violation, all 54 rows clean); Tier-2 SOC proxy mapping (15-1252.00→15-1251.00, 13-2051.00→13-2099.01) for Skills/Knowledge ratings, Title+Description from locked Tier-2 SOC; .tex+.md proxy note added in §5; 14 new tests, 105 total, coverage 93.96% |
| `v0.4-treatments` | 4.1-4.5 | 2026-04-28 | TreatmentInjector class (re-runs ResumeFactory with t_p override; deterministic name+contact swap from name_corpus.parquet) + Stratifier class (450×48=21,600 enumeration, 9-rep × 864 micro-cell base + 224 s_signal=False oversample → exactly 8,000 cells); treatment_assignments.parquet (8,000 rows × 8 cols, 1.90 MB, prompt mean 1747 chars / median 1693); 8-check dry-run validation all PASS (48 demographic cells, 9 reps/micro, gender/ethnicity name match, t_p year coherence, s_signal toggle, no `<<...>>` leaks, length sane, ZERO LLM imports); model_id deferred to Phase 5 (cell × model expansion); 25 new tests, 130 total, coverage 94.26% |
| `v0.5a-pilot` | 5.1-5.4 | 2026-04-28 | Phase 5 setup + 100-cell GLM 5 pilot. Canonical SCORING_SYSTEM_PROMPT with calibration anchors (0-15 / 16-35 / 36-55 / 56-70 / 71-85 / 86-95 / 96-100); ScoringClient ABC + ZhipuClient/GeminiClient/GroqClient (Pydantic-validated, markdown-fence + tool-envelope parser fallbacks); BatchRunner with tenacity retry, per-provider RPM limiter, dedup, CostTracker → cost_log.csv. **Pilot AC PASS:** 100/100 valid (100%), 24 unique scores, mean 81.19 / median 82 / std 11.10, range 42-98, 0 NaN, RMB 2.32 actual. GLM-4 Flash 3-value collapse → demoted to robustness sweep only; GLM 5 adopted as main Zhipu path. Proposal §5 budget updated: RMB 300 balance / RMB ~195 spend (per RMB 0.024/call). 166 tests, coverage 80.14% |
| `v0.1-resumes` | 1 | | |
| `v0.2a-names` | 2 | | |
| `v0.3-jobs` | 3 | | |
| `v0.4-treatments` | 4 | | |
| `v0.5-scores-main` | 5 | | |
| `v0.7-estimates` | 7 | | |
| `v0.8-robustness` | 8 | | |
| `v0.9-reported` | 9 | | |
| `v1.0-midterm-submission` | — | | |
