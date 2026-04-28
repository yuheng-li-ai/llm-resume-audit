# llm-resume-audit

Implementation of *When Algorithms Hire: A Factorial Audit of Demographic Bias in LLM Résumé Screeners* (Proposal A, midterm phase).

A pre-registered randomised factorial experiment in which 8,000 synthetic résumés are scored by two LLMs (GLM 5.1 and GLM 4.5, both Zhipu paid API) across eighteen occupations. Identification follows the potential-outcomes framework with the do-operator implemented directly via random treatment assignment; ATE estimated by OLS with cluster-robust standard errors at the base-résumé level, CATE by generalized random forest (`econml`), with multi-list-bootstrap multiple-hypothesis correction. See the authoritative spec for the full design and identification argument.

- **Authoritative spec:** `/mnt/c/Obsidian/note/Data Mining/proposal_final_A.tex`
- **Build plan:** [`CHECKLIST.md`](./CHECKLIST.md)
- **Owner:** Yuheng Li

## Setup

```bash
# Conda env (Python 3.11)
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n llm-audit python=3.11 -y
conda activate llm-audit

# Install package + dev tools (editable)
pip install -e .[dev]

# Pre-commit hooks
pre-commit install

# Secrets
cp .env.example .env
chmod 600 .env
# then fill in ZHIPUAI_API_KEY, OSF_TOKEN
```

## Provider matrix

| Model | Provider | Tier | Notes |
|---|---|---|---|
| GLM 5.1 | Zhipu | paid (~¥195 of ¥300 balance) | main score; calibration-anchored |
| GLM 4.5 | Zhipu | paid (~¥80 of ¥300 balance, est.) | second score; same prompt + schema |

Wall-clock budget for the main 8,000-cell × 2-model batch is single-vendor (Zhipu paid API only), so the run is paced by RPM rather than daily token caps and completes in roughly 9 hours of wall-clock time.

## Common commands

```bash
make install         # pip install -e .[dev]
make lint            # ruff + black --check + mypy
make format          # ruff --fix + black
make test            # pytest with --cov-fail-under=80
make audit-pilot     # 100-cell GLM 5 pilot (Phase 5.4)
make audit-main      # 8,000 × 2 main batch (Phase 5.5; requires OSF lock)
make clean           # remove caches, build artefacts, htmlcov
```

## Project layout

```
src/llm_audit/
  schema.py                 Pydantic résumé / treatment / score schemas
  onet_loader.py            O*NET 28.1 ingest
  resume_factory.py         Faker + Jinja2 templated résumés (Phase 1)
  name_corpus.py            Demographic name corpus (Phase 2)
  job_descriptions.py       Eighteen-occupation job postings (Phase 3)
  treatment_injector.py     Factorial enumerator + 8,000-cell subsampler (Phase 4)
  scoring/
    base.py                 Provider-agnostic scorer interface
    zhipu_client.py         GLM 5.1 + GLM 4.5 (Zhipu paid API)
    batch_runner.py         Per-provider rate-limited orchestrator
  analysis/
    ols.py                  OLS ATE + cluster-robust SE
    cate.py                 econml Generalized Random Forest
    mht.py                  List–Shaikh–Xu (2019) multi-list bootstrap
    robustness.py           Permutation placebo, Borda, drift, photo
  utils/
    io.py                   Parquet readers/writers
    prompts.py              Canonical scoring prompt
    cost_tracker.py         Per-provider spend log
    logging.py              Structured logger with secret redaction
config/                     seeds.toml, models.toml, occupations.toml
templates/                  resume.j2
tests/                      Mirrors src layout
data/                       Git-ignored. raw/, processed/, audit/
outputs/                    Git-ignored. figures/, tables/
```

## Operating policy

- **One CHECKLIST item at a time.** Tag the project with `git tag v<phase>-<name>` at the close of each phase.
- **Stop on uncertainty.** Any unexpected error or ambiguous output halts work; rollback to the prior tag with `git reset --hard <tag>`.
- **Locked content.** Hypotheses H1/H2/H3, the two-model panel (GLM 5.1 + GLM 4.5), the 2×4×3×2 factorial, and the identification argument are frozen; changes require explicit approval before edit.
- **Determinism.** All RNG draws seeded from `config/seeds.toml`. No bare `random.choice`.
- **Secrets.** Live keys live in `.env` (chmod 600, git-ignored). Rotate after every audit run; assume any key pasted into chat or terminal history is exposed.

## License

MIT. See `pyproject.toml`.
