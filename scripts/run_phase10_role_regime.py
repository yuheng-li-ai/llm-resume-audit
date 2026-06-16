"""Phase 10 local runner -- role-regime OLS and transportability sensitivity.

This script addresses employer heterogeneity as an observable role-regime
proxy: occupation gender stereotype x skill tier. It makes no network calls.

Reads:
    data/audit/scores.parquet
    data/processed/treatment_assignments.parquet
    config/occupations.toml

Writes:
    outputs/tables/role_regime_ols.csv
    outputs/tables/role_regime_heterogeneity.csv
    outputs/tables/role_regime_weights.csv
    outputs/tables/transportability_sensitivity.csv
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

REPO = Path(__file__).resolve().parents[1]

EFFECT_TERMS = {
    'C(t_g, Treatment(reference="male"))[T.female]': "female",
    'C(t_e, Treatment(reference="white"))[T.asian]': "asian",
    'C(t_e, Treatment(reference="white"))[T.black]': "black",
    'C(t_e, Treatment(reference="white"))[T.hispanic]': "hispanic",
    'C(t_p, Treatment(reference="early_career"))[T.mid_career]': "mid_career",
    'C(t_p, Treatment(reference="early_career"))[T.late_career]': "late_career",
    "s_signal": "objective_signal",
}
EFFECT_ORDER = [
    "female",
    "asian",
    "black",
    "hispanic",
    "mid_career",
    "late_career",
    "objective_signal",
]
STEREOTYPE_ORDER = {"male": 0, "female": 1, "neutral": 2}
TIER_ORDER = {"high": 0, "mid": 1, "low": 2}


def _load_occupations() -> pd.DataFrame:
    cfg = tomllib.loads((REPO / "config/occupations.toml").read_text())
    rows = []
    for occ in cfg["occupation"]:
        rows.append(
            {
                "occupation_soc": str(occ["onet_soc"]),
                "title": str(occ["title"]),
                "stereotype": str(occ["stereotype"]),
                "skill_tier": str(occ["skill_tier"]),
                "regime": f"{occ['stereotype']}_{occ['skill_tier']}",
                "oes_may_2024": float(occ["oes_may_2024"]),
            }
        )
    occs = pd.DataFrame(rows)
    if len(occs) != 18:
        raise ValueError(f"expected 18 occupations, found {len(occs)}")
    duplicated = occs["occupation_soc"].duplicated()
    if duplicated.any():
        raise ValueError(
            f"duplicate occupations: {occs.loc[duplicated, 'occupation_soc'].tolist()}"
        )
    return occs


def _build_design_frame(occs: pd.DataFrame) -> pd.DataFrame:
    scores = pd.read_parquet(REPO / "data/audit/scores.parquet")
    treatments = pd.read_parquet(REPO / "data/processed/treatment_assignments.parquet")
    df = scores[["cell_id", "model_id", "hiring_score"]].merge(
        treatments[["cell_id", "resume_id", "t_g", "t_e", "t_p", "s_signal", "occupation_soc"]],
        on="cell_id",
        how="inner",
        validate="m:1",
    )
    df = df.merge(
        occs[["occupation_soc", "title", "stereotype", "skill_tier", "regime"]],
        on="occupation_soc",
        how="left",
        validate="m:1",
    )
    if df["regime"].isna().any():
        missing = sorted(df.loc[df["regime"].isna(), "occupation_soc"].unique())
        raise ValueError(f"occupations missing regime mapping: {missing}")
    if df["occupation_soc"].nunique() != 18:
        raise ValueError(
            f"expected 18 occupations in joined data, found {df['occupation_soc'].nunique()}"
        )
    return df.assign(s_signal=df["s_signal"].astype(bool).astype(int))


def _formula() -> str:
    return (
        'hiring_score ~ C(t_g, Treatment(reference="male"))'
        ' + C(t_e, Treatment(reference="white"))'
        ' + C(t_p, Treatment(reference="early_career"))'
        " + s_signal + C(occupation_soc) + C(model_id)"
    )


def _fit_one_regime(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    part = df.loc[df["regime"] == regime].copy()
    if part["occupation_soc"].nunique() < 2:
        raise ValueError(f"{regime} has fewer than two occupations")
    if part["model_id"].nunique() < 2:
        raise ValueError(f"{regime} has fewer than two models")

    res = smf.ols(formula=_formula(), data=part).fit(
        cov_type="cluster",
        cov_kwds={"groups": part["resume_id"].to_numpy()},
    )
    ci = res.conf_int(alpha=0.05)
    rows = []
    stereotype, skill_tier = regime.split("_", maxsplit=1)
    for term, effect in EFFECT_TERMS.items():
        if term not in res.params.index:
            raise ValueError(f"{regime}: missing coefficient {term}")
        rows.append(
            {
                "regime": regime,
                "stereotype": stereotype,
                "skill_tier": skill_tier,
                "effect": effect,
                "term": term,
                "coef": float(res.params[term]),
                "se": float(res.bse[term]),
                "t": float(res.tvalues[term]),
                "p": float(res.pvalues[term]),
                "ci_low": float(ci.loc[term, 0]),
                "ci_high": float(ci.loc[term, 1]),
                "n_obs": int(res.nobs),
                "n_clusters": int(part["resume_id"].nunique()),
                "n_occupations": int(part["occupation_soc"].nunique()),
                "n_models": int(part["model_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _sort_regimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_st"] = out["stereotype"].map(STEREOTYPE_ORDER)
    out["_tier"] = out["skill_tier"].map(TIER_ORDER)
    out["_effect"] = out["effect"].map({e: i for i, e in enumerate(EFFECT_ORDER)})
    return out.sort_values(["_st", "_tier", "_effect"]).drop(columns=["_st", "_tier", "_effect"])


def _wide_effects(ols: pd.DataFrame) -> pd.DataFrame:
    wide = ols.pivot_table(
        index=["regime", "stereotype", "skill_tier"],
        columns="effect",
        values="coef",
        aggfunc="first",
    ).reset_index()
    cols = ["regime", "stereotype", "skill_tier", *EFFECT_ORDER]
    wide = wide[cols].copy()
    wide["_st"] = wide["stereotype"].map(STEREOTYPE_ORDER)
    wide["_tier"] = wide["skill_tier"].map(TIER_ORDER)
    return wide.sort_values(["_st", "_tier"]).drop(columns=["_st", "_tier"])


def _regime_market_weights(occs: pd.DataFrame) -> pd.DataFrame:
    regimes = occs.groupby(["regime", "stereotype", "skill_tier"], as_index=False).agg(
        oes_may_2024=("oes_may_2024", "sum"), n_occupations=("occupation_soc", "size")
    )

    def norm(mask: pd.Series | None = None, values: pd.Series | None = None) -> pd.Series:
        w = pd.Series(0.0, index=regimes.index)
        if mask is None:
            mask = pd.Series(True, index=regimes.index)
        raw = values if values is not None else pd.Series(1.0, index=regimes.index)
        w.loc[mask] = raw.loc[mask].astype(float)
        total = float(w.sum())
        if total <= 0:
            raise ValueError("market weights sum to zero")
        return w / total

    specs = {
        "balanced_audit_market": norm(),
        "bls_oes_employment_weighted_market": norm(values=regimes["oes_may_2024"]),
        "high_skill_professional_market": norm(
            mask=regimes["skill_tier"].eq("high"), values=regimes["oes_may_2024"]
        ),
        "service_manual_market": norm(
            mask=regimes["skill_tier"].isin(["mid", "low"]), values=regimes["oes_may_2024"]
        ),
        "male_typed_market": norm(
            mask=regimes["stereotype"].eq("male"), values=regimes["oes_may_2024"]
        ),
        "female_typed_market": norm(
            mask=regimes["stereotype"].eq("female"), values=regimes["oes_may_2024"]
        ),
    }
    rows = []
    for market, weights in specs.items():
        part = regimes.copy()
        part["market"] = market
        part["weight"] = weights
        rows.append(part)
    return _sort_regimes(pd.concat(rows, ignore_index=True).assign(effect="female")).drop(
        columns=["effect"]
    )


def _transportability(ols: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    effects = ols[["regime", "effect", "coef"]].copy()
    weighted = effects.merge(weights[["market", "regime", "weight"]], on="regime", how="inner")
    out = (
        weighted.assign(weighted_coef=weighted["coef"] * weighted["weight"])
        .groupby(["market", "effect"], as_index=False)
        .agg(weighted_effect=("weighted_coef", "sum"))
    )
    baseline = out.loc[out["market"] == "balanced_audit_market", ["effect", "weighted_effect"]]
    baseline = baseline.rename(columns={"weighted_effect": "balanced_audit_effect"})
    out = out.merge(baseline, on="effect", how="left")
    out["transportability_gap"] = out["weighted_effect"] - out["balanced_audit_effect"]
    out["_effect"] = out["effect"].map({e: i for i, e in enumerate(EFFECT_ORDER)})
    market_order = {
        "balanced_audit_market": 0,
        "bls_oes_employment_weighted_market": 1,
        "high_skill_professional_market": 2,
        "service_manual_market": 3,
        "male_typed_market": 4,
        "female_typed_market": 5,
    }
    out["_market"] = out["market"].map(market_order)
    return out.sort_values(["_market", "_effect"]).drop(columns=["_market", "_effect"])


def main() -> int:
    occs = _load_occupations()
    df = _build_design_frame(occs)
    regime_counts = df.groupby(["stereotype", "skill_tier"])["occupation_soc"].nunique()
    if len(regime_counts) != 9 or not (regime_counts == 2).all():
        raise ValueError(f"bad regime coverage:\n{regime_counts}")

    regime_tables = [_fit_one_regime(df, regime) for regime in sorted(df["regime"].unique())]
    ols = _sort_regimes(pd.concat(regime_tables, ignore_index=True))
    if ols[["coef", "se", "p", "ci_low", "ci_high"]].isna().any().any():
        raise ValueError("role-regime OLS produced NaN values")

    wide = _wide_effects(ols)
    weights = _regime_market_weights(occs)
    transport = _transportability(ols, weights)

    out_dir = REPO / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    ols.to_csv(out_dir / "role_regime_ols.csv", index=False)
    wide.to_csv(out_dir / "role_regime_heterogeneity.csv", index=False)
    weights.to_csv(out_dir / "role_regime_weights.csv", index=False)
    transport.to_csv(out_dir / "transportability_sensitivity.csv", index=False)

    print("Regime coverage: 18 occupations mapped to 9 regimes; two occupations per regime.")
    print(f"Joined design rows: {len(df):,}; clusters: {df['resume_id'].nunique():,}.")
    print()
    print("Role-regime effects:")
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}"):
        print(wide.to_string(index=False))
    print()
    print("Transportability sensitivity:")
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}"):
        print(transport.to_string(index=False))
    print()
    for name in [
        "role_regime_ols.csv",
        "role_regime_heterogeneity.csv",
        "role_regime_weights.csv",
        "transportability_sensitivity.csv",
    ]:
        print(f"Wrote {(out_dir / name).relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
