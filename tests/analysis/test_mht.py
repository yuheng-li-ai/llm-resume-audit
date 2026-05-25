"""Phase 7.3 unit tests — MHTCorrector (Romano-Wolf / List-Shaikh-Xu step-down)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from llm_audit.analysis.mht import MHTConfig, MHTCorrector, MHTResult


def _make_bootstrap_fn(
    keys: list[str],
    cov: np.ndarray,
    rng: np.random.Generator,
) -> Callable[[int], dict[str, float]]:
    """Return a callable that simulates a bootstrap draw of stats from N(0, cov).

    keys: ordered list of contrast names matching cov dimensions
    cov: KxK positive-definite covariance matrix for the studentized null distribution.

    The returned callable accepts a seed (used only as reproducibility marker)
    and returns dict[key] -> bootstrap statistic.
    """

    def _draw(_seed: int) -> dict[str, float]:
        z = rng.multivariate_normal(np.zeros(cov.shape[0]), cov)
        return {k: float(z[i]) for i, k in enumerate(keys)}

    return _draw


class TestMHTConfig:
    def test_defaults(self) -> None:
        cfg = MHTConfig()
        assert cfg.n_bootstrap >= 500
        assert 0 < cfg.alpha < 1
        assert cfg.random_state is not None


class TestMHTCorrectorAPI:
    def test_returns_one_result_per_observed_stat(self) -> None:
        keys = ["a", "b", "c"]
        cov = np.eye(3)
        rng = np.random.default_rng(7)
        boot_fn = _make_bootstrap_fn(keys, cov, rng)
        observed = {"a": 3.0, "b": 1.0, "c": 0.5}
        results = MHTCorrector(MHTConfig(n_bootstrap=500, random_state=7)).fit(observed, boot_fn)
        assert len(results) == 3
        names = {r.contrast_name for r in results}
        assert names == set(keys)
        for r in results:
            assert isinstance(r, MHTResult)
            assert 0.0 <= r.adjusted_p <= 1.0
            assert 0.0 <= r.raw_p <= 1.0

    def test_raises_when_bootstrap_dict_misaligned(self) -> None:
        observed = {"a": 1.0}

        def bad_boot(_seed: int) -> dict[str, float]:
            return {"different_key": 0.0}

        with pytest.raises(ValueError, match="contrast names"):
            MHTCorrector(MHTConfig(n_bootstrap=100, random_state=7)).fit(observed, bad_boot)


class TestStepdownProperties:
    def test_adjusted_p_geq_raw_p(self) -> None:
        keys = ["a", "b", "c", "d", "e"]
        cov = np.eye(5)
        rng = np.random.default_rng(7)
        boot_fn = _make_bootstrap_fn(keys, cov, rng)
        observed = {"a": 3.0, "b": 2.5, "c": 2.0, "d": 1.5, "e": 0.5}
        results = MHTCorrector(MHTConfig(n_bootstrap=1000, random_state=7)).fit(observed, boot_fn)
        for r in results:
            assert r.adjusted_p >= r.raw_p - 1e-6

    def test_adjusted_p_monotone_in_descending_test_stat(self) -> None:
        keys = ["a", "b", "c", "d"]
        cov = np.eye(4)
        rng = np.random.default_rng(7)
        boot_fn = _make_bootstrap_fn(keys, cov, rng)
        observed = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
        results = MHTCorrector(MHTConfig(n_bootstrap=1000, random_state=7)).fit(observed, boot_fn)
        results_by_name = {r.contrast_name: r for r in results}
        # When |stat| decreases, adjusted_p should be non-decreasing
        prev = -1.0
        for k in keys:  # observed sorted descending
            p = results_by_name[k].adjusted_p
            assert p >= prev - 1e-6
            prev = p


class TestPowerVsBonferroni:
    def test_adjusted_p_no_larger_than_bonferroni(self) -> None:
        """LSX bootstrap should be at least as powerful as Bonferroni when
        contrasts are correlated."""
        keys = ["a", "b", "c", "d"]
        cov = 0.2 * np.eye(4) + 0.8 * np.ones((4, 4))  # strong positive correlation
        rng = np.random.default_rng(7)
        boot_fn = _make_bootstrap_fn(keys, cov, rng)
        observed = {"a": 3.5, "b": 2.8, "c": 2.0, "d": 0.5}
        results = MHTCorrector(MHTConfig(n_bootstrap=2000, random_state=7)).fit(observed, boot_fn)
        k = len(observed)
        for r in results:
            bonf = min(1.0, r.raw_p * k)
            # LSX must not be more conservative than Bonferroni (within MC noise)
            assert (
                r.adjusted_p <= bonf + 0.05
            ), f"{r.contrast_name}: adj_p={r.adjusted_p:.4f} > bonf={bonf:.4f}"

    def test_independent_contrasts_close_to_sidak(self) -> None:
        """Under independence the adjusted-p of the most-significant contrast
        should be close to 1 - (1 - raw_p)^K (Sidak)."""
        keys = [f"k{i}" for i in range(5)]
        cov = np.eye(5)
        rng = np.random.default_rng(11)
        boot_fn = _make_bootstrap_fn(keys, cov, rng)
        observed = {k: 0.0 for k in keys}
        observed["k0"] = 2.0  # only k0 is meaningful
        results = MHTCorrector(MHTConfig(n_bootstrap=4000, random_state=11)).fit(observed, boot_fn)
        r = next(r for r in results if r.contrast_name == "k0")
        sidak = 1.0 - (1.0 - r.raw_p) ** 5
        # 0.05 absolute tolerance: bootstrap MC noise + finite-sample drift
        assert abs(r.adjusted_p - sidak) < 0.05, f"adj_p={r.adjusted_p:.4f} vs Sidak={sidak:.4f}"


class TestNullCases:
    def test_all_zero_observed_gives_high_p(self) -> None:
        keys = ["a", "b", "c"]
        rng = np.random.default_rng(7)
        boot_fn = _make_bootstrap_fn(keys, np.eye(3), rng)
        observed = {k: 0.0 for k in keys}
        results = MHTCorrector(MHTConfig(n_bootstrap=500, random_state=7)).fit(observed, boot_fn)
        for r in results:
            assert r.adjusted_p > 0.5

    def test_one_extreme_observed_gives_low_p(self) -> None:
        keys = ["a", "b", "c"]
        rng = np.random.default_rng(7)
        boot_fn = _make_bootstrap_fn(keys, np.eye(3), rng)
        observed = {"a": 10.0, "b": 0.0, "c": 0.0}
        results = MHTCorrector(MHTConfig(n_bootstrap=500, random_state=7)).fit(observed, boot_fn)
        by_name = {r.contrast_name: r for r in results}
        assert by_name["a"].adjusted_p < 0.05
        assert by_name["b"].adjusted_p > 0.5
        assert by_name["c"].adjusted_p > 0.5
