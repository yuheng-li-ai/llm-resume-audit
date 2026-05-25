"""Phase 7.3 — Multiple-hypothesis correction via Romano-Wolf step-down bootstrap.

Implements the family-wise error rate (FWER) controlling step-down procedure
of List, Shaikh, and Xu (2019) (a generalisation of Romano-Wolf 2005). The
procedure is more powerful than Bonferroni when contrasts are correlated.

Algorithm (per LSX 2019 Algorithm 1):

  1. Compute observed |T_k| for k=1..K and sort indices by descending |T_k|.
  2. Generate B bootstrap replications of the (studentized) null distribution
     T*_b[k] = (theta*_b[k] - theta_hat[k]) / SE_b[k].
  3. For the most-extreme observed |T_(1)|:
         p_adj_(1) = (1 + sum_b 1[ max_k |T*_b[k]| >= |T_(1)| ]) / (B+1)
  4. For the next observed |T_(2)|:
         active = indices not yet rejected (i.e., indices ranked >= 2)
         p_adj_(2) = (1 + sum_b 1[ max_{k in active} |T*_b[k]| >= |T_(2)| ]) / (B+1)
  5. Enforce monotonicity: p_adj_(j) = max(p_adj_(j-1), p_adj_(j)).

The runner supplies `bootstrap_stats_fn` that returns a dict[name] -> float
of bootstrap stats for each replication. Decoupling the bootstrap loop from
the correction loop lets callers plug in any cluster bootstrap on a fitting
procedure of their choice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class MHTConfig:
    """Configuration for the Romano-Wolf / LSX step-down corrector."""

    n_bootstrap: int = 1000
    alpha: float = 0.05
    random_state: int = 7


@dataclass(frozen=True)
class MHTResult:
    """One adjusted contrast."""

    contrast_name: str
    observed_stat: float
    raw_p: float
    adjusted_p: float


class MHTCorrector:
    """Romano-Wolf / List-Shaikh-Xu step-down family-wise error corrector."""

    def __init__(self, config: MHTConfig | None = None) -> None:
        self._config = config or MHTConfig()

    @property
    def config(self) -> MHTConfig:
        return self._config

    def fit(
        self,
        observed_stats: dict[str, float],
        bootstrap_stats_fn: Callable[[int], dict[str, float]],
    ) -> list[MHTResult]:
        """Apply step-down correction.

        observed_stats: {name -> observed test statistic (z-score)}.
        bootstrap_stats_fn(seed): returns {name -> studentized bootstrap stat}
            for one replication. Must return all the same keys as observed_stats
            on every call. The corrector calls it n_bootstrap times with seeds
            derived from config.random_state.
        """
        cfg = self._config
        names = list(observed_stats.keys())
        observed = np.array([observed_stats[n] for n in names], dtype=float)
        abs_obs = np.abs(observed)

        boot_matrix = np.zeros((cfg.n_bootstrap, len(names)), dtype=float)
        for b in range(cfg.n_bootstrap):
            draw = bootstrap_stats_fn(cfg.random_state + b)
            if set(draw.keys()) != set(names):
                raise ValueError(
                    "bootstrap_stats_fn returned different contrast names than "
                    f"observed_stats; observed={set(names)}, got={set(draw.keys())}"
                )
            for j, n in enumerate(names):
                boot_matrix[b, j] = abs(draw[n])

        # raw two-sided z-test p-values
        raw_p = 2.0 * (1.0 - stats.norm.cdf(abs_obs))

        # Step-down in descending |observed| order; remove each index from the
        # active set after processing (so subsequent contrasts compare against
        # the supremum of the remaining bootstrap stats).
        order = np.argsort(-abs_obs)
        adjusted = np.empty(len(names), dtype=float)
        prev_p = 0.0
        active_mask = np.ones(len(names), dtype=bool)
        for idx in order:
            obs_k = abs_obs[idx]
            active_cols = np.where(active_mask)[0]
            max_boot = boot_matrix[:, active_cols].max(axis=1)
            # Romano-Wolf bias-correction: (1 + count) / (B + 1), bounded by [0, 1]
            p_rw = (1.0 + float(np.sum(max_boot >= obs_k))) / (cfg.n_bootstrap + 1.0)
            p = min(1.0, max(p_rw, prev_p))  # enforce monotonicity + cap at 1
            adjusted[idx] = p
            prev_p = p
            active_mask[idx] = False

        return [
            MHTResult(
                contrast_name=names[i],
                observed_stat=float(observed[i]),
                raw_p=float(raw_p[i]),
                adjusted_p=float(adjusted[i]),
            )
            for i in range(len(names))
        ]
