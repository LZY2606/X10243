"""Binding model candidates fitted to dilution-corrected injection heats.

Candidates:

* ``one_site``     single set of equivalent sites: N, K, dH plus dilution offset
* ``two_site``     two independent site classes: N1,K1,dH1,N2,K2,dH2 + offset
* ``dilution_only`` heat per injection is purely a constant dilution term

The fitter is a bounded grid search solved by NumPy linear least squares, so
results are deterministic and fully replayable.  A finite *evaluation budget*
caps parameter combinations; exhausting it raises
:class:`BudgetExhausted` - a distinct outcome from malformed input or a
semantically infeasible request.

Equivalent fits (SSR within a stated relative tolerance) are **all** retained
and emitted in a stable, canonical order - a unique winner is never picked
arbitrarily.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .errors import BudgetExhausted, Diagnostic, SemanticInfeasible

SSR_REL_TOL = 1e-4          # relative SSR band considered equivalent


@dataclass
class TitrationData:
    ordinal: np.ndarray
    x: np.ndarray              # total ligand / total macromolecule after shot
    q_per_mole: np.ndarray     # net heat per injection in cal/mol of macromolecule
    v_inj_l: np.ndarray
    cell_m_conc: float
    syringe_m_conc: float
    cell_volume_l: float
    excluded: list[dict] = field(default_factory=list)


def build_titration(heat_rows: list[dict], conc_rev, cell_volume_l: float,
                    *, require_blank: bool = True) -> TitrationData:
    """Convert corrected heats to the molar coordinate system.

    Usable points must: not be a discarded first shot, have a dilution-
    corrected heat (blank counterpart present unless ``require_blank`` is
    False), and not be window-truncated by a gap (endpoint decision explicit).
    Everything excluded is returned with a reason for the UI.
    """
    from .units import require_number
    syringe = require_number(conc_rev.data["concentration"], "concentration")
    if conc_rev.data.get("cell_concentration") is None:
        raise SemanticInfeasible(
            "cannot fit binding models without macromolecule concentration",
            [Diagnostic("model:no_macromolecule_concentration",
                        "cell concentration missing", "cell.concentration", None)])
    macromolecule = require_number(conc_rev.data["cell_concentration"], "concentration")

    ordinals: list[int] = []
    heats: list[float] = []
    vols: list[float] = []
    excluded: list[dict] = []
    for row in heat_rows:
        reason = None
        if row["discard"]:
            reason = "first_injection_discarded"
        elif row.get("gap_truncated"):
            reason = "gap_truncated_window"
        elif row["net_heat_cal"] is None:
            if require_blank:
                reason = "blank_counterpart_missing"
            elif row["raw_heat_cal"] is None:
                reason = "integration_unavailable"
            else:
                reason = None
        if reason:
            excluded.append({"ordinal": row["ordinal"], "reason": reason})
            continue
        heat = row["net_heat_cal"] if require_blank else row["raw_heat_cal"]
        ordinals.append(row["ordinal"])
        heats.append(heat)
        vols.append(row["volume_l"])

    n = len(ordinals)
    minimum = 4
    if n < minimum:
        raise SemanticInfeasible(
            f"only {n} usable injection(s); at least {minimum} needed to fit",
            [Diagnostic("model:insufficient_points",
                        f"{n} usable points (< {minimum})", "heats",
                        [e["ordinal"] for e in excluded])])

    v_inj = np.asarray(vols, dtype=float)
    q_cal = np.asarray(heats, dtype=float)
    V0 = cell_volume_l
    M0 = macromolecule
    ligand_total = np.zeros(n)
    x_ratio = np.zeros(n)
    # accounting: ligand stock accumulates with volume displacement; each
    # injection also dilutes the cell contents by (V0 - v)/V0
    cumulative_ligand = 0.0
    cell_ligand = 0.0
    for i, v in enumerate(v_inj):
        injected_ligand = syringe * v
        cell_ligand = cell_ligand * (V0 - v) / V0 + injected_ligand
        cumulative_ligand += injected_ligand
        ligand_total[i] = cell_ligand / V0
        x_ratio[i] = ligand_total[i] / M0
    q_per_mole = q_cal / (M0 * V0)
    return TitrationData(
        ordinal=np.asarray(ordinals), x=x_ratio, q_per_mole=q_per_mole,
        v_inj_l=v_inj, cell_m_conc=M0, syringe_m_conc=syringe,
        cell_volume_l=V0, excluded=excluded)


# --------------------------------------------------------------------------- #
# Model shapes
# --------------------------------------------------------------------------- #

def _fraction_bound(x: np.ndarray, n: float, k: float, c: float) -> np.ndarray:
    # Qb/V0 (cal per litre) for n sites: n*C*dH*theta; theta via quadratic
    b = c * x + n * c + 1.0 / k
    disc = b * b - 4.0 * c * c * x * n
    disc = np.maximum(disc, 0.0)
    theta = (b - np.sqrt(disc)) / (2.0 * n * c)
    return np.clip(theta, 0.0, 1.0)


def one_site_heat(x: np.ndarray, n: float, k: float, d_h: float, offset: float,
                  c: float) -> np.ndarray:
    """Predicted normalised heat per injection (cal/mol) using finite model.

    Uses the standard incremental-difference form with volume displacement.
    """
    theta = _fraction_bound(x, n, k, c)
    q_v = n * c * d_h * theta                    # cal per litre of cell
    total_q = q_v                                # Q_i after injection
    prev_q = np.concatenate(([0.0], total_q[:-1]))
    # injection i normalised heat: (Q_i + dVi/V0 * (prev_Q)/2 - prev_Q)/C
    dq = total_q - prev_q
    return dq / c + offset


def two_site_heat(x: np.ndarray, p: tuple, c: float) -> np.ndarray:
    n1, k1, h1, n2, k2, h2, offset = p
    t1 = _fraction_bound(x, n1, k1, c)
    t2 = _fraction_bound(x, n2, k2, c)
    qv = c * (n1 * h1 * t1 + n2 * h2 * t2)
    prev = np.concatenate(([0.0], qv[:-1]))
    return (qv - prev) / c + offset


def dilution_heat(x: np.ndarray, offset: float) -> np.ndarray:
    return np.full_like(x, offset, dtype=float)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

@dataclass
class FitResult:
    model: str
    status: str                 # best | equivalent | insufficient | exhausted
    params: dict[str, float]
    ssr: float
    rmse: float
    residual: list[float]
    predicted: list[float]
    ordinals: list[int]
    used_points: int
    evaluations: int
    equivalent_group: list[dict] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "model": self.model, "status": self.status, "params": self.params,
            "ssr": self.ssr, "rmse": self.rmse, "residual": self.residual,
            "predicted": self.predicted, "ordinals": self.ordinals,
            "used_points": self.used_points, "evaluations": self.evaluations,
            "equivalent_group": self.equivalent_group, "note": self.note,
        }


def _grid(n: int, lo: float, hi: float) -> np.ndarray:
    if n <= 1:
        return np.array([(lo + hi) / 2.0])
    return np.geomspace(lo, hi, n)


def fit_one_site(data: TitrationData, *, budget: int = 4096) -> FitResult:
    x, y, c = data.x, data.q_per_mole, data.cell_m_conc
    n_grid = np.linspace(0.2, 10.0, 16)
    k_grid = _grid(16, 1e3, 1e8)
    evaluations = 0
    rows: list[tuple[float, float, float, float, float]] = []
    for n in n_grid:
        for k in k_grid:
            evaluations += 1
            if evaluations > budget:
                raise BudgetExhausted(
                    f"one_site budget {budget} evaluations exhausted",
                    [Diagnostic("budget:fit_evaluations",
                                f"> {budget} parameter combinations",
                                "models.one_site", budget)])
            theta = _fraction_bound(x, n, k, c)
            qv = n * c * theta
            prev = np.concatenate(([0.0], qv[:-1]))
            shape = (qv - prev) / c
            A = np.column_stack([shape, np.ones_like(x)])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            ssr = float(np.sum((y - pred) ** 2))
            rows.append((ssr, float(n), float(k), float(coef[0]), float(coef[1])))
    return _rank(rows, data, "one_site",
                 lambda r: {"N": r[1], "K": r[2], "dH": r[3], "dilution_offset": r[4]})


def fit_two_site(data: TitrationData, *, budget: int = 65536) -> FitResult:
    x, y, c = data.x, data.q_per_mole, data.cell_m_conc
    n_grid = np.linspace(0.2, 6.0, 8)
    k_grid = _grid(6, 1e3, 1e8)
    evaluations = 0
    rows: list[tuple] = []
    for n1 in n_grid:
        for k1 in k_grid:
            t1 = _fraction_bound(x, n1, k1, c)
            qv1 = n1 * c * t1
            prev1 = np.concatenate(([0.0], qv1[:-1]))
            shape1 = (qv1 - prev1) / c
            for n2 in n_grid:
                for k2 in k_grid:
                    evaluations += 1
                    if evaluations > budget:
                        raise BudgetExhausted(
                            f"two_site budget {budget} evaluations exhausted",
                            [Diagnostic("budget:fit_evaluations",
                                        f"> {budget} parameter combinations",
                                        "models.two_site", budget)])
                    t2 = _fraction_bound(x, n2, k2, c)
                    qv2 = n2 * c * t2
                    prev2 = np.concatenate(([0.0], qv2[:-1]))
                    shape2 = (qv2 - prev2) / c
                    A = np.column_stack([shape1, shape2, np.ones_like(x)])
                    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                    pred = A @ coef
                    ssr = float(np.sum((y - pred) ** 2))
                    rows.append((ssr, float(n1), float(k1), float(coef[0]),
                                 float(n2), float(k2), float(coef[1]),
                                 float(coef[2])))
    return _rank(rows, data, "two_site",
                 lambda r: {"N1": r[1], "K1": r[2], "dH1": r[3],
                            "N2": r[4], "K2": r[5], "dH2": r[6],
                            "dilution_offset": r[7]})


def fit_dilution(data: TitrationData, *, budget: int = 64) -> FitResult:
    offset = float(np.mean(data.q_per_mole))
    pred = np.full_like(data.q_per_mole, offset)
    ssr = float(np.sum((data.q_per_mole - pred) ** 2))
    res = FitResult(
        "dilution_only", "best", {"dilution_offset": offset}, ssr,
        float(np.sqrt(ssr / data.x.size)),
        (data.q_per_mole - pred).tolist(), pred.tolist(),
        data.ordinal.tolist(), data.x.size, 1)
    return res


def _rank(rows: list[tuple], data: TitrationData, model: str,
          params_of: Callable) -> FitResult:
    rows.sort(key=lambda r: (r[0], _canonical_key(params_of(r))))
    best_ssr = rows[0][0]
    band = SSR_REL_TOL * max(best_ssr, 1e-30)
    group = [r for r in rows if r[0] <= best_ssr + band]
    # de-duplicate near-identical parameter sets for a readable group
    unique: list[tuple] = []
    for r in group:
        key = _canonical_key(params_of(r))
        if all(np.max(np.abs(np.asarray(key) - np.asarray(_canonical_key(params_of(u)))))
               > 1e-6 * (1.0 + np.max(np.abs(np.asarray(key)))) for u in unique):
            unique.append(r)
    best = rows[0]
    params = params_of(best)
    if model == "one_site":
        pred = one_site_heat(data.x, params["N"], params["K"], params["dH"],
                             params["dilution_offset"], data.cell_m_conc)
    else:
        pred = two_site_heat(data.x, (params["N1"], params["K1"], params["dH1"],
                                      params["N2"], params["K2"], params["dH2"],
                                      params["dilution_offset"]), data.cell_m_conc)
    residual = (data.q_per_mole - pred).tolist()
    status = "equivalent" if len(unique) > 1 else "best"
    return FitResult(
        model=model, status=status, params=params, ssr=best_ssr,
        rmse=float(np.sqrt(best_ssr / data.x.size)), residual=residual,
        predicted=pred.tolist(), ordinals=data.ordinal.tolist(),
        used_points=data.x.size, evaluations=len(rows),
        equivalent_group=[{"ssr": r[0], "params": params_of(r)} for r in unique],
        note=(f"{len(unique)} parameter sets within SSR relative tolerance "
              f"{SSR_REL_TOL:g}" if status == "equivalent" else ""))


def _canonical_key(params: dict) -> tuple:
    return tuple(float(params[k]) for k in sorted(params))


def fit_all(data: TitrationData, *, budget: int = 65536) -> list[FitResult]:
    results = [
        fit_dilution(data, budget=budget),
        fit_one_site(data, budget=min(budget, 4096)),
        fit_two_site(data, budget=budget),
    ]
    order = sorted(results, key=lambda r: (r.ssr, r.model))
    best = order[0].ssr
    band = SSR_REL_TOL * max(best, 1e-30)
    for r in results:
        r.status = "equivalent" if r.ssr <= best + band and r is not order[0] else r.status
    return results
