"""Binding-model candidate fitting: one-site, two-site and dilution.

Rule set version: ``binding-v1``.

Thermodynamic convention (stored in every result's ``inputs``): the cell is
treated as a single stirred compartment of fixed volume ``V0`` (the classic
"displaced volume / no cell-volume change" Wiseman convention). After shot
``i`` with volume ``v_i`` of ligand at ``L0`` into cell concentration ``M0``:

    X_i = sum_k<=i v_k * L0 / V0          (total ligand per V0, mM)

and the one-site free concentration ``Lf`` solves the mass balance

    X = Lf + M0 * n * K Lf / (1 + K Lf).

Observed heats are per-shot normalised differential heats ``q_i`` (kcal/mol of
injectant); the first (discarded) shot is excluded.

Only NumPy is used. The nonlinear search is a coarse grid plus golden-section
coordinate refinement; each evaluation of the nonlinear parameters costs one
budget unit, and exceeding ``budget`` raises :class:`BudgetExhausted` (a
distinct outcome from format or semantic failures).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .errors import BudgetExhausted, Diagnostic, SemanticImpasse
from .materials import content_hash

RULE_VERSION = "binding-v1"

UCAL_PER_KCAL_MOL_MM_ML = 1000.0  # kcal/mol * mmol/L * mL  -> ucal (see notes)


def _ucal_to_kcal_per_mol(net_ucal: np.ndarray, v_uL: np.ndarray,
                          l0_mM: float) -> np.ndarray:
    """ucal heat -> kcal/mol injectant.

    moles injectant in shot i = (v_i uL)(1 L/1e6 uL)(l0 mmol/mL?) ->
    v[uL] * L0[mM] * 1e-6 mmol ; heat[kcal] = area[u cal] * 1e-6 kcal/u cal.
    q[kcal/mol] = area_u cal / (v_uL * L0_mM)   (units cancel to kcal/mol).
    """
    return net_ucal / (v_uL * l0_mM)


def titrant_ratio(v_uL: np.ndarray, v0_uL: float) -> np.ndarray:
    return np.cumsum(v_uL) / v0_uL


@dataclass
class Dataset:
    order: np.ndarray
    injection_id: list[str]
    x: np.ndarray                     # cumulative ligand / cell, mM-equivalent
    q: np.ndarray                     # observed differential heat, kcal/mol
    v: np.ndarray                     # per-shot volumes uL

    def to_compact(self) -> dict[str, Any]:
        return {"orders": self.order.tolist(),
                "injection_ids": self.injection_id,
                "x_mM": self.x.tolist(), "q_kcal_per_mol": self.q.tolist(),
                "volume_uL": self.v.tolist()}


def build_dataset(adjusted: dict[str, Any], concentration_mat,
                  cell_volume_mat) -> tuple[Dataset | None, list[dict[str, Any]]]:
    """Assemble the model dataset from a dilution-adjusted result.

    A concentration revision never alters raw integration: only this derived
    layer re-reads the concentration material. Shots without a net heat (blank
    counterpart missing, blocked window, discarded first) are excluded with a
    recorded reason.
    """
    diags: list[dict[str, Any]] = []
    conc = {e["key"]: e["quantity"] for e in concentration_mat.payload["entries"]}
    l0q = conc.get("syringe_concentration")
    if l0q is None or not l0q["convertible"]:
        raise SemanticImpasse(
            "syringe concentration missing or unconvertible",
            [Diagnostic("missing_concentration",
                        "model needs a convertible syringe concentration",
                        concentration_mat.file)])
    l0 = float(l0q["canonical"])
    if l0 <= 0:
        raise SemanticImpasse("syringe concentration must be positive",
                              [Diagnostic("bad_concentration", "L0 <= 0",
                                          concentration_mat.file)])
    v0 = None
    for key, q in ((e["key"], e["quantity"])
                   for e in cell_volume_mat.payload["entries"]):
        if key == "cell_volume" and q["convertible"]:
            v0 = float(q["canonical"])
    if v0 is None or v0 <= 0:
        raise SemanticImpasse("cell volume missing or unconvertible",
                              [Diagnostic("missing_cell_volume",
                                          "model needs convertible V0",
                                          cell_volume_mat.file)])
    orders, ids, qlist, vlist, excluded = [], [], [], [], []
    for r in adjusted["injections"]:
        if r["status"] == "discarded_first":
            excluded.append((r["injection_id"], "discarded_first"))
            continue
        if r["status"] != "integrated":
            excluded.append((r["injection_id"], f"sample_{r['status']}"))
            continue
        if r.get("net_area_ucal") is None:
            excluded.append((r["injection_id"], r.get("dilution_status", "no_net")))
            continue
        orders.append(r["order"])
        ids.append(r["injection_id"])
        qlist.append(r["net_area_ucal"])
        vlist.append(r["injection_volume_uL"])
    for iid, reason in excluded:
        diags.append(Diagnostic("point_excluded",
                                "point excluded from binding fit",
                                detail={"injection_id": iid, "reason": reason}).to_dict())
    if not orders:
        diags.append(Diagnostic("empty_dataset", "no usable points for fitting",
                                detail=dict(excluded)).to_dict())
        return None, diags
    v = np.asarray(vlist, dtype=float)
    x = np.cumsum(v) * l0 / v0
    q = _ucal_to_kcal_per_mol(np.asarray(qlist, dtype=float), v, l0)
    return Dataset(order=np.asarray(orders, dtype=int), injection_id=ids,
                   x=x, q=q, v=v), diags


# ---------------------------------------------------------------------------
# Model thermodynamics
# ---------------------------------------------------------------------------
def _free_one_site(x: np.ndarray, m0: float, n: float, logk: float) -> np.ndarray:
    # logk is log10(K / M^-1); equations use mM, so K_mM = K_M / 1000.
    K = (10.0 ** logk) / 1000.0
    lf = np.empty_like(x)
    for i, xi in enumerate(x):
        b = m0 * n + 1.0 / K - xi
        c = -xi / K
        disc = b * b - 4.0 * c
        if disc < 0:
            disc = 0.0
        lf[i] = (-b + np.sqrt(disc)) / 2.0
    return np.maximum(lf, 0.0)


def bound_one_site(x: np.ndarray, m0: float, n: float, logk: float) -> np.ndarray:
    lf = _free_one_site(x, m0, n, logk)
    K = (10.0 ** logk) / 1000.0
    return m0 * n * K * lf / (1.0 + K * lf)


def one_site_dq(x: np.ndarray, v: np.ndarray, v0: float, l0: float,
                m0: float, n: float, logk: float, dh: float) -> np.ndarray:
    theta_sum = bound_one_site(x, m0, n, logk)
    bound_moles = theta_sum * v0  # mM*uL proportional units
    delta = np.empty_like(x)
    delta[0] = bound_moles[0]
    delta[1:] = np.diff(bound_moles)
    inj_moles = v * l0
    return dh * delta / inj_moles


def _free_two_site(x: np.ndarray, m0: float, f1: float,
                   logk1: float, logk2: float) -> np.ndarray:
    K1, K2 = (10.0 ** logk1) / 1000.0, (10.0 ** logk2) / 1000.0
    m1, m2 = m0 * f1, m0 * (1.0 - f1)
    lf = np.empty_like(x)

    def total(lf_v: float, xi: float) -> float:
        b1 = m1 * K1 * lf_v / (1.0 + K1 * lf_v)
        b2 = m2 * K2 * lf_v / (1.0 + K2 * lf_v)
        return lf_v + b1 + b2 - xi

    lo, hi = 0.0, max(float(x[-1]) * 1.2, 1.0)
    for i, xi in enumerate(x):
        a, b = lo, hi
        while total(b, xi) < 0:
            b *= 2.0
        for _ in range(60):
            mid = 0.5 * (a + b)
            if total(mid, xi) > 0:
                b = mid
            else:
                a = mid
        lf[i] = 0.5 * (a + b)
        lo = max(0.0, lf[i] * 0.5)
    return lf


def bound_two_site(x: np.ndarray, m0: float, f1: float,
                   logk1: float, logk2: float) -> np.ndarray:
    lf = _free_two_site(x, m0, f1, logk1, logk2)
    K1, K2 = (10.0 ** logk1) / 1000.0, (10.0 ** logk2) / 1000.0
    return (m0 * f1 * K1 * lf / (1.0 + K1 * lf)
            + m0 * (1.0 - f1) * K2 * lf / (1.0 + K2 * lf))


def two_site_dq(x: np.ndarray, v: np.ndarray, v0: float, l0: float,
                m0: float, f1: float, logk1: float, logk2: float,
                dh1: float, dh2: float) -> np.ndarray:
    K1, K2 = (10.0 ** logk1) / 1000.0, (10.0 **  logk2) / 1000.0
    lf = _free_two_site(x, m0, f1, logk1, logk2)
    b1 = m0 * f1 * K1 * lf / (1.0 + K1 * lf) * v0
    b2 = m0 * (1.0 - f1) * K2 * lf / (1.0 + K2 * lf) * v0
    db1 = np.empty_like(x)
    db2 = np.empty_like(x)
    db1[0], db2[0] = b1[0], b2[0]
    db1[1:], db2[1:] = np.diff(b1), np.diff(b2)
    inj = v * l0
    return (dh1 * db1 + dh2 * db2) / inj


def dilution_dq(v: np.ndarray, l0: float, q_dil: float) -> np.ndarray:
    """Constant heat of dilution: same kcal/mol for every shot."""
    return np.full_like(v, q_dil, dtype=float)


# ---------------------------------------------------------------------------
# Budgeted nonlinear search (NumPy only)
# ---------------------------------------------------------------------------
@dataclass
class Budget:
    limit: int
    used: int = 0

    def tick(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise BudgetExhausted(
                "fit evaluation budget exhausted before convergence",
                self.used, self.limit,
                [Diagnostic("budget_exhausted",
                            "nonlinear parameter evaluations exceeded limit",
                            detail={"used": self.used, "limit": self.limit})])


def _linear_dh(y: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    """Non-negative-free OLS of y on k basis columns."""
    A = basis.T  # (k, n)
    gram = A @ A.T
    rhs = A @ y
    try:
        coef, *_ = np.linalg.lstsq(
            gram + np.eye(k) * 1e-12 * np.trace(gram) / max(k, 1),
            rhs, rcond=None)
    except np.linalg.LinAlgError:
        coef = np.zeros(k)
    return coef


def _golden(budget: Budget, lo: float, hi: float, fixed: list[float],
            index: int, bounds: list[tuple[float, float]],
            loss: Callable[[list[float]], float], iters: int = 10) -> float:
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)

    def at(v: float) -> float:
        z = list(fixed)
        z[index] = v
        budget.tick()
        return loss(z)

    fc, fd = at(c), at(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = at(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = at(d)
    return 0.5 * (a + b)


def _coordinate_refine(budget: Budget, start: list[float],
                       bounds: list[tuple[float, float]],
                       loss: Callable[[list[float]], float],
                       rounds: int = 2) -> list[float]:
    z = list(start)
    for _ in range(rounds):
        for index, (lo, hi) in enumerate(bounds):
            span = hi - lo
            local_lo = max(lo, z[index] - 0.35 * span)
            local_hi = min(hi, z[index] + 0.35 * span)
            z[index] = _golden(budget, local_lo, local_hi, z, index,
                               bounds, loss, iters=8)
    return z


def _aicc(sse: float, n: int, k: int) -> float:
    if n - k - 1 <= 0:
        return float("inf")
    if sse <= 0:
        sse = 1e-12
    return n * np.log(sse / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def _fit_one_site(ds: Dataset, ctx: dict[str, float], budget: Budget
                  ) -> dict[str, Any]:
    m0, l0, v0 = ctx["m0"], ctx["l0"], ctx["v0"]
    x, v, y = ds.x, ds.v, ds.q

    def predict(p: list[float]) -> np.ndarray:
        n, logk = p
        bsum = bound_one_site(x, m0, n, logk)
        bm = bsum * v0
        delta = np.empty_like(x)
        delta[0] = bm[0]
        delta[1:] = np.diff(bm)
        return delta / (v * l0)

    def loss(p: list[float]) -> float:
        pred = predict(p)
        basis = pred.reshape(1, -1)
        dh = _linear_dh(y, basis, 1)[0]
        resid = y - dh * pred
        return float(np.sum(resid * resid))

    logk_grid = np.linspace(2.0, 9.0, 15)
    n_grid = np.linspace(0.2, 3.0, 8)
    best = None
    for lk in logk_grid:
        for nn in n_grid:
            budget.tick()
            val = loss([nn, lk])
            if best is None or val < best[0]:
                best = (val, [nn, lk])
    p = _coordinate_refine(budget, best[1], [(0.05, 5.0), (1.0, 10.0)], loss)
    pred = predict(p)
    dh = _linear_dh(y, pred.reshape(1, -1), 1)[0]
    fitted = dh * pred
    sse = float(np.sum((y - fitted) ** 2))
    return {
        "model": "one_site", "k_params": 3, "converged": True,
        "parameters": {
            "n_sites": {"value": float(p[0]), "unit": "sites/cell molecule",
                        "raw_basis": "model"},
            "K": {"value": float(10.0 ** p[1]), "unit": "M^-1",
                  "raw_basis": "model"},
            "log10_K": {"value": float(p[1]), "unit": "log10(M^-1)",
                        "raw_basis": "model"},
            "delta_H": {"value": float(dh), "unit": "kcal/mol",
                        "raw_basis": "model"},
        },
        "sse": sse, "predicted": fitted.tolist(),
        "residuals": (y - fitted).tolist(),
    }


def _fit_dilution(ds: Dataset, ctx: dict[str, float], budget: Budget
                  ) -> dict[str, Any]:
    budget.tick()
    y = ds.q
    qd = float(np.mean(y))
    fitted = np.full_like(y, qd)
    sse = float(np.sum((y - fitted) ** 2))
    return {
        "model": "dilution", "k_params": 1, "converged": True,
        "parameters": {
            "heat_of_dilution": {"value": qd, "unit": "kcal/mol injectant",
                                 "raw_basis": "model"},
        },
        "sse": sse, "predicted": fitted.tolist(),
        "residuals": (y - fitted).tolist(),
    }


def _fit_two_site(ds: Dataset, ctx: dict[str, float], budget: Budget
                  ) -> dict[str, Any]:
    m0, l0, v0 = ctx["m0"], ctx["l0"], ctx["v0"]
    x, v, y = ds.x, ds.v, ds.q
    n_pts = len(x)

    def basis_for(p: list[float]) -> np.ndarray:
        f1, lk1, lk2 = p
        K1, K2 = (10.0 ** lk1) / 1000.0, (10.0 ** lk2) / 1000.0
        lf = _free_two_site(x, m0, f1, lk1, lk2)
        c1 = m0 * f1 * K1 * lf / (1.0 + K1 * lf) * v0
        c2 = m0 * (1.0 - f1) * K2 * lf / (1.0 + K2 * lf) * v0
        d1 = np.empty_like(x)
        d2 = np.empty_like(x)
        d1[0], d2[0] = c1[0], c2[0]
        d1[1:], d2[1:] = np.diff(c1), np.diff(c2)
        inj = v * l0
        return np.vstack([d1 / inj, d2 / inj])

    def loss(p: list[float]) -> float:
        B = basis_for(p)
        coef = _linear_dh(y, B, 2)
        resid = y - coef @ B
        return float(np.sum(resid * resid))

    status = "fitted"
    diagnostics: list[dict[str, Any]] = []
    if n_pts < 7:
        status = "underidentified"
        diagnostics.append(Diagnostic(
            "two_site_underidentified",
            f"two-site needs >= 7 points ({n_pts} provided); reported as "
            "structural candidate without a unique fit",
            detail={"n_points": n_pts, "minimum": 7}).to_dict())
    f1_grid = np.linspace(0.25, 0.75, 3)
    lk_grid = np.linspace(2.5, 8.5, 7)
    best = None
    for f1 in f1_grid:
        for lk1 in lk_grid:
            for lk2 in lk_grid:
                if lk2 <= lk1:
                    continue
                budget.tick()
                val = loss([f1, lk1, lk2])
                if best is None or val < best[0]:
                    best = (val, [f1, lk1, lk2])
    if status == "fitted":
        p = _coordinate_refine(budget, best[1],
                               [(0.05, 0.95), (1.0, 10.0), (1.0, 10.0)],
                               loss, rounds=1)
    else:
        p = best[1]
    B = basis_for(p)
    dh1, dh2 = _linear_dh(y, B, 2)
    fitted = dh1 * B[0] + dh2 * B[1]
    sse = float(np.sum((y - fitted) ** 2))
    return {
        "model": "two_site", "k_params": 5, "converged": status == "fitted",
        "status": status, "diagnostics": diagnostics,
        "parameters": {
            "fraction_site1": {"value": float(p[0]), "unit": "fraction",
                               "raw_basis": "model"},
            "K1": {"value": float(10.0 ** p[1]), "unit": "M^-1",
                   "raw_basis": "model"},
            "log10_K1": {"value": float(p[1]), "unit": "log10(M^-1)",
                         "raw_basis": "model"},
            "K2": {"value": float(10.0 ** p[2]), "unit": "M^-1",
                   "raw_basis": "model"},
            "log10_K2": {"value": float(p[2]), "unit": "log10(M^-1)",
                         "raw_basis": "model"},
            "delta_H1": {"value": float(dh1), "unit": "kcal/mol",
                         "raw_basis": "model"},
            "delta_H2": {"value": float(dh2), "unit": "kcal/mol",
                         "raw_basis": "model"},
        },
        "sse": sse, "predicted": fitted.tolist(),
        "residuals": (y - fitted).tolist(),
    }


TIE_AICC = 2.0
TIE_REL_SSE = 0.05


def fit_models(adjusted: dict[str, Any], concentration_mat, cell_volume_mat,
               *, budget: int = 4000, models: tuple[str, ...] =
               ("one_site", "two_site", "dilution"),
               tie_aicc: float = TIE_AICC, tie_rel_sse: float = TIE_REL_SSE,
               parent_ids: list[str] | None = None,
               parameter_versions: dict[str, str] | None = None
               ) -> dict[str, Any]:
    ds, build_diags = build_dataset(adjusted, concentration_mat, cell_volume_mat)
    if ds is None:
        raise SemanticImpasse(
            "no dataset can be assembled for model fitting",
            [Diagnostic(d["code"], d["message"], d.get("file"),
                        d.get("row"), d.get("column"), d.get("t_s"),
                        d.get("detail", {})) for d in build_diags])
    conc = {e["key"]: e["quantity"] for e in concentration_mat.payload["entries"]}
    l0 = float(conc["syringe_concentration"]["canonical"])
    cell_conc = conc.get("cell_concentration")
    m0 = float(cell_conc["canonical"]) if cell_conc and cell_conc["convertible"] else 0.0
    if m0 <= 0 and "dilution" not in models:
        pass
    if m0 <= 0:
        raise SemanticImpasse(
            "binding fits require a positive cell (macromolecule) concentration",
            [Diagnostic("missing_cell_concentration",
                        "cell concentration missing/unconvertible",
                        concentration_mat.file)])
    v0 = next(float(e["quantity"]["canonical"])
              for e in cell_volume_mat.payload["entries"]
              if e["key"] == "cell_volume" and e["quantity"]["convertible"])
    ctx = {"m0": m0, "l0": l0, "v0": v0}
    bg = Budget(budget)
    fitters = {"one_site": _fit_one_site, "two_site": _fit_two_site,
               "dilution": _fit_dilution}
    candidates = [fitters[name](ds, ctx, bg) for name in models]
    n = len(ds.q)
    for c in candidates:
        c["aicc"] = _aicc(c["sse"], n, c["k_params"])
        c["n_points"] = n

    def rank_key(c: dict[str, Any]) -> tuple:
        # deterministic ordering: lower SSE first, then model name.
        return (round(c["sse"], 12), c["model"])

    candidates.sort(key=rank_key)
    best = candidates[0]
    tied: list[str] = []
    for c in candidates:
        daicc = c["aicc"] - best["aicc"]
        rel = (abs(c["sse"] - best["sse"]) / best["sse"]) if best["sse"] > 0 else 0.0
        c["delta_aicc"] = daicc
        c["relative_sse"] = rel
        if daicc <= tie_aicc and rel <= tie_rel_sse:
            tied.append(c["model"])
    ambiguous = len(tied) > 1
    for rank, c in enumerate(candidates):
        c["rank"] = rank
        c["selected"] = (c["model"] == best["model"])
        c["in_equivalent_set"] = c["model"] in tied

    inputs = {
        "rule_version": RULE_VERSION,
        "dilution_result": adjusted["id"],
        "concentration_sha1": concentration_mat.sha1,
        "cell_volume_sha1": cell_volume_mat.sha1,
        "l0_mM": l0, "m0_mM": m0, "v0_uL": v0,
        "budget": budget, "models": list(models),
        "tie_aicc": tie_aicc, "tie_rel_sse": tie_rel_sse,
        "dataset": ds.to_compact(),
        "parameter_versions": parameter_versions or {},
        "parents": parent_ids or [adjusted["id"]],
    }
    result_id = "res:bind:" + content_hash(inputs)[5:21]
    return {
        "id": result_id, "kind": "model_candidates", "rule_version": RULE_VERSION,
        "inputs": inputs,
        "convention": "fixed V0 stirred cell (Wiseman displaced-volume); "
                      "heights in kcal/mol injectant; K reported in M^-1",
        "selected_model": best["model"],
        "ambiguous": ambiguous,
        "equivalent_models": tied,
        "candidates": candidates,
        "budget_used": bg.used, "budget_limit": budget,
        "dataset_orders": ds.order.tolist(),
        "dataset_injection_ids": ds.injection_id,
        "diagnostics": build_diags,
    }
