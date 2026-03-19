"""
uncertainty_analysis.py
=======================

Sensitivity analysis of the robust traffic controller to the uncertainty
level δ (fractional perturbation of all initial densities).

Functions
---------
sweep_uncertainty   – sweep δ from 0 % to 50 % and record key metrics.
simulate_extreme_scenarios – apply a fixed robust controller to upper/lower
                             IC bound realisations (Figure 3 equivalent).
"""

from __future__ import annotations

import warnings
from typing import Sequence, Tuple, Dict

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from robust_traffic_control import (
    TrafficParams, GridParams, RobustTrafficController, SolverResult,
    paper_default_ic, paper_milp_ic,
)

# ── Default scenario — paper Section IV.C (fits within Gurobi licence limits) ─
# N=10, X=300m, ΔT=30s=1/120 h → alpha=1.667 (LP does not require CFL≤1)
# K=12 is max feasible (paper uses K=30 which needs 9149 constraints > 2000)
DEFAULT_TP  = TrafficParams(v_f=60.0, w=15.0, rho_max=150.0)
DEFAULT_GRD = GridParams(L=10 * 0.3, T=12 / 120.0, N=10, K=12)


# =============================================================================
# Main sweep
# =============================================================================

def sweep_uncertainty(
        rho_0: np.ndarray,
        delta_values: Sequence[float] = (0.00, 0.02, 0.05, 0.10,
                                          0.15, 0.20, 0.30, 0.40, 0.50),
        tp: TrafficParams  = DEFAULT_TP,
        grd: GridParams    = DEFAULT_GRD,
        verbose: bool      = True,
) -> Dict[str, np.ndarray]:
    """
    Sweep the uncertainty level δ ∈ [0, 0.5] and return summary statistics.

    For each δ the function:
      (a) Solves the nominal optimal LP once (δ = 0, reference).
      (b) Solves the robust MILP with uncertainty level δ.

    Parameters
    ----------
    rho_0        : array_like (N,)  Nominal initial density [veh/km].
    delta_values : sequence         Uncertainty levels to sweep.
    tp, grd      : model parameters (grid must satisfy Gurobi licence limits).
    verbose      : bool

    Returns
    -------
    dict with keys:
        'deltas'         – uncertainty levels (fraction)
        'opt_obj'        – negated LP objective (higher = better)
        'rob_obj'        – negated MILP objective
        'opt_n_cong'     – congested cells at t=T (optimal)
        'rob_n_cong'     – congested cells at t=T (robust)
        'rob_solve_time' – MILP solve time [s]
        'gap'            – (opt_obj − rob_obj) / opt_obj  (relative conservatism)
    """
    rho_0  = np.asarray(rho_0, dtype=float)
    ctrl   = RobustTrafficController(tp, grd)
    deltas = np.asarray(delta_values, dtype=float)
    n = len(deltas)

    opt_obj        = np.zeros(n)
    rob_obj        = np.zeros(n)
    opt_n_cong     = np.zeros(n, dtype=int)
    rob_n_cong     = np.zeros(n, dtype=int)
    rob_solve_time = np.zeros(n)

    # Reference: MILP with nominal IC (δ=0) — same objective as robust
    # (Using solve_optimal's LP would give an incomparable objective value.)
    r_nom = ctrl.solve_robust(rho_0, delta=0.0, verbose=False)
    opt_obj_ref   = -r_nom.obj_value
    opt_ncong_ref = int(np.sum(r_nom.b_cong)) if r_nom.b_cong is not None else 0

    for i, delta in enumerate(deltas):
        opt_obj[i]    = opt_obj_ref
        opt_n_cong[i] = opt_ncong_ref

        if verbose:
            print(f'  δ = {delta*100:5.1f}%  …', end='  ', flush=True)

        try:
            r_rob = ctrl.solve_robust(rho_0, delta=delta, verbose=False)
            rob_obj[i]        = -r_rob.obj_value
            rob_n_cong[i]     = int(np.sum(r_rob.b_cong))
            rob_solve_time[i] = r_rob.solve_time
            if verbose:
                print(f'OK  obj={rob_obj[i]:.2f}  '
                      f't={r_rob.solve_time:.3f}s')
        except Exception as exc:
            warnings.warn(f'Robust solve (δ={delta}) failed: {exc}')
            rob_obj[i] = rob_obj[i-1] if i > 0 else np.nan
            if verbose:
                print(f'FAILED: {exc}')

    gap = np.where(opt_obj > 0,
                   (opt_obj - rob_obj) / opt_obj,
                   np.zeros(n))

    return dict(deltas=deltas, opt_obj=opt_obj, rob_obj=rob_obj,
                opt_n_cong=opt_n_cong, rob_n_cong=rob_n_cong,
                rob_solve_time=rob_solve_time, gap=gap)


# =============================================================================
# Extreme-scenario simulation (Figure 3 equivalent)
# =============================================================================

def simulate_extreme_scenarios(
        rho_0: np.ndarray,
        delta: float       = 0.10,
        tp: TrafficParams  = DEFAULT_TP,
        grd: GridParams    = DEFAULT_GRD,
        verbose: bool      = True,
) -> Tuple[SolverResult, SolverResult, SolverResult]:
    """
    Verify robustness by applying the robust controller to two extreme ICs.

    Steps
    -----
    1. Solve the robust MILP with nominal IC → get q_in*, q_out*.
    2. Fix those boundary flows and forward-simulate with:
       • upper-bound IC: ρ₀·(1+δ)
       • lower-bound IC: ρ₀·(1−δ)

    Returns
    -------
    (rob_nominal, sim_upper, sim_lower) : three SolverResult objects.
    """
    ctrl = RobustTrafficController(tp, grd)

    if verbose:
        print(f'  Solving robust MILP (δ = {delta*100:.0f}%) …')
    rob_nominal = ctrl.solve_robust(rho_0, delta=delta, verbose=False)

    # Upper-bound IC: cap densities at ρ_max (physical limit)
    rho_upper = np.minimum(rho_0 * (1.0 + delta), tp.rho_max)
    if verbose:
        print('  Forward-simulating upper-bound IC …')
    sim_upper = _simulate_fixed_control(
        rho_upper, rob_nominal.q_in, rob_nominal.q_out, tp, grd)

    if verbose:
        print('  Forward-simulating lower-bound IC …')
    sim_lower = _simulate_fixed_control(
        np.maximum(0.0, rho_0 * (1.0 - delta)),
        rob_nominal.q_in, rob_nominal.q_out, tp, grd)

    return rob_nominal, sim_upper, sim_lower


# =============================================================================
# Fixed-control forward simulation
# =============================================================================

def _simulate_fixed_control(rho_0: np.ndarray,
                              q_in:  np.ndarray,
                              q_out: np.ndarray,
                              tp: TrafficParams,
                              grd: GridParams) -> SolverResult:
    """
    Forward-simulate the LWR model with **fixed** boundary flows.

    The boundary flows q_in and q_out are pinned (not decision variables).
    A feasibility LP finds the consistent Moskowitz function.

    Parameters
    ----------
    rho_0 : array_like (N,)   IC for this simulation.
    q_in  : array_like (K,)   Fixed upstream flow [veh/h].
    q_out : array_like (K,)   Fixed downstream flow [veh/h].

    Returns
    -------
    SolverResult
    """
    import time as _time

    rho_0 = np.asarray(rho_0, dtype=float)
    q_in  = np.asarray(q_in,  dtype=float)
    q_out = np.asarray(q_out, dtype=float)

    N, K   = grd.N, grd.K
    dx, dt = grd.dx, grd.dt
    q_max  = tp.q_max
    rho_c  = tp.rho_c
    rho_max= tp.rho_max
    v_f, w = tp.v_f, tp.w
    alpha, beta = grd.cfl_numbers(tp)

    # Moskowitz IC
    N0 = np.zeros(N + 1)
    for k in range(1, N + 1):
        N0[k] = -np.sum(rho_0[:k]) * dx

    env   = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model(name='fixed_sim', env=env)

    N_var = model.addVars(K + 1, N + 1, lb=-GRB.INFINITY, name='N')

    # Fix IC
    for k in range(N + 1):
        model.addConstr(N_var[0, k] == N0[k])

    # Fix boundary flows (cumulative count)
    for n in range(K + 1):
        model.addConstr(N_var[n, 0] == float(np.sum(q_in[:n]) * dt))
    for n in range(K + 1):
        model.addConstr(N_var[n, N] == float(N0[N] - np.sum(q_out[:n]) * dt))

    # Lax-Hopf constraints (IC all-pairs + one-step BC)
    for n in range(1, K + 1):
        for k in range(N + 1):
            j_lo = max(0, int(np.ceil(k - n * alpha - 1e-9)))
            j_hi = min(N, int(np.floor(k + n * beta  + 1e-9)))
            for j in range(j_lo, j_hi + 1):
                v = (k - j) * dx / (n * dt)
                if -w - 1e-9 <= v <= v_f + 1e-9:
                    rhs = N0[j] + n * dt * q_max - (k - j) * dx * rho_c
                    model.addConstr(N_var[n, k] <= rhs)
            for m in range(n):
                dtn = (n - m) * dt
                v = k * dx / dtn
                if -w - 1e-9 <= v <= v_f + 1e-9:
                    model.addConstr(N_var[n, k] <= N_var[m, 0]
                                    + dtn * q_max - k * dx * rho_c)
            for m in range(n):
                dtn = (n - m) * dt
                v = (k - N) * dx / dtn
                if -w - 1e-9 <= v <= v_f + 1e-9:
                    model.addConstr(N_var[n, k] <= N_var[m, N]
                                    + dtn * q_max - (k - N) * dx * rho_c)

    # Physical density bounds
    for n in range(K + 1):
        for i in range(N):
            model.addConstr(N_var[n, i] - N_var[n, i + 1] >= 0)
            model.addConstr(N_var[n, i] - N_var[n, i + 1] <= rho_max * dx)

    model.setObjective(
        -gp.quicksum(N_var[K, k] for k in range(N + 1)), GRB.MINIMIZE)

    t0 = _time.perf_counter()
    model.optimize()
    solve_time = _time.perf_counter() - t0

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f'Fixed simulation failed (status {model.Status})')

    N_sol = np.array([[N_var[n, k].X for k in range(N + 1)]
                       for n in range(K + 1)])
    rho_sol  = (N_sol[:, :-1] - N_sol[:, 1:]) / dx
    # Forward-difference flow for the first K time steps; row K is padded with 0
    flow_sol = np.zeros((K + 1, N + 1))
    flow_sol[:K] = (N_sol[1:] - N_sol[:K]) / dt

    return SolverResult(
        N_matz=N_sol, q_in=q_in, q_out=q_out,
        rho=rho_sol, flow_grid=flow_sol, b_cong=None,
        obj_value=model.ObjVal, solve_time=solve_time,
        status='Optimal (fixed control)')
