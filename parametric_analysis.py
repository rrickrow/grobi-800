"""
parametric_analysis.py
======================

Parametric studies for the robust/optimal traffic-control MILP.

Sweeps the following parameters (others held at safe defaults):
  1. Road length L  [km]             — N ∝ L, K = 8 (fixed)
  2. Time horizon T [h]              — N = 6 fixed, K ∝ T
  3. Spatial resolution N            — K = 10 (fixed)
  4. Initial density level           — uniform fraction of ρ_max
  5. Free-flow speed v_f [km/h]      — N = 8, K adjusted for CFL

All grids are chosen to respect the Gurobi restricted-licence limits
(≤ 2000 variables, ≤ 2000 linear constraints).
"""

from __future__ import annotations

import warnings
from typing import Sequence, Tuple, Dict

import numpy as np

from robust_traffic_control import (
    TrafficParams, GridParams, RobustTrafficController,
    paper_default_ic, count_constraints,
)

DELTA_ROB = 0.10    # 10 % uncertainty for the robust run


# =============================================================================
# Generic sweep helper
# =============================================================================

def _sweep(tp_list, grd_list, rho_list,
           delta: float = DELTA_ROB,
           verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Run one LP + one MILP solve per (tp, grd, rho_0) triple.

    Returns
    -------
    dict with keys:
        'opt_flows'  – negated LP objective   (higher = better)
        'rob_flows'  – negated MILP objective
        'opt_times'  – LP solve time [s]
        'rob_times'  – MILP solve time [s]
        'opt_n_cong' – congested cells at t=T (optimal, from density)
        'rob_n_cong' – congested cells at t=T (robust, from b_cong)
    """
    n = len(tp_list)
    opt_flows  = np.full(n, np.nan)
    rob_flows  = np.full(n, np.nan)
    opt_times  = np.full(n, np.nan)
    rob_times  = np.full(n, np.nan)
    opt_n_cong = np.zeros(n, dtype=int)
    rob_n_cong = np.zeros(n, dtype=int)

    for i, (tp, grd, rho_0) in enumerate(zip(tp_list, grd_list, rho_list)):
        if verbose:
            print(f'    Run {i+1}/{n}: N={grd.N}, K={grd.K}, '
                  f'T={grd.T*60:.1f} min …', end='  ', flush=True)

        try:
            ctrl = RobustTrafficController(tp, grd)
        except ValueError as e:
            if verbose:
                print(f'Skip (grid too large): {e}')
            continue

        try:
            r_opt = ctrl.solve_optimal(rho_0, verbose=False)
            opt_flows[i]  = -r_opt.obj_value
            opt_times[i]  = r_opt.solve_time
            opt_n_cong[i] = int(np.sum(r_opt.rho[-1] > tp.rho_c))
        except Exception as e:
            warnings.warn(f'Optimal solve {i} failed: {e}')

        try:
            r_rob = ctrl.solve_robust(rho_0, delta=delta, verbose=False)
            rob_flows[i]  = -r_rob.obj_value
            rob_times[i]  = r_rob.solve_time
            rob_n_cong[i] = int(np.sum(r_rob.b_cong))
        except Exception as e:
            warnings.warn(f'Robust solve {i} failed: {e}')

        if verbose:
            print(f'opt={opt_flows[i]:.2f}  rob={rob_flows[i]:.2f}  '
                  f't_lp={opt_times[i]:.3f}s  t_milp={rob_times[i]:.3f}s')

    return dict(opt_flows=opt_flows, rob_flows=rob_flows,
                opt_times=opt_times, rob_times=rob_times,
                opt_n_cong=opt_n_cong, rob_n_cong=rob_n_cong)


# =============================================================================
# 1. Road length L
# =============================================================================

def sweep_road_length(
        L_values: Sequence[float] = (5, 7, 10, 12, 15),
        K: int = 8,
        delta: float = DELTA_ROB,
        verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Vary road length L [km] with one cell per km (N = round(L)) and K = 8.

    CFL condition is satisfied by setting dt = dx / v_f (α = 1).
    T = K · dt adjusts automatically.
    """
    print('Parametric study: Road length L')
    tp_list, grd_list, rho_list = [], [], []
    L_arr = np.asarray(L_values, dtype=float)

    for L in L_arr:
        tp  = TrafficParams()          # default FD
        N   = max(4, round(float(L)))
        dx  = L / N
        dt  = dx / tp.v_f             # CFL α = 1
        T   = K * dt
        grd = GridParams(L=L, T=T, N=N, K=K)
        tp_list.append(tp)
        grd_list.append(grd)
        rho_list.append(paper_default_ic(tp, grd))

    return L_arr, _sweep(tp_list, grd_list, rho_list,
                          delta=delta, verbose=verbose)


# =============================================================================
# 2. Time horizon T
# =============================================================================

def sweep_time_horizon(
        K_values: Sequence[int] = (4, 6, 8, 10, 12, 15),
        N: int = 6,
        delta: float = DELTA_ROB,
        verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Vary the simulation time horizon by sweeping K (number of time steps)
    with N = 6 fixed.  T = K · dt where dt = dx / v_f (CFL = 1).
    """
    print('Parametric study: Time horizon T')
    tp_list, grd_list, rho_list = [], [], []
    tp_ref = TrafficParams()
    dx     = 10.0 / N
    dt     = dx / tp_ref.v_f          # CFL = 1
    T_arr  = np.array(K_values, dtype=float) * dt

    for K in K_values:
        tp  = TrafficParams()
        T   = K * dt
        grd = GridParams(L=10.0, T=T, N=N, K=K)
        tp_list.append(tp)
        grd_list.append(grd)
        rho_list.append(paper_default_ic(tp, grd))

    return T_arr * 60, _sweep(tp_list, grd_list, rho_list,
                               delta=delta, verbose=verbose)


# =============================================================================
# 3. Spatial resolution N
# =============================================================================

def sweep_spatial_resolution(
        N_values: Sequence[int] = (4, 6, 8, 10),
        K: int = 10,
        delta: float = DELTA_ROB,
        verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Vary the number of spatial cells N.  K is fixed at 10.
    dt is set to satisfy CFL = 1 for the given N, and T = K · dt.
    """
    print('Parametric study: Spatial resolution N')
    tp_list, grd_list, rho_list = [], [], []
    N_arr = np.asarray(N_values, dtype=int)

    for N in N_arr:
        tp  = TrafficParams()
        dx  = 10.0 / int(N)
        dt  = dx / tp.v_f
        T   = K * dt
        grd = GridParams(L=10.0, T=T, N=int(N), K=K)
        tp_list.append(tp)
        grd_list.append(grd)
        rho_list.append(paper_default_ic(tp, grd))

    return N_arr.astype(float), _sweep(tp_list, grd_list, rho_list,
                                        delta=delta, verbose=verbose)


# =============================================================================
# 4. Initial density level (uniform)
# =============================================================================

def sweep_initial_density(
        rho_fracs: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
        delta: float = DELTA_ROB,
        verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Vary the initial density as a fraction of ρ_max (uniform profile).
    """
    print('Parametric study: Initial density level')
    tp_ref  = TrafficParams()
    grd_ref = GridParams(L=10.0, T=0.2, N=10, K=12)  # default safe grid
    tp_list, grd_list, rho_list = [], [], []

    for frac in rho_fracs:
        tp_list.append(TrafficParams())
        grd_list.append(grd_ref)
        rho_list.append(np.full(grd_ref.N, frac * tp_ref.rho_max))

    return np.asarray(rho_fracs) * 100, _sweep(
        tp_list, grd_list, rho_list, delta=delta, verbose=verbose)


# =============================================================================
# 5. Free-flow speed v_f
# =============================================================================

def sweep_vf(
        vf_values: Sequence[float] = (40, 50, 60, 70, 80),
        N: int = 8,
        delta: float = DELTA_ROB,
        verbose: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Vary the free-flow speed v_f [km/h] with N = 8 spatial cells.
    K is computed for CFL = 1 (dt = dx / v_f) and T = 0.2 h.
    """
    print('Parametric study: Free-flow speed v_f')
    tp_list, grd_list, rho_list = [], [], []
    vf_arr = np.asarray(vf_values, dtype=float)

    for vf in vf_arr:
        tp  = TrafficParams(v_f=vf, w=15.0, rho_max=150.0)
        dx  = 10.0 / N
        dt  = dx / vf              # CFL α = 1
        T   = 0.2                  # fixed time horizon [h]
        K   = int(np.ceil(T / dt))
        grd = GridParams(L=10.0, T=T, N=N, K=K)
        tp_list.append(tp)
        grd_list.append(grd)
        rho_list.append(paper_default_ic(tp, grd))

    return vf_arr, _sweep(tp_list, grd_list, rho_list,
                           delta=delta, verbose=verbose)


# =============================================================================
# Run all parametric studies
# =============================================================================

def run_all_parametric(delta: float = DELTA_ROB,
                        verbose: bool = True) -> dict:
    """
    Execute all five built-in parametric sweeps.

    Returns
    -------
    dict keyed by study name, each value is
    (param_values, summary_dict, x_label, study_name).
    """
    studies = {}

    print('=' * 60)
    pv, res = sweep_road_length(delta=delta, verbose=verbose)
    studies['road_length'] = (
        pv, res, 'Road length $L$ [km]', 'Road length')

    print('=' * 60)
    pv, res = sweep_time_horizon(delta=delta, verbose=verbose)
    studies['time_horizon'] = (
        pv, res, 'Time horizon $T$ [min]', 'Time horizon')

    print('=' * 60)
    pv, res = sweep_spatial_resolution(delta=delta, verbose=verbose)
    studies['spatial_res'] = (
        pv, res, 'Spatial cells $N$', 'Spatial resolution')

    print('=' * 60)
    pv, res = sweep_initial_density(delta=delta, verbose=verbose)
    studies['init_density'] = (
        pv, res, r'Initial density [% of $\rho_{max}$]', 'Initial density')

    print('=' * 60)
    pv, res = sweep_vf(delta=delta, verbose=verbose)
    studies['vf'] = (
        pv, res, 'Free-flow speed $v_f$ [km/h]', 'Free-flow speed')

    return studies
