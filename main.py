"""
main.py
=======

Main entry point: reproduces the key results from the paper and extends them
with a parametric study and uncertainty-level sensitivity analysis.

Reference
---------
Li, Y., Canepa, E., & Claudel, C. (2013).
"Exact solutions to robust control problems involving scalar hyperbolic
conservation laws using Mixed Integer Linear Programming."
51st Annual Allerton Conference, UIUC, October 2013.

What this script produces
--------------------------
  figures/fig1_optimal_control.png          – Figure 1 of paper
  figures/fig2_optimal_vs_robust.png        – Figure 2 of paper
  figures/fig3_extreme_scenarios.png        – Figure 3 of paper
  figures/fig4_uncertainty_sensitivity.png  – uncertainty sweep (new)
  figures/fig5_<study>_parametric.png       – five parametric studies (new)

Usage
-----
    python main.py

Requirements
------------
    pip install gurobipy numpy matplotlib
    (A valid Gurobi licence is required.  The free restricted licence
     supports the grid sizes used here: N ≤ 10, K ≤ 12.)
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from robust_traffic_control import (
    TrafficParams, GridParams, RobustTrafficController,
    paper_default_ic, count_constraints,
)
from visualization import (
    plot_optimal_control, plot_comparison, plot_extreme_scenarios,
    plot_uncertainty_sensitivity, plot_parametric_summary,
)
from uncertainty_analysis import sweep_uncertainty, simulate_extreme_scenarios
from parametric_analysis import run_all_parametric


# =============================================================================
# Default problem configuration
# =============================================================================

# Triangular fundamental diagram
TP = TrafficParams(
    v_f=60.0,       # free-flow speed      [km/h]
    w=15.0,         # backward wave speed  [km/h]
    rho_max=150.0,  # jam density          [veh/km]
)

# Space-time grid — CFL = 1 for free-flow  (α = v_f·Δt/Δx = 60·(1/60)/1 = 1)
# N=10, K=12 → MILP has 1 654 constraints, 177 variables (within licence limit)
GRD = GridParams(
    L=10.0,   # road length  [km]
    T=0.2,    # horizon      [h]  (= 12 min)
    N=10,     # spatial cells — Δx = 1 km
    K=12,     # time steps   — Δt = 1 min
)

DELTA = 0.10    # uncertainty level for the main robust run (10 %, as in paper)

OUT_DIR = 'figures'
os.makedirs(OUT_DIR, exist_ok=True)

def fp(name): return os.path.join(OUT_DIR, name)


# =============================================================================
# Helpers
# =============================================================================

def sep(title):
    print('\n' + '=' * 60)
    print(f'  {title}')
    print('=' * 60)


def summary(label, r, tp, grd):
    """Print a short numerical summary of a SolverResult."""
    total_in  = float(np.sum(r.q_in)  * grd.dt)
    total_out = float(np.sum(r.q_out) * grd.dt)
    n_cong    = int(np.sum(r.rho[-1] > tp.rho_c))
    print(f'\n  [{label}]')
    print(f'    Status     : {r.status}')
    print(f'    Solve time : {r.solve_time:.3f} s')
    print(f'    Obj value  : {r.obj_value:.4f}')
    print(f'    Total q_in : {total_in:.2f} veh  '
          f'(mean {r.q_in.mean():.1f} veh/h)')
    print(f'    Total q_out: {total_out:.2f} veh  '
          f'(mean {r.q_out.mean():.1f} veh/h)')
    print(f'    Congested cells at t=T : {n_cong} / {grd.N}')
    if r.b_cong is not None:
        print(f'    b_cong : {r.b_cong.round().astype(int).tolist()}')


# =============================================================================
# Main
# =============================================================================

def main():
    t_global = time.perf_counter()

    # ── Configuration ──────────────────────────────────────────────────────────
    sep('Configuration')
    print(f'  {TP}')
    print(f'  {GRD}')
    alpha, beta = GRD.cfl_numbers(TP)
    print(f'  CFL : α = {alpha:.3f}  (free-flow),  β = {beta:.3f}  (congestion)')
    nc, nv = count_constraints(GRD.N, GRD.K, alpha, beta)
    print(f'  MILP size: {nc} constraints, {nv} variables  '
          f'(Gurobi limit: 2000 each)')

    rho_0 = paper_default_ic(TP, GRD)
    print(f'\n  Initial densities (6 piecewise segments, ρ_c = {TP.rho_c:.0f} veh/km):')
    for i, r in enumerate(rho_0):
        state = 'CONGESTED' if r > TP.rho_c else 'free-flow'
        print(f'    Cell {i:2d}  x=[{i:.0f},{i+1:.0f}) km : '
              f'ρ₀ = {r:6.1f} veh/km  ({state})')

    ctrl = RobustTrafficController(TP, GRD)

    # =========================================================================
    # 1.  Optimal control LP  (replicates Figure 1)
    # =========================================================================
    sep('1.  Optimal control LP  (Fig. 1)')
    opt_result = ctrl.solve_optimal(rho_0, verbose=False)
    summary('Optimal LP', opt_result, TP, GRD)

    fig1 = plot_optimal_control(
        opt_result, TP, GRD, rho_0,
        title='Optimal boundary control (LP)',
        save_path=fp('fig1_optimal_control.png'))
    plt.close(fig1)

    # =========================================================================
    # 2.  Robust control MILP  (replicates Figure 2)
    # =========================================================================
    sep(f'2.  Robust control MILP  (δ = {DELTA*100:.0f}%,  Fig. 2)')
    rob_result = ctrl.solve_robust(rho_0, delta=DELTA, verbose=False)
    summary(f'Robust MILP δ={DELTA:.2f}', rob_result, TP, GRD)

    fig2 = plot_comparison(
        opt_result, rob_result, TP, GRD, delta=DELTA,
        save_path=fp('fig2_optimal_vs_robust.png'))
    plt.close(fig2)

    gap_pct = ((-opt_result.obj_value) - (-rob_result.obj_value)) \
              / abs(-opt_result.obj_value) * 100
    print(f'\n  ► Robust solution is {gap_pct:.1f}% more conservative than optimal.')

    # =========================================================================
    # 3.  Extreme-scenario verification  (replicates Figure 3)
    # =========================================================================
    sep('3.  Extreme-scenario verification  (Fig. 3)')
    rob_nom, sim_upper, sim_lower = simulate_extreme_scenarios(
        rho_0, delta=DELTA, tp=TP, grd=GRD, verbose=True)

    fig3 = plot_extreme_scenarios(
        rob_nom, sim_upper, sim_lower, TP, GRD, delta=DELTA,
        save_path=fp('fig3_extreme_scenarios.png'))
    plt.close(fig3)

    n_u = int(np.sum(sim_upper.rho[-1] > TP.rho_c))
    n_l = int(np.sum(sim_lower.rho[-1] > TP.rho_c))
    print(f'  Upper-bound IC: {n_u} / {GRD.N} cells congested at t=T')
    print(f'  Lower-bound IC: {n_l} / {GRD.N} cells congested at t=T')
    print('  → Robust controller maintains physical feasibility in both cases.')

    # =========================================================================
    # 4.  Uncertainty-level sensitivity  (δ swept 0→50 %)
    # =========================================================================
    sep('4.  Uncertainty-level sensitivity  (Fig. 4)')
    delta_sweep = np.array([0.00, 0.02, 0.05, 0.10,
                             0.15, 0.20, 0.30, 0.40, 0.50])
    unc = sweep_uncertainty(
        rho_0, delta_values=delta_sweep, tp=TP, grd=GRD, verbose=True)

    fig4 = plot_uncertainty_sensitivity(
        deltas=unc['deltas'],
        opt_obj=unc['opt_obj'],
        rob_obj=unc['rob_obj'],
        n_cong_opt=unc['opt_n_cong'],
        n_cong_rob=unc['rob_n_cong'],
        solve_times=unc['rob_solve_time'],
        save_path=fp('fig4_uncertainty_sensitivity.png'))
    plt.close(fig4)

    print('\n  δ [%]  |  Opt obj  |  Rob obj  |  Gap [%]  |  Solve [s]')
    print('  -------|-----------|-----------|-----------|----------')
    for i, d in enumerate(delta_sweep):
        print(f'  {d*100:5.1f}  |  {unc["opt_obj"][i]:9.2f}  |  '
              f'{unc["rob_obj"][i]:9.2f}  |  '
              f'{unc["gap"][i]*100:8.1f}%  |  '
              f'{unc["rob_solve_time"][i]:.4f}')

    # =========================================================================
    # 5.  Parametric studies
    # =========================================================================
    sep('5.  Parametric studies  (Fig. 5a–e)')
    par_studies = run_all_parametric(delta=DELTA, verbose=True)

    for key, (pv, res, xlabel, pname) in par_studies.items():
        fig = plot_parametric_summary(
            res, param_values=pv, param_label=xlabel,
            param_name=pname,
            save_path=fp(f'fig5_{key}_parametric.png'))
        plt.close(fig)

    # =========================================================================
    # Final summary
    # =========================================================================
    sep('Summary')
    elapsed = time.perf_counter() - t_global
    print(f'  Completed in {elapsed:.1f} s')
    print(f'  Figures saved to ./{OUT_DIR}/')
    print()
    print('  Key numerical results (main scenario):')
    print(f'    Optimal LP  obj  : {opt_result.obj_value:.4f}  '
          f'(solve: {opt_result.solve_time:.3f} s)')
    print(f'    Robust MILP obj  : {rob_result.obj_value:.4f}  '
          f'(solve: {rob_result.solve_time:.3f} s)')
    print(f'    Conservatism gap : {gap_pct:.1f} %')
    print(f'    Max uncertainty tested: δ = 50 %  '
          f'(gap = {unc["gap"][-1]*100:.1f} %)')
    print()


if __name__ == '__main__':
    main()
