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

Paper examples
--------------
Section III.C — Optimal control LP (Figure 1):
  • N = 6 segments, X = 643 m → L = 3.858 km
  • K = 15 time steps, ΔT = 30 s → T = 450 s = 7.5 min
  • Objective: min −Σ q_out(t)  (eq. 6 — maximise downstream flow only)
  • 36 decision variables, ~600 constraints in the paper's compact formulation

Section IV.C — Robust MILP (Figures 2–3):
  • N = 10 segments, X = 300 m → L = 3 km
  • ρ_meas = [3,5,2,8,6,9,10,7,1,4] × 0.5·ρ_c,  uncertainty δ = 10 %
  • K = 30 in the paper (not feasible with Gurobi restricted licence);
    this code uses K = 12 (maximum within the 2000-constraint limit)
  • Objective: min −Σ w(t)·(q_out+q_in) + Σ b_cong(i)

What this script produces
--------------------------
  figures/fig1_optimal_control.png          – Figure 1 of paper (LP)
  figures/fig2_nominal_vs_robust.png        – Figure 2 of paper (MILP, δ=0 vs δ=10%)
  figures/fig3_extreme_scenarios.png        – Figure 3 of paper
  figures/fig4_uncertainty_sensitivity.png  – uncertainty sweep (new)
  figures/fig5_<study>_parametric.png       – five parametric studies (new)

Usage
-----
    python main.py

Requirements
------------
    pip install gurobipy numpy matplotlib
"""

import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from robust_traffic_control import (
    TrafficParams, GridParams, RobustTrafficController,
    paper_lp_ic, paper_milp_ic, paper_default_ic, count_constraints,
)
from visualization import (
    plot_optimal_control, plot_comparison, plot_extreme_scenarios,
    plot_uncertainty_sensitivity, plot_parametric_summary,
)
from uncertainty_analysis import sweep_uncertainty, simulate_extreme_scenarios
from parametric_analysis import run_all_parametric


# =============================================================================
# Paper grid configurations
# =============================================================================

# Triangular fundamental diagram (parameters not explicitly stated in paper)
TP = TrafficParams(
    v_f=60.0,       # free-flow speed      [km/h]
    w=15.0,         # backward wave speed  [km/h]
    rho_max=150.0,  # jam density          [veh/km]
)

# ── Section III.C — LP example ──────────────────────────────────────────────
# N=6, X=643m, K=15, ΔT=30s=1/120 h  →  T=450s=7.5 min  (paper: <7 min)
DT = 1.0 / 120.0       # 30 s = 1/120 h (paper's time granularity)
GRD_LP = GridParams(
    L=6 * 0.643,         # 6 segments × 643 m = 3.858 km
    T=15 * DT,           # 15 × 30 s = 450 s = 7.5 min
    N=6,
    K=15,
)

# ── Section IV.C — Robust MILP example ──────────────────────────────────────
# N=10, X=300m, K=30 in paper; limited to K=12 by Gurobi restricted licence
# alpha = v_f·ΔT/X = 60·(1/120)/0.3 = 1.667  (LP valid even with α > 1)
GRD_MILP = GridParams(
    L=10 * 0.3,          # 10 segments × 300 m = 3 km
    T=12 * DT,           # K=12 × 30 s = 360 s = 6 min (paper uses K=30=15 min)
    N=10,
    K=12,
)

DELTA = 0.10    # uncertainty level δ = 10 % (as in paper Section IV.C)

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
    print(f'    Total q_in : {total_in:.2f} veh  (mean {r.q_in.mean():.1f} veh/h)')
    print(f'    Total q_out: {total_out:.2f} veh  (mean {r.q_out.mean():.1f} veh/h)')
    print(f'    Congested cells at t=T : {n_cong} / {grd.N}')
    if r.b_cong is not None:
        print(f'    b_cong : {r.b_cong.round().astype(int).tolist()}')


# =============================================================================
# Main
# =============================================================================

def main():
    t_global = time.perf_counter()

    # ── FD parameters ──────────────────────────────────────────────────────────
    sep('Fundamental diagram & grid parameters')
    print(f'  {TP}')
    print(f'  ρ_c = {TP.rho_c:.2f} veh/km,  q_max = {TP.q_max:.1f} veh/h')
    print()
    for label, grd in [('LP   (Sec. III.C)', GRD_LP), ('MILP (Sec. IV.C)', GRD_MILP)]:
        alpha, beta = grd.cfl_numbers(TP)
        nc, nv = count_constraints(grd.N, grd.K, alpha, beta)
        print(f'  {label}: {grd}')
        print(f'    α={alpha:.3f} (free-flow CFL),  β={beta:.3f} (congestion CFL)')
        print(f'    Model size: {nc} constraints, {nv} vars '
              f'(Gurobi limit 2000 each)  {"✓" if nc<=1950 else "✗"}')
        print()

    # =========================================================================
    # 1.  Optimal control LP  — paper Figure 1
    # =========================================================================
    sep('1.  Optimal control LP  (paper Fig. 1 — Section III.C)')
    rho_lp = paper_lp_ic(TP, GRD_LP)
    print(f'  IC (N={GRD_LP.N}, X={GRD_LP.dx*1000:.0f} m per cell, '
          f'range [{rho_lp.min():.1f}, {rho_lp.max():.1f}] veh/km):')
    for i, r in enumerate(rho_lp):
        state = 'CONGESTED' if r > TP.rho_c else 'free-flow'
        print(f'    Seg {i}: ρ₀={r:5.1f} veh/km = {r/TP.rho_c:.2f}·ρ_c  ({state})')

    ctrl_lp = RobustTrafficController(TP, GRD_LP)
    opt_result = ctrl_lp.solve_optimal(rho_lp, verbose=False)
    summary('Optimal LP', opt_result, TP, GRD_LP)
    print(f'\n  Downstream flows at capacity? '
          f'{np.allclose(opt_result.q_out, TP.q_max, atol=1)}  '
          f'(paper Fig.1: "downstream flows are all maximal")')

    fig1 = plot_optimal_control(
        opt_result, TP, GRD_LP, rho_lp,
        title=f'Optimal boundary control LP — paper Fig. 1\n'
              f'N={GRD_LP.N}, X={GRD_LP.dx*1000:.0f}m, '
              f'K={GRD_LP.K}, ΔT=30s, T={GRD_LP.T*3600:.0f}s',
        save_path=fp('fig1_optimal_control.png'))
    plt.close(fig1)

    # =========================================================================
    # 2.  Robust MILP — paper Figures 2 & 3
    # =========================================================================
    sep(f'2.  MILP: nominal (δ=0) vs robust (δ={DELTA*100:.0f}%)  (paper Fig. 2 — Sec. IV.C)')
    rho_milp = paper_milp_ic(TP, GRD_MILP)
    print(f'\n  Paper IC: ρ_meas = [3,5,2,8,6,9,10,7,1,4] × 0.5·ρ_c')
    print(f'  (N={GRD_MILP.N}, X={GRD_MILP.dx*1000:.0f} m per cell, '
          f'ρ_c = {TP.rho_c:.1f} veh/km):')
    for i, r in enumerate(rho_milp):
        state = 'CONGESTED' if r > TP.rho_c else 'free-flow'
        print(f'    Seg {i}: ρ₀={r:6.1f} veh/km = {r/TP.rho_c:.1f}·ρ_c  ({state})')

    ctrl_milp = RobustTrafficController(TP, GRD_MILP)

    # Nominal (δ=0): MILP formulation with no uncertainty  → "optimal" in Fig. 2
    nom_result = ctrl_milp.solve_robust(rho_milp, delta=0.0, verbose=False)
    summary('Nominal MILP (δ=0)', nom_result, TP, GRD_MILP)

    # Robust (δ=10%): MILP with worst-case IC in constraints → "robust" in Fig. 2
    rob_result = ctrl_milp.solve_robust(rho_milp, delta=DELTA, verbose=False)
    summary(f'Robust MILP (δ={DELTA:.0%})', rob_result, TP, GRD_MILP)

    # Conservatism gap (same MILP objective, compare -obj values directly)
    gap_pct = ((-nom_result.obj_value) - (-rob_result.obj_value)) \
              / abs(-nom_result.obj_value) * 100
    print(f'\n  ► Robust solution is {gap_pct:.1f}% more conservative than nominal.')
    print(f'    (Paper: "robust control outputs admit less upstream flows")')

    fig2 = plot_comparison(
        nom_result, rob_result, TP, GRD_MILP, delta=DELTA,
        save_path=fp('fig2_nominal_vs_robust.png'))
    plt.close(fig2)

    # ── Figure 3: extreme scenarios ───────────────────────────────────────────
    sep('3.  Extreme-scenario verification  (paper Fig. 3)')
    rob_nom, sim_upper, sim_lower = simulate_extreme_scenarios(
        rho_milp, delta=DELTA, tp=TP, grd=GRD_MILP, verbose=True)

    fig3 = plot_extreme_scenarios(
        rob_nom, sim_upper, sim_lower, TP, GRD_MILP, delta=DELTA,
        save_path=fp('fig3_extreme_scenarios.png'))
    plt.close(fig3)

    n_u = int(np.sum(sim_upper.rho[-1] > TP.rho_c))
    n_l = int(np.sum(sim_lower.rho[-1] > TP.rho_c))
    print(f'  Upper IC (ρ_nom × (1+{DELTA:.0%})): {n_u}/{GRD_MILP.N} cells congested at t=T')
    print(f'  Lower IC (ρ_nom × (1−{DELTA:.0%})): {n_l}/{GRD_MILP.N} cells congested at t=T')
    print(f'  Paper: "robust control can still maintain a reasonable performance"')

    # =========================================================================
    # 4.  Uncertainty-level sensitivity  (δ swept 0→50 %)
    # =========================================================================
    sep('4.  Uncertainty-level sensitivity  (Fig. 4)')
    delta_sweep = np.array([0.00, 0.02, 0.05, 0.10,
                             0.15, 0.20, 0.30, 0.40, 0.50])
    unc = sweep_uncertainty(
        rho_milp, delta_values=delta_sweep, tp=TP, grd=GRD_MILP, verbose=True)

    fig4 = plot_uncertainty_sensitivity(
        deltas=unc['deltas'],
        opt_obj=unc['opt_obj'],
        rob_obj=unc['rob_obj'],
        n_cong_opt=unc['opt_n_cong'],
        n_cong_rob=unc['rob_n_cong'],
        solve_times=unc['rob_solve_time'],
        save_path=fp('fig4_uncertainty_sensitivity.png'))
    plt.close(fig4)

    print('\n  δ [%]  |  Nominal obj |  Robust obj  |  Gap [%]  |  Solve [s]')
    print('  -------|--------------|--------------|-----------|----------')
    for i, d in enumerate(delta_sweep):
        print(f'  {d*100:5.1f}  |  {unc["opt_obj"][i]:12.2f}  |  '
              f'{unc["rob_obj"][i]:12.2f}  |  '
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
    # Summary
    # =========================================================================
    sep('Summary')
    elapsed = time.perf_counter() - t_global
    print(f'  Completed in {elapsed:.1f} s')
    print(f'  Figures saved to ./{OUT_DIR}/')
    print()
    print('  Key results (matching paper examples):')
    print(f'    LP obj value (−Σ q_out)          : {opt_result.obj_value:.4f}  '
          f'solve: {opt_result.solve_time:.3f} s')
    print(f'    Nominal MILP obj (δ=0)            : {nom_result.obj_value:.4f}  '
          f'solve: {nom_result.solve_time:.3f} s')
    print(f'    Robust  MILP obj (δ={DELTA:.0%})       : {rob_result.obj_value:.4f}  '
          f'solve: {rob_result.solve_time:.3f} s')
    print(f'    Conservatism gap (nominal vs rob)  : {gap_pct:.1f} %')
    print(f'    Max uncertainty tested: δ = 50 %   '
          f'gap = {unc["gap"][-1]*100:.1f} %')
    print()


if __name__ == '__main__':
    main()
