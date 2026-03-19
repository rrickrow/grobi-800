"""
visualization.py
================

Plotting utilities that replicate and extend Figures 1–4 from the paper:

  Fig 1 – Space-time density heat-map + initial condition + boundary flows.
  Fig 2 – Side-by-side: robust vs optimal density + boundary flows.
  Fig 3 – Robust controller applied to extreme IC realisations.
  Fig 4 – Uncertainty-level sensitivity: flow, congestion, solve time.
  Fig 5 – Parametric study summary.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from robust_traffic_control import GridParams, TrafficParams, SolverResult

TRAFFIC_CMAP = 'RdYlGn_r'   # high density = red, low density = green


# =============================================================================
# Low-level helpers
# =============================================================================

def _density_imshow(ax, rho, grd, tp, vmin=0.0, vmax=None, cmap=TRAFFIC_CMAP):
    """Draw a density heat-map on *ax* (rows=time, cols=space)."""
    vmax = vmax or tp.rho_max
    # extent: [x_min, x_max, t_min, t_max]  (x in km, t in minutes)
    ext = [0, grd.L, 0, grd.T * 60]
    im = ax.imshow(
        rho, origin='lower', extent=ext, aspect='auto',
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xlabel('Position $x$ [km]', fontsize=9)
    ax.set_ylabel('Time $t$ [min]', fontsize=9)
    # overlay ρ_c contour
    t_arr = np.linspace(0, grd.T * 60, rho.shape[0])
    x_arr = grd.x_c
    try:
        ax.contour(x_arr, t_arr, rho, levels=[tp.rho_c],
                   colors='white', linewidths=0.9, linestyles='--', alpha=0.75)
    except Exception:
        pass
    return im


def _flow_step(ax, q, t_edges_min, q_max, label, color, ls='-'):
    """Draw a step-style flow time-series."""
    ax.step(t_edges_min, q, where='post', color=color, lw=1.8,
            ls=ls, label=label)
    ax.fill_between(t_edges_min, q, step='post', color=color, alpha=0.18)
    ax.axhline(q_max, color='grey', lw=0.8, ls=':', label='$q_{max}$')
    ax.set_xlim(0, t_edges_min[-1])
    ax.set_ylim(0, q_max * 1.08)
    ax.set_xlabel('Time [min]', fontsize=9)
    ax.set_ylabel('Flow [veh/h]', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')


# =============================================================================
# Figure 1 – Single optimal-control result
# =============================================================================

def plot_optimal_control(result: SolverResult,
                         tp: TrafficParams,
                         grd: GridParams,
                         rho_0: np.ndarray,
                         title: str = 'Optimal boundary control',
                         save_path: Optional[str] = None):
    """
    Replicate Figure 1 of the paper.

    Panel layout
    ------------
    (a) Space-time density heat-map with ρ_c contour (white dashed).
    (b) Initial density bar chart (red = congested, blue = free-flow).
    (c) Upstream boundary flow q_in(t).
    (d) Downstream boundary flow q_out(t).
    """
    fig = plt.figure(figsize=(12, 8))
    gs  = GridSpec(2, 2, fig, hspace=0.42, wspace=0.36)
    t_min = grd.t_edges[:grd.K] * 60   # [min]

    # (a) density heat-map
    ax = fig.add_subplot(gs[0, 0])
    im = _density_imshow(ax, result.rho, grd, tp)
    ax.set_title('(a) Density $\\rho(t,x)$ [veh/km]', fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('[veh/km]', fontsize=8)

    # (b) initial condition
    ax = fig.add_subplot(gs[0, 1])
    colors_ic = ['#d62728' if r > tp.rho_c else '#1f77b4' for r in rho_0]
    ax.bar(grd.x_c, rho_0, width=grd.dx * 0.85,
           color=colors_ic, edgecolor='k', linewidth=0.5, alpha=0.85)
    ax.axhline(tp.rho_c,   color='green', ls='--', lw=1.2,
               label=f'$\\rho_c={tp.rho_c:.0f}$')
    ax.axhline(tp.rho_max, color='k',     ls=':',  lw=1.0,
               label=f'$\\rho_{{max}}={tp.rho_max:.0f}$')
    ax.set_xlabel('Position $x$ [km]', fontsize=9)
    ax.set_ylabel('Density [veh/km]', fontsize=9)
    ax.set_title('(b) Initial condition $\\rho_0(x)$', fontsize=10)
    ax.set_ylim(0, tp.rho_max * 1.1)
    ax.legend(fontsize=8)

    # (c) upstream flow
    ax = fig.add_subplot(gs[1, 0])
    _flow_step(ax, result.q_in, t_min, tp.q_max,
               label='$q_{in}(t)$', color='#1f77b4')
    ax.set_title('(c) Upstream flow $q_{in}$ (control input)', fontsize=10)

    # (d) downstream flow
    ax = fig.add_subplot(gs[1, 1])
    _flow_step(ax, result.q_out, t_min, tp.q_max,
               label='$q_{out}(t)$', color='#2ca02c')
    ax.set_title('(d) Downstream flow $q_{out}$ (control output)', fontsize=10)

    fig.suptitle(
        f'{title}  |  Obj = {result.obj_value:.2f}  |  '
        f'Status: {result.status}  |  Solve: {result.solve_time:.3f} s',
        fontsize=11, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig


# =============================================================================
# Figure 2 – Optimal vs Robust comparison
# =============================================================================

def plot_comparison(opt: SolverResult,
                    rob: SolverResult,
                    tp: TrafficParams,
                    grd: GridParams,
                    delta: float = 0.10,
                    save_path: Optional[str] = None):
    """
    Replicate Figure 2 of the paper.

    Left column: robust control.   Right column: optimal control.
    Top row: density heat-maps.    Bottom row: boundary-flow profiles.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.subplots_adjust(hspace=0.42, wspace=0.36)
    t_min = grd.t_edges[:grd.K] * 60

    pairs = [(rob, f'Robust (δ={delta*100:.0f}%)', '#d62728'),
             (opt, 'Optimal',                      '#1f77b4')]

    for col, (res, ttl, clr) in enumerate(pairs):
        # density
        ax = axes[0, col]
        im = _density_imshow(ax, res.rho, grd, tp)
        ax.set_title(f'{ttl}\nDensity $\\rho(t,x)$', fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('[veh/km]',
                                                                      fontsize=8)
        # boundary flows
        ax = axes[1, col]
        ax.step(t_min, res.q_in,  where='post', color=clr,     lw=1.8,
                label='$q_{in}$')
        ax.step(t_min, res.q_out, where='post', color='green',  lw=1.8,
                ls='--', label='$q_{out}$')
        ax.axhline(tp.q_max, color='grey', ls=':', lw=0.8, label='$q_{max}$')
        ax.set_xlim(0, grd.T * 60)
        ax.set_ylim(0, tp.q_max * 1.08)
        ax.set_xlabel('Time [min]', fontsize=9)
        ax.set_ylabel('Flow [veh/h]', fontsize=9)
        ax.set_title(f'{ttl}\nBoundary flows', fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        f'Comparison: Optimal vs Robust control (δ = {delta*100:.0f}%)',
        fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig


# =============================================================================
# Figure 3 – Extreme-scenario verification
# =============================================================================

def plot_extreme_scenarios(rob: SolverResult,
                            upper: SolverResult,
                            lower: SolverResult,
                            tp: TrafficParams,
                            grd: GridParams,
                            delta: float = 0.10,
                            save_path: Optional[str] = None):
    """
    Replicate Figure 3 of the paper.

    Apply the robust controller to the two extreme IC realisations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.subplots_adjust(wspace=0.38)
    delta_pct = delta * 100

    pairs = [
        (rob,   f'Nominal IC (δ={delta_pct:.0f}%, robust ctrl)'),
        (upper, f'Upper-bound IC  ρ₀·(1+{delta_pct:.0f}%)'),
        (lower, f'Lower-bound IC  ρ₀·(1−{delta_pct:.0f}%)'),
    ]
    for ax, (res, ttl) in zip(axes, pairs):
        im = _density_imshow(ax, res.rho, grd, tp)
        ax.set_title(ttl, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('[veh/km]',
                                                                      fontsize=7)

    fig.suptitle(
        'Robust control performance in extreme IC scenarios\n'
        '(Same control inputs, different IC realisations)',
        fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig


# =============================================================================
# Figure 4 – Uncertainty sensitivity
# =============================================================================

def plot_uncertainty_sensitivity(deltas: np.ndarray,
                                 opt_obj:  np.ndarray,
                                 rob_obj:  np.ndarray,
                                 n_cong_opt: np.ndarray,
                                 n_cong_rob: np.ndarray,
                                 solve_times: np.ndarray,
                                 save_path: Optional[str] = None):
    """
    Figure 4 – Three panels: total flow, congested cells, solve time vs δ.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    pct = deltas * 100

    ax = axes[0]
    ax.plot(pct, opt_obj, 'o-', color='#1f77b4', lw=1.8, ms=5,
            label='Optimal LP')
    ax.plot(pct, rob_obj, 's--', color='#d62728', lw=1.8, ms=5,
            label='Robust MILP')
    ax.fill_between(pct, rob_obj, opt_obj, alpha=0.12, color='grey',
                    label='Conservatism gap')
    ax.set_xlabel('Uncertainty level δ [%]', fontsize=10)
    ax.set_ylabel('Total weighted flow', fontsize=10)
    ax.set_title('(a) Objective vs uncertainty', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.plot(pct, n_cong_opt, 'o-', color='#1f77b4', lw=1.8, ms=5,
            label='Optimal')
    ax.plot(pct, n_cong_rob, 's--', color='#d62728', lw=1.8, ms=5,
            label='Robust')
    ax.set_xlabel('Uncertainty level δ [%]', fontsize=10)
    ax.set_ylabel('Congested cells at $t=T$', fontsize=10)
    ax.set_title('(b) Final-time congestion', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax = axes[2]
    ax.semilogy(pct, np.maximum(solve_times, 1e-4), 's--', color='#ff7f0e',
                lw=1.8, ms=5, label='Robust MILP')
    ax.set_xlabel('Uncertainty level δ [%]', fontsize=10)
    ax.set_ylabel('Solve time [s]  (log)', fontsize=10)
    ax.set_title('(c) MILP solve time', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4, which='both')

    fig.suptitle('Uncertainty level sensitivity analysis', fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig


# =============================================================================
# Figure 5 – Parametric study
# =============================================================================

def plot_parametric_summary(results: dict,
                             param_values: Sequence,
                             param_label: str,
                             param_name: str,
                             save_path: Optional[str] = None):
    """
    Figure 5 – Two panels: objective and solve time vs a swept parameter.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    pv = np.asarray(param_values, dtype=float)

    ax = axes[0]
    ax.plot(pv, results['opt_flows'], 'o-b', lw=1.8, ms=5, label='Optimal LP')
    ax.plot(pv, results['rob_flows'], 's--r', lw=1.8, ms=5, label='Robust MILP')
    ax.set_xlabel(param_label, fontsize=10)
    ax.set_ylabel('Total weighted flow', fontsize=10)
    ax.set_title(f'Objective vs {param_name}', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.semilogy(pv, results['opt_times'], 'o-b', lw=1.8, ms=5,
                label='Optimal LP')
    ax.semilogy(pv, results['rob_times'], 's--r', lw=1.8, ms=5,
                label='Robust MILP')
    ax.set_xlabel(param_label, fontsize=10)
    ax.set_ylabel('Solve time [s]  (log)', fontsize=10)
    ax.set_title(f'Solve time vs {param_name}', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4, which='both')

    fig.suptitle(f'Parametric study — {param_name}', fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig
