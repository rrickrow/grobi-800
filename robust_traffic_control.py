"""
robust_traffic_control.py
=========================

Exact solution of the robust/optimal traffic boundary-control problem for the
LWR (Lighthill-Whitham-Richards) scalar conservation law, using the
Lax-Hopf / Hamilton-Jacobi framework and the Gurobi MILP/LP solver.

Reference
---------
Li, Y., Canepa, E., & Claudel, C. (2013).
"Exact solutions to robust control problems involving scalar hyperbolic
conservation laws using Mixed Integer Linear Programming."
51st Annual Allerton Conference, UIUC, Illinois, October 2013.

Model summary
-------------
Traffic density ρ(t, x) on a single highway link [0, L] × [0, T] obeys:

    ∂ρ/∂t  +  ∂Q(ρ)/∂x  =  0           (LWR PDE)
    Q(ρ)  =  min(v_f·ρ,  w·(ρ_max − ρ))  (triangular fundamental diagram)

Moskowitz (cumulative vehicle count) function N(t, x) satisfies:

    N_t  =  Q(−N_x)     so that ρ = −∂N/∂x,  q = ∂N/∂t

Lax-Hopf upper-bound constraints
---------------------------------
For any source (t₁, x₁) → target (t₂, x₂) with dt = t₂−t₁ > 0 and
characteristic speed v = (x₂−x₁)/dt ∈ [−w, v_f]:

    N(t₂, x₂) ≤ N(t₁, x₁)  +  dt·q_max  −  (x₂−x₁)·ρ_c

where  ρ_c = w·ρ_max/(v_f+w)  and  q_max = v_f·ρ_c.

The unified formula  dt·H*(v) = dt·q_max − (x₂−x₁)·ρ_c  comes from the
Legendre-Fenchel transform H*(v) = (v_f − v)·ρ_c  of the Hamiltonian.

Three condition types are encoded as linear upper-bound constraints:
  (a) Initial conditions (IC):   source at (0, j·Δx), j = 0…N
  (b) Upstream boundary (UP BC): source at (m·Δt, 0), m = 0…K−1
  (c) Downstream boundary (DN BC): source at (m·Δt, L), m = 0…K−1

Robust formulation
------------------
Uncertainty: ρ₀[i] ∈ [ρ₀_nom[i]·(1−δ),  ρ₀_nom[i]·(1+δ)].
The robust controller uses worst-case (lower-bound) Moskowitz IC values.
Binary congestion indicators b_cong[i] convert the LP to a MILP.

License note
------------
The Gurobi size-limited licence allows ≤ 2000 variables and ≤ 2000
linear constraints.  Grid parameters should be chosen accordingly; the
helper ``max_grid_K`` computes the largest K that fits for a given N.
"""

import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Hard limit for restricted Gurobi licence
_GUROBI_CONSTR_LIMIT = 1950   # leave a small safety margin
_GUROBI_VAR_LIMIT    = 1950


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class TrafficParams:
    """
    Parameters of the triangular fundamental diagram (FD).

    The triangular FD is:  Q(ρ) = min(v_f · ρ,  w · (ρ_max − ρ))

    Attributes
    ----------
    v_f : float   Free-flow speed [km/h].  Default 60.
    w   : float   Backward congestion-wave speed (positive) [km/h].  Default 15.
    rho_max : float  Jam density [veh/km].  Default 150.
    """
    v_f: float = 60.0
    w: float = 15.0
    rho_max: float = 150.0

    def __post_init__(self):
        # Critical density: kink of triangular FD (onset of congestion)
        self.rho_c = self.w * self.rho_max / (self.v_f + self.w)
        # Road capacity [veh/h]
        self.q_max = self.v_f * self.rho_c

    def flow(self, rho: np.ndarray) -> np.ndarray:
        """Compute flow from density using triangular FD."""
        rho = np.asarray(rho, dtype=float)
        return np.minimum(self.v_f * rho, self.w * (self.rho_max - rho))

    def __str__(self):
        return (f"TrafficParams(v_f={self.v_f} km/h, w={self.w} km/h, "
                f"ρ_max={self.rho_max} veh/km, "
                f"ρ_c={self.rho_c:.2f} veh/km, q_max={self.q_max:.1f} veh/h)")


@dataclass
class GridParams:
    """
    Space-time grid for the discretised LWR model.

    Attributes
    ----------
    L : float   Road length [km].         Default 10.
    T : float   Simulation horizon [h].   Default 0.2  (= 12 min for CFL=1).
    N : int     Number of spatial cells.  Default 10.
    K : int     Number of time steps.     Default 12.
    """
    L: float = 10.0
    T: float = 0.2
    N: int = 10
    K: int = 12

    def __post_init__(self):
        self.dx = self.L / self.N
        self.dt = self.T / self.K
        self.x_edges = np.linspace(0.0, self.L, self.N + 1)
        self.t_edges = np.linspace(0.0, self.T, self.K + 1)
        self.x_c = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])

    def cfl_numbers(self, tp: TrafficParams):
        """Return CFL numbers α (free-flow) and β (congestion)."""
        return tp.v_f * self.dt / self.dx, tp.w * self.dt / self.dx

    def check_cfl(self, tp: TrafficParams) -> bool:
        alpha, beta = self.cfl_numbers(tp)
        ok = True
        if alpha > 1.0 + 1e-6:
            warnings.warn(
                f"CFL violated (free-flow): α = {alpha:.3f} > 1. "
                "Increase K or decrease N.")
            ok = False
        if beta > 1.0 + 1e-6:
            warnings.warn(
                f"CFL violated (congestion): β = {beta:.3f} > 1. "
                "Increase K or decrease N.")
            ok = False
        return ok

    def __str__(self):
        return (f"GridParams(L={self.L} km, T={self.T} h = {self.T*60:.1f} min, "
                f"N={self.N}, K={self.K}, "
                f"Δx={self.dx:.3f} km, Δt={self.dt*60:.2f} min)")


@dataclass
class SolverResult:
    """
    Solution returned by :class:`RobustTrafficController`.

    Attributes
    ----------
    N_matz : ndarray (K+1, N+1)
        Moskowitz values on the grid; N_matz[n, k] = N(n·Δt, k·Δx).
    q_in   : ndarray (K,)   Upstream boundary flow [veh/h].
    q_out  : ndarray (K,)   Downstream boundary flow [veh/h].
    rho    : ndarray (K+1, N)  Density [veh/km]; ρ[n,i]=(N[n,i]−N[n,i+1])/Δx.
    flow_grid : ndarray (K+1, N+1)  Flow [veh/h]; q[n,k]=(N[n+1,k]−N[n,k])/Δt.
    b_cong : ndarray (N,) or None
        Binary congestion flag at final time (1 = congested). None for LP.
    obj_value  : float  Gurobi objective value.
    solve_time : float  Wall-clock solve time [s].
    status     : str    Gurobi status string.
    """
    N_matz: np.ndarray
    q_in: np.ndarray
    q_out: np.ndarray
    rho: np.ndarray
    flow_grid: np.ndarray
    b_cong: Optional[np.ndarray]
    obj_value: float
    solve_time: float
    status: str


# =============================================================================
# Constraint-count helper (for licence compliance)
# =============================================================================

def count_constraints(N: int, K: int,
                      alpha: float = 1.0,
                      beta: float = 0.25) -> tuple:
    """
    Count the number of (MILP constraints, variables) for a given grid.

    Parameters
    ----------
    N, K   : grid dimensions.
    alpha  : free-flow CFL number  v_f·Δt/Δx.
    beta   : congestion CFL number w·Δt/Δx.

    Returns
    -------
    (n_constraints, n_variables) : ints
    """
    ic_eq = N + 1
    up_eq = K + 1
    dn_eq = K + 1
    density_bounds = (K + 1) * N * 2
    lh = 0
    for n in range(1, K + 1):
        for k in range(N + 1):
            # IC all-pairs
            j_lo = max(0, int(np.ceil(k - n * alpha - 1e-9)))
            j_hi = min(N, int(np.floor(k + n * beta + 1e-9)))
            lh += max(0, j_hi - j_lo + 1)
            # UP BC all-pairs
            for m in range(n):
                v = k * alpha / (n - m)
                if -beta - 1e-9 <= v <= alpha + 1e-9:
                    lh += 1
            # DN BC all-pairs
            for m in range(n):
                v = (k - N) * alpha / (n - m)
                if -beta - 1e-9 <= v <= alpha + 1e-9:
                    lh += 1
    total_constr = ic_eq + up_eq + dn_eq + density_bounds + lh + N  # +N for b_cong
    total_vars   = (K + 1) * (N + 1) + 2 * K + N                   # +N binary
    return total_constr, total_vars


def max_grid_K(N: int, tp: TrafficParams, grd_template: GridParams,
               limit: int = _GUROBI_CONSTR_LIMIT) -> int:
    """
    Find the maximum K such that the MILP stays within the Gurobi licence limit.

    Parameters
    ----------
    N            : number of spatial cells.
    tp           : TrafficParams (needed for CFL numbers).
    grd_template : GridParams template (used for dx).
    limit        : maximum allowed number of constraints.

    Returns
    -------
    K : int
    """
    dx = grd_template.L / N
    for K in range(2, 200):
        dt = dx / tp.v_f   # exact CFL=1 step
        alpha = tp.v_f * dt / dx
        beta  = tp.w   * dt / dx
        nc, nv = count_constraints(N, K, alpha, beta)
        if nc > limit or nv > _GUROBI_VAR_LIMIT:
            return max(2, K - 1)
    return 200


# =============================================================================
# Core solver
# =============================================================================

class RobustTrafficController:
    """
    Build and solve the LP (optimal) or MILP (robust) single-link
    boundary-traffic-control problem using the Lax-Hopf framework.

    The Moskowitz function N(t, x) is the primary decision variable.
    LWR physics is encoded as Lax-Hopf upper-bound constraints; boundary
    flows q_in / q_out are the control inputs.

    Usage
    -----
    >>> tp  = TrafficParams(v_f=60, w=15, rho_max=150)
    >>> grd = GridParams(L=10, T=0.2, N=10, K=12)
    >>> ctrl = RobustTrafficController(tp, grd)
    >>> opt_result = ctrl.solve_optimal(rho_0)
    >>> rob_result = ctrl.solve_robust(rho_0, delta=0.10)
    """

    def __init__(self, tp: TrafficParams, grd: GridParams):
        self.tp  = tp
        self.grd = grd
        grd.check_cfl(tp)
        # Warn if model will be too large for the restricted licence
        alpha, beta = grd.cfl_numbers(tp)
        nc, nv = count_constraints(grd.N, grd.K, alpha, beta)
        if nc > _GUROBI_CONSTR_LIMIT or nv > _GUROBI_VAR_LIMIT:
            raise ValueError(
                f"Grid (N={grd.N}, K={grd.K}) requires {nc} constraints "
                f"and {nv} variables, which exceeds the Gurobi size-limited "
                f"licence (max {_GUROBI_CONSTR_LIMIT} constraints, "
                f"{_GUROBI_VAR_LIMIT} variables). "
                f"Reduce N or K.  Use count_constraints() to check first.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_optimal(self, rho_0: np.ndarray,
                      verbose: bool = False) -> SolverResult:
        """
        Solve the **optimal** boundary-control LP (no uncertainty, no binaries).

        Objective: maximise Σ w(m)·[q_out(m) + q_in(m)], where
        w(m) = exp(5·(K−1−m)/K) is a time-decaying weight (heavier weight
        on early time steps encourages swift congestion clearance).

        Parameters
        ----------
        rho_0   : array_like (N,)  Nominal initial density [veh/km].
        verbose : bool             Print Gurobi log.

        Returns
        -------
        SolverResult
        """
        return self._solve(rho_0, robust=False, delta=0.0, verbose=verbose)

    def solve_robust(self, rho_0: np.ndarray,
                     delta: float = 0.10,
                     verbose: bool = False) -> SolverResult:
        """
        Solve the **robust** boundary-control MILP.

        Uncertainty model: each initial density lies in
        ρ₀[i] ∈ [ρ₀_nom[i]·(1−δ),  ρ₀_nom[i]·(1+δ)].

        The robust controller uses worst-case (lower-bound) Moskowitz IC
        values in all Lax-Hopf constraints, yielding the smallest feasible
        set and guaranteeing feasibility for every IC realisation in the
        uncertainty box.

        Binary variables b_cong[i] penalise congested segments at t = T:
        b_cong[i] = 1 forces ρ(T, x_i) ≤ ρ_max (not ≤ ρ_c).

        Parameters
        ----------
        rho_0   : array_like (N,)  Nominal initial density [veh/km].
        delta   : float            Relative uncertainty level (0 < δ ≤ 1).
        verbose : bool             Print Gurobi log.

        Returns
        -------
        SolverResult
        """
        return self._solve(rho_0, robust=True, delta=delta, verbose=verbose)

    # ------------------------------------------------------------------
    # Internal solver
    # ------------------------------------------------------------------

    def _solve(self, rho_0: np.ndarray,
               robust: bool, delta: float,
               verbose: bool) -> SolverResult:
        """Build and solve the Gurobi LP or MILP."""
        tp  = self.tp
        grd = self.grd
        N, K   = grd.N, grd.K
        dx, dt = grd.dx, grd.dt
        q_max   = tp.q_max
        rho_c   = tp.rho_c
        rho_max = tp.rho_max
        v_f, w  = tp.v_f, tp.w

        rho_0 = np.asarray(rho_0, dtype=float)
        if rho_0.shape != (N,):
            raise ValueError(f"rho_0 must have shape ({N},), got {rho_0.shape}")

        alpha, beta = grd.cfl_numbers(tp)   # CFL numbers

        # ── Moskowitz IC values ────────────────────────────────────────────
        # N(0, k·Δx) = −Σ_{j<k} ρ₀[j]·Δx   (N(0,0) = 0)
        N0_nom = np.zeros(N + 1)
        for k in range(1, N + 1):
            N0_nom[k] = -np.sum(rho_0[:k]) * dx

        # Worst-case lower bounds (higher density ⟹ more negative N)
        # Density is capped at ρ_max (physical jam density cannot be exceeded)
        N0_wc = np.zeros(N + 1)
        for k in range(1, N + 1):
            rho_wc = np.minimum(rho_0[:k] * (1.0 + delta), rho_max)
            N0_wc[k] = -np.sum(rho_wc) * dx

        # For the robust formulation the LP uses WORST-CASE Moskowitz initial
        # values throughout, so that all constraints are simultaneously
        # consistent with the same (lower-bound) IC realisation.
        # For the optimal LP the nominal values are used.
        N0_eq = N0_wc if robust else N0_nom   # used in equality constraints
        N0_lh = N0_wc if robust else N0_nom   # used in Lax-Hopf constraints

        # ── Build Gurobi model ─────────────────────────────────────────────
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 1 if verbose else 0)
        env.start()
        model = gp.Model(
            name='robust_milp' if robust else 'optimal_lp', env=env)

        # ── Variables ──────────────────────────────────────────────────────
        # Moskowitz function on full (K+1)×(N+1) grid
        N_var   = model.addVars(K + 1, N + 1, lb=-GRB.INFINITY, name='N')
        # Control inputs: upstream and downstream boundary flows
        q_in_v  = model.addVars(K, lb=0.0, ub=q_max, name='q_in')
        q_out_v = model.addVars(K, lb=0.0, ub=q_max, name='q_out')
        # Binary congestion indicators at t = K·Δt (MILP only)
        b_cong_v = (model.addVars(N, vtype=GRB.BINARY, name='b_cong')
                    if robust else None)

        # ── Initial-condition equality constraints ─────────────────────────
        # Fix N(0, k·Δx) = N0_eq[k]  for all k
        # (nominal for LP, worst-case lower bound for robust MILP)
        for k in range(N + 1):
            model.addConstr(N_var[0, k] == N0_eq[k], name=f'ic_{k}')

        # ── Upstream boundary equality constraints ─────────────────────────
        # N(n·Δt, 0) = Σ_{m<n} q_in[m]·Δt   (cumulative upstream count)
        for n in range(K + 1):
            model.addConstr(
                N_var[n, 0] == gp.quicksum(q_in_v[m] * dt for m in range(n)),
                name=f'up_bc_{n}')

        # ── Downstream boundary equality constraints ───────────────────────
        # N(n·Δt, L) = N0_eq[N] − Σ_{m<n} q_out[m]·Δt
        # (uses worst-case N0 for robust so all IC constraints remain feasible)
        for n in range(K + 1):
            model.addConstr(
                N_var[n, N] == (N0_eq[N]
                                - gp.quicksum(q_out_v[m] * dt for m in range(n))),
                name=f'dn_bc_{n}')

        # ── Lax-Hopf upper-bound constraints ──────────────────────────────
        #
        # For each target (n, k) with n ≥ 1 and source (t₁, x₁):
        #   N[n,k] ≤ N_src + (n−m)·dt·q_max − (k−j)·dx·ρ_c
        # provided  −w ≤ (k−j)·dx/((n−m)·dt) ≤ v_f  (causal range).
        #
        # Three source families:
        #   (a) IC      : (0, j)   — N_src = N0_lh[j] (constant, worst-case)
        #   (b) UP BC   : (m, 0)   — N_src = N_var[m, 0] (decision variable)
        #   (c) DN BC   : (m, N)   — N_src = N_var[m, N] (decision variable)

        for n in range(1, K + 1):
            for k in range(N + 1):

                # (a) IC all-pairs -------------------------------------------
                j_lo = max(0, int(np.ceil(k - n * alpha - 1e-9)))
                j_hi = min(N, int(np.floor(k + n * beta  + 1e-9)))
                for j in range(j_lo, j_hi + 1):
                    v = (k - j) * dx / (n * dt)
                    if -w - 1e-9 <= v <= v_f + 1e-9:
                        rhs = N0_lh[j] + n * dt * q_max - (k - j) * dx * rho_c
                        model.addConstr(
                            N_var[n, k] <= rhs, name=f'lh_ic_{n}_{k}_{j}')

                # (b) Upstream BC all-pairs -----------------------------------
                for m in range(n):
                    dtn = (n - m) * dt
                    v   = k * dx / dtn
                    if -w - 1e-9 <= v <= v_f + 1e-9:
                        cost = dtn * q_max - k * dx * rho_c
                        model.addConstr(
                            N_var[n, k] <= N_var[m, 0] + cost,
                            name=f'lh_up_{n}_{k}_{m}')

                # (c) Downstream BC all-pairs ---------------------------------
                for m in range(n):
                    dtn = (n - m) * dt
                    v   = (k - N) * dx / dtn
                    if -w - 1e-9 <= v <= v_f + 1e-9:
                        cost = dtn * q_max - (k - N) * dx * rho_c
                        model.addConstr(
                            N_var[n, k] <= N_var[m, N] + cost,
                            name=f'lh_dn_{n}_{k}_{m}')

        # ── Physical density bounds ────────────────────────────────────────
        # 0 ≤ ρ[n, i] = (N[n,i] − N[n,i+1]) / Δx ≤ ρ_max
        for n in range(K + 1):
            for i in range(N):
                model.addConstr(
                    N_var[n, i] - N_var[n, i + 1] >= 0,
                    name=f'rho_lb_{n}_{i}')
                model.addConstr(
                    N_var[n, i] - N_var[n, i + 1] <= rho_max * dx,
                    name=f'rho_ub_{n}_{i}')

        # ── Congestion-indicator constraints (MILP only) ───────────────────
        # b_cong[i] = 1  iff  ρ[K, i] > ρ_c  at the final time.
        # Encoding:  ρ[K,i] ≤ ρ_c + (ρ_max − ρ_c)·b_cong[i]
        # → when b_cong=0 this forces ρ ≤ ρ_c (free-flow at horizon).
        if robust:
            for i in range(N):
                model.addConstr(
                    (N_var[K, i] - N_var[K, i + 1]) / dx
                    <= rho_c + (rho_max - rho_c) * b_cong_v[i],
                    name=f'cong_{i}')

        # ── Objective function ─────────────────────────────────────────────
        # Time-decaying weight: w(m) = exp(5·(K−1−m)/K)
        # (heavier weight on early time steps, consistent with paper)
        weights = np.exp(5.0 * np.arange(K - 1, -1, -1) / K)

        if robust:
            # MILP objective (paper eq. 10):
            # min −Σ_m w(m)·[q_out(m)+q_in(m)]  +  Σ_i b_cong(i)
            obj = (
                -gp.quicksum(weights[m] * (q_out_v[m] + q_in_v[m])
                             for m in range(K))
                + gp.quicksum(b_cong_v[i] for i in range(N))
            )
        else:
            # LP objective: min −Σ_m w(m)·[q_out(m)+q_in(m)]
            obj = -gp.quicksum(weights[m] * (q_out_v[m] + q_in_v[m])
                               for m in range(K))

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()

        # ── Solve ──────────────────────────────────────────────────────────
        t_start = time.perf_counter()
        model.optimize()
        solve_time = time.perf_counter() - t_start

        _STATUS = {GRB.OPTIMAL: 'Optimal', GRB.INFEASIBLE: 'Infeasible',
                   GRB.INF_OR_UNBD: 'Infeasible/Unbounded',
                   GRB.UNBOUNDED: 'Unbounded', GRB.TIME_LIMIT: 'Time limit'}
        status = _STATUS.get(model.Status, f'Code {model.Status}')

        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Gurobi returned non-optimal status: {status}. "
                "Check problem formulation or grid parameters.")

        # ── Extract solution ───────────────────────────────────────────────
        N_sol = np.array([[N_var[n, k].X for k in range(N + 1)]
                          for n in range(K + 1)])
        q_in_sol  = np.array([q_in_v[m].X  for m in range(K)])
        q_out_sol = np.array([q_out_v[m].X for m in range(K)])
        b_cong_sol = (np.array([b_cong_v[i].X for i in range(N)])
                      if robust else None)

        # Density: ρ[n,i] = (N[n,i] − N[n,i+1]) / Δx
        rho_sol = (N_sol[:, :-1] - N_sol[:, 1:]) / dx
        # Flow: q[n,k] = (N[n+1,k] − N[n,k]) / Δt  (zero-padded at n=K)
        flow_sol = np.zeros((K + 1, N + 1))
        flow_sol[:K] = (N_sol[1:] - N_sol[:K]) / dt

        return SolverResult(
            N_matz=N_sol,
            q_in=q_in_sol,
            q_out=q_out_sol,
            rho=rho_sol,
            flow_grid=flow_sol,
            b_cong=b_cong_sol,
            obj_value=model.ObjVal,
            solve_time=solve_time,
            status=status,
        )


# =============================================================================
# Convenience helpers
# =============================================================================

def make_piecewise_ic(segments: list,
                      tp: TrafficParams,
                      grd: GridParams) -> np.ndarray:
    """
    Create a piecewise-constant initial density profile.

    Parameters
    ----------
    segments : list of (x_start, x_end, rho_fraction)
        Each tuple defines [x_start, x_end) with density = rho_fraction·ρ_max.
    tp, grd  : model parameters.

    Returns
    -------
    rho_0 : ndarray (N,)
    """
    rho_0 = np.zeros(grd.N)
    for x0, x1, frac in segments:
        i0 = max(0, int(np.floor(x0 / grd.dx + 1e-9)))
        i1 = min(grd.N, int(np.ceil(x1  / grd.dx - 1e-9)))
        rho_0[i0:i1] = frac * tp.rho_max
    return rho_0


def paper_default_ic(tp: TrafficParams, grd: GridParams) -> np.ndarray:
    """
    Six-segment piecewise-constant initial density as used in Figure 1 of
    the paper (partially congested highway).
    """
    L = grd.L
    # Alternating congested / free-flow / congested segments
    segs = [
        (0.00 * L, 0.20 * L, 0.80),   # heavily congested  (120 veh/km)
        (0.20 * L, 0.35 * L, 0.45),   # near critical
        (0.35 * L, 0.55 * L, 0.15),   # free-flow
        (0.55 * L, 0.70 * L, 0.72),   # congested
        (0.70 * L, 0.85 * L, 0.30),   # light traffic
        (0.85 * L, 1.00 * L, 0.55),   # moderately congested
    ]
    return make_piecewise_ic(segs, tp, grd)
