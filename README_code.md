# Robust Traffic Control via MILP – Lax-Hopf Framework

## Overview

This repository reproduces the exact robust traffic boundary-control solution
from the paper:

> Li, Y., Canepa, E., & Claudel, C. (2013).  
> **"Exact solutions to robust control problems involving scalar hyperbolic
> conservation laws using Mixed Integer Linear Programming."**  
> 51st Annual Allerton Conference, UIUC, Illinois, October 2013.

The implementation is in **Python** using the **Gurobi** MILP/LP solver.

---

## Physical Model

### LWR PDE with Triangular Fundamental Diagram

Traffic density ρ(t, x) on a single highway link [0, L] × [0, T] satisfies:

```
∂ρ/∂t  +  ∂Q(ρ)/∂x  =  0            (LWR conservation law)
Q(ρ) = min(v_f · ρ,   w · (ρ_max − ρ))   (triangular FD)
```

Parameters:

| Symbol | Value | Description |
|--------|-------|-------------|
| v_f    | 60 km/h | Free-flow speed |
| w      | 15 km/h | Backward congestion wave speed |
| ρ_max  | 150 veh/km | Jam density |
| ρ_c    | 30 veh/km | Critical density = w·ρ_max/(v_f+w) |
| q_max  | 1800 veh/h | Road capacity = v_f·ρ_c |

### Moskowitz (Cumulative Count) Function

The equivalent Hamilton-Jacobi form uses N(t, x) — cumulative vehicle count:

```
ρ = −∂N/∂x,    q = ∂N/∂t
```

### Lax-Hopf Formula

For any source point (t₁, x₁) and target (t₂, x₂) within the causal range
−w ≤ (x₂−x₁)/(t₂−t₁) ≤ v_f:

```
N(t₂, x₂)  ≤  N(t₁, x₁)  +  (t₂−t₁)·q_max  −  (x₂−x₁)·ρ_c
```

This unified formula (from H*(v) = (v_f − v)·ρ_c on [−w, v_f]) gives **all**
Lax-Hopf constraints as simple linear inequalities.

---

## LP/MILP Formulation

### Decision Variables

| Variable | Size | Description |
|----------|------|-------------|
| `N[n,k]` | (K+1)×(N+1) | Moskowitz function on space-time grid |
| `q_in[m]` | K | Upstream boundary flow [veh/h] |
| `q_out[m]` | K | Downstream boundary flow [veh/h] |
| `b_cong[i]` | N | Binary: cell i congested at t=T (MILP only) |

### Constraints

1. **IC equalities**: `N[0,k] = N₀[k]`  (initial Moskowitz values from ρ₀)
2. **Upstream BC**: `N[n,0] = Σ q_in[m]·Δt`
3. **Downstream BC**: `N[n,N] = N₀[N] − Σ q_out[m]·Δt`
4. **Lax-Hopf** (from IC, UP BC, DN BC to every interior point)
5. **Density bounds**: `0 ≤ (N[n,i]−N[n,i+1])/Δx ≤ ρ_max`
6. **Congestion** (MILP only): `ρ[K,i] ≤ ρ_c + (ρ_max−ρ_c)·b_cong[i]`

### Objective

```
Minimise  −Σ_m w(m)·[q_out(m) + q_in(m)]  +  Σ_i b_cong(i)
```

where `w(m) = exp(5·(K−1−m)/K)` is a time-decaying weight (higher weight
encourages early congestion clearance).

### Robust Formulation

Uncertainty model: `ρ₀[i] ∈ [ρ₀_nom[i]·(1−δ),  min(ρ₀_nom[i]·(1+δ), ρ_max)]`.

The robust controller uses **worst-case** (lower-bound) Moskowitz initial
values in **all** equality and Lax-Hopf constraints:

```
N₀_wc[k] = −Σ_{j<k} min(ρ₀[j]·(1+δ), ρ_max) · Δx
```

This ensures the control remains feasible for every IC realisation in the
uncertainty set, including near-jam conditions.

---

## Grid Parameters and Licence Constraints

The Gurobi restricted (size-limited) licence allows at most **2000 variables**
and **2000 linear constraints**. The default grid:

```
L=10 km,  T=12 min,  N=10 cells,  K=12 time steps
Δx=1 km,  Δt=1 min,  CFL α=1 (free-flow),  β=0.25 (congestion)
MILP: 1654 constraints, 177 variables  ✓
```

The helper `count_constraints(N, K, α, β)` checks feasibility before building.

---

## File Structure

```
.
├── robust_traffic_control.py   # Core LP/MILP solver (TrafficParams, GridParams,
│                               #   RobustTrafficController, SolverResult)
├── visualization.py            # Plotting (Figs 1–5)
├── parametric_analysis.py      # Five parametric sweeps
├── uncertainty_analysis.py     # Uncertainty-level sensitivity + extreme scenarios
├── main.py                     # Main entry point
├── figures/                    # Auto-generated output figures
│   ├── fig1_optimal_control.png
│   ├── fig2_optimal_vs_robust.png
│   ├── fig3_extreme_scenarios.png
│   ├── fig4_uncertainty_sensitivity.png
│   └── fig5_<study>_parametric.png  (×5)
└── README_code.md              # This file
```

---

## Usage

```bash
pip install gurobipy numpy matplotlib
python main.py
```

The script runs all sections in ~5 seconds and saves figures to `./figures/`.

---

## Results Summary

### Main scenario (N=10, K=12, T=12 min, δ=10%)

| | Optimal LP | Robust MILP |
|-|-----------|-------------|
| Objective | −735 187 | −650 813 |
| Conservatism gap | — | **11.5 %** |
| Solve time | 0.002 s | 0.003 s |
| Mean upstream flow | 712 veh/h | 559 veh/h |
| Mean downstream flow | 1800 veh/h | 1800 veh/h |

**Key observations:**
- The robust controller reduces upstream inflow by ~22% compared to optimal,
  providing a safety margin against worst-case initial congestion.
- The downstream outflow hits the capacity q_max in both cases.
- The MILP solves in < 5 ms, confirming the paper's claim of very fast solution.

### Uncertainty sensitivity (δ = 0→50%)

| δ | Gap |
|---|-----|
| 0% | 0.0% |
| 10% | 11.5% |
| 20% | 23.0% |
| 30% | 28.7% |
| 50% | 29.0% |

The gap grows roughly linearly with δ up to ~20%, then saturates because
worst-case densities are capped at ρ_max (jam density cannot be exceeded).

### Parametric studies

1. **Road length L** – Flow objective scales with L; robust gap remains ~13%.
2. **Time horizon T** – Longer horizon increases total flow linearly.
3. **Spatial resolution N** – Flow is stable across N=4..10 for fixed T.
4. **Initial density** – Gap increases sharply as ρ₀ → ρ_max (near jam).
5. **Free-flow speed v_f** – Higher v_f increases capacity and total throughput.

---

## Simplifications vs. Paper

| Aspect | Paper | This implementation |
|--------|-------|---------------------|
| Road topology | Single link | Single link ✓ |
| FD shape | Triangular | Triangular ✓ |
| Lax-Hopf constraints | All-pairs (IC + BC) | All-pairs ✓ |
| Uncertainty model | Additive on IC | Multiplicative, capped at ρ_max |
| Grid size | ~30 time steps | K=12 (Gurobi size limit) |
| Binary variables | b_cong (yes) | b_cong ✓ |
| Time weighting | exp(5(T−t)/T) | Same ✓ |

The main simplification is the smaller grid (K=12 vs. ~30 in the paper) due
to the Gurobi restricted-licence limit of 2000 constraints. The mathematical
structure and results are otherwise faithful to the paper.
