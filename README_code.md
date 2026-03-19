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

### LWR PDE with Triangular Fundamental Diagram (paper eq. 1–3)

Traffic density ρ(t, x) on a single highway link [0, L] × [0, T] satisfies:

```
∂ρ/∂t  +  ∂ψ(ρ)/∂x  =  0            (LWR / eq. 1)
```

The Hamiltonian (triangular FD, paper eq. 3):

```
ψ(ρ) = v_f · ρ              for ρ ∈ [0, ρ_c]
ψ(ρ) = w · (ρ − ρ_max)      for ρ ∈ [ρ_c, ρ_max]   (w > 0 in this code)
```

| Symbol | Value | Description |
|--------|-------|-------------|
| v_f    | 60 km/h | Free-flow speed |
| w      | 15 km/h | Backward congestion-wave speed (magnitude, positive) |
| ρ_max  | 150 veh/km | Jam density |
| ρ_c    | 30 veh/km | Critical density = w·ρ_max/(v_f+w) |
| q_max  | 1800 veh/h | Road capacity = v_f·ρ_c |

### Moskowitz (Cumulative Count) Function (paper eq. 2)

The Hamilton-Jacobi (HJ) equivalent uses M(t, x) — the Moskowitz function:

```
∂M/∂t  −  ψ(−∂M/∂x)  =  0
```

so that  ρ = −∂M/∂x  and  q = ∂M/∂t.

### Lax-Hopf Formula (paper Proposition 1 / eq. 4)

For the triangular FD, the Legendre-Fenchel transform of ψ is:

```
φ*(v) = ρ_c · (v_f − v)    for v ∈ [−w, v_f]
φ*(v) = +∞                  otherwise
```

The Lax-Hopf formula gives, for any source (t₁, x₁) and target (t₂, x₂)
with v = (x₂−x₁)/(t₂−t₁) ∈ [−w, v_f]:

```
M(t₂, x₂)  ≤  M(t₁, x₁)  +  (t₂−t₁)·φ*(v)
            =  M(t₁, x₁)  +  (t₂−t₁)·q_max  −  (x₂−x₁)·ρ_c
```

This gives **all** Lax-Hopf compatibility constraints as simple linear
inequalities (paper Proposition 3 / Compatibility conditions).

---

## LP/MILP Formulation

### Decision Variables

| Variable | Size | Description |
|----------|------|-------------|
| `N[n,k]` | (K+1)×(N+1) | Moskowitz function on space-time grid |
| `q_in[m]` | K | Upstream boundary flow [veh/h] |
| `q_out[m]` | K | Downstream boundary flow [veh/h] |
| `b_cong[i]` | N | Binary: cell i congested at t=T (MILP only) |

> **Note:** The paper uses a more compact decision vector
> y = [ρ_ini(1..N), q_in(1..K), q_out(1..K)] (36 vars for the LP example).
> This code uses the full Moskowitz grid which is mathematically equivalent
> but involves more variables, because the Lax-Hopf constraints are
> implemented as explicit upper-bound inequalities rather than being evaluated
> analytically. Both formulations yield identical optimal boundary flows.

### Constraints

1. **IC equalities**: `N[0,k] = N₀[k]` — initial Moskowitz values from ρ₀
2. **Upstream BC**: `N[n,0] = Σ_{m<n} q_in[m]·Δt`
3. **Downstream BC**: `N[n,N] = N₀[N] − Σ_{m<n} q_out[m]·Δt`
4. **Lax-Hopf upper bounds** (from IC, UP BC, DN BC to every interior point):
   `N[n,k] ≤ N_src + (n−m)·Δt·q_max − (k−j)·Δx·ρ_c`
5. **Density bounds**: `0 ≤ (N[n,i]−N[n,i+1])/Δx ≤ ρ_max`
6. **Congestion** (MILP only): `ρ[K,i] ≤ ρ_c + (ρ_max−ρ_c)·b_cong[i]`

### LP Objective — paper equation (6), Section III.B

Maximise total cumulative downstream flow (no time weights, no upstream term):

```
Minimise  −Σ_{t=1}^{n_max}  q_out(t)
```

**Verification:** the LP solution sends all downstream flows to the capacity
q_max, consistent with the paper's Figure 1 caption: *"downstream flows are
all the maximal allowable flow."*

### MILP Objective — paper Section IV.B

Maximise weighted total boundary flow and penalise congested segments at t=T:

```
Minimise  −Σ_{t=1}^{n_max} w(t)·[q_out(t) + q_in(t)]  +  Σ_{i=1}^{N} b_cong(i)

where  w(t) = exp(5·(n_max − t) / n_max)   (heaviest weight on first step)
```

Upstream flows are included *"to obtain a more visible difference between the
robust solution and the optimal solution"* (paper Section IV.B).

### Robust Formulation — paper Section IV.B–C

Uncertainty model: `ρ₀[i] ∈ [ρ₀_nom[i]·(1−δ), min(ρ₀_nom[i]·(1+δ), ρ_max)]`.

The robust LP (paper eq. 10, Pbc) uses the **worst-case (lowest) Moskowitz**
initial values in every equality and Lax-Hopf constraint:

```
N₀_wc[k] = −Σ_{j<k} min(ρ₀[j]·(1+δ), ρ_max) · Δx
```

Solving with `N₀_wc` yields the tightest feasible set, guaranteeing feasibility
for **all** IC realisations in the uncertainty box (paper Section IV.B, last
paragraph: *"the lower bound of the interval which defines the smallest
feasible solution set is used to solve the robust LP"*).

---

## Paper Examples and Grid Parameters

### Section III.C — Optimal LP (Figure 1)

| Parameter | Value |
|-----------|-------|
| N | 6 segments |
| X | 643 m per segment |
| L = N·X | 3.858 km |
| K | 15 time steps |
| ΔT | 30 s = 1/120 h |
| T = K·ΔT | 450 s = 7.5 min |
| CFL α = v_f·ΔT/X | 0.778 |
| Constraints / Variables | 1559 / 148 |
| IC | partly congested, values in [0.5ρ_c, 3ρ_c] |

### Section IV.C — Robust MILP (Figures 2–3)

| Parameter | Value |
|-----------|-------|
| N | 10 segments |
| X | 300 m per segment |
| L = N·X | 3.0 km |
| K | **12** (paper uses 30; limited by Gurobi licence) |
| ΔT | 30 s = 1/120 h |
| T = K·ΔT | 360 s = 6.0 min |
| CFL α = v_f·ΔT/X | 1.667 (LP does not require α ≤ 1) |
| Constraints / Variables | 1814 / 177 |
| δ | 10% |
| ρ_meas | [3,5,2,8,6,9,10,7,1,4] × 0.5·ρ_c |

**Gurobi restricted licence:** allows ≤ 2000 variables and ≤ 2000 constraints.
The paper's K=30 requires 9149 constraints and is infeasible with this licence.
The maximum K=12 is used, giving T=6 min instead of the paper's 15 min.
The helper `count_constraints(N, K, α, β)` checks feasibility before building.

---

## File Structure

```
.
├── robust_traffic_control.py   # Core LP/MILP solver (TrafficParams, GridParams,
│                               #   RobustTrafficController, SolverResult,
│                               #   paper_lp_ic, paper_milp_ic)
├── visualization.py            # Plotting (Figs 1–5)
├── parametric_analysis.py      # Five parametric sweeps
├── uncertainty_analysis.py     # Uncertainty-level sensitivity + extreme scenarios
├── main.py                     # Main entry point
├── figures/                    # Auto-generated output figures
│   ├── fig1_optimal_control.png
│   ├── fig2_nominal_vs_robust.png
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

### Section III.C — Optimal LP (N=6, K=15, L=3.858 km)

| Metric | Value |
|--------|-------|
| Objective −Σ q_out | −27 000 veh/h |
| Downstream flows | All at q_max = 1800 veh/h ✓ |
| Total q_in | 0 veh (initial IC sufficient) |
| Solve time | ~0.002 s |

*Paper: "downstream flows are all the maximal allowable flow... in less than
7 minutes."* ✓

### Section IV.C — Robust MILP (N=10, K=12, L=3 km, δ=10%)

| | Nominal MILP (δ=0) | Robust MILP (δ=10%) |
|-|-----------|-------------|
| Objective | −931 473 | −906 137 |
| Mean upstream flow | 870 veh/h | 834 veh/h |
| Mean downstream flow | 1155 veh/h | 989 veh/h |
| Conservatism gap | — | **2.7%** |
| Solve time | 0.005 s | 0.005 s |

*Paper: "robust control outputs admit less upstream flows comparing to the
optimal solution."* ✓

### Uncertainty sensitivity (δ = 0 → 50%)

| δ | Gap |
|---|-----|
| 0% | 0.0% |
| 10% | 2.7% |
| 20% | 5.6% |
| 30% | 8.3% |
| 50% | 13.9% |

The gap grows monotonically with δ, showing increasing conservatism at
higher uncertainty levels.

---

## Formula Audit vs. Paper

| Aspect | Paper | This code |
|--------|-------|-----------|
| LWR PDE | ∂ρ/∂t + ∂ψ(ρ)/∂x = 0 | Implemented via Lax-Hopf ✓ |
| HJ PDE | ∂M/∂t − ψ(−∂M/∂x) = 0 | M = N_var on grid ✓ |
| Triangular FD | ψ(ρ) = min(v_f·ρ, w·(ρ−ρ_max)) | Same ✓ |
| Critical density | ρ_c = −w·ρ_max/(v_f−w) (w<0 in paper) | ρ_c = w·ρ_max/(v_f+w) (w>0 here) ✓ |
| Lax-Hopf φ* | φ*(v) = ρ_c·(v_f−v) on [−w, v_f] | Same ✓ |
| LP constraint | N(t₂,x₂) ≤ N(t₁,x₁) + Δt·q_max − Δx·ρ_c | Same ✓ |
| LP objective (eq. 6) | min −Σ q_out(t) | Same ✓ |
| MILP objective | min −Σ w(t)·(q_out+q_in) + Σ b_cong(i) | Same ✓ |
| Weight w(t) | exp(5·(n_max−t)/n_max) | Same ✓ |
| Congestion indicator | b_cong(i)=1 if ρ(T,i)>ρ_c | Same ✓ |
| Robust IC (eq. 9–10) | use worst-case b_c (lower RHS bound) | N₀_wc uses ρ_max cap ✓ |
| LP grid (Sec. III) | N=6, X=643m, K=15, ΔT=30s | Same ✓ |
| MILP IC (Sec. IV) | [3,5,2,8,6,9,10,7,1,4]×0.5ρ_c | Same ✓ |
| MILP grid (Sec. IV) | N=10, X=300m, K=30 | K=12 (licence limit) ⚠ |

