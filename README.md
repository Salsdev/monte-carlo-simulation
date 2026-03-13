# 🔥 Fire Reliability Analysis of Structural Timber
## High-Fidelity Monte Carlo Simulation (MCS) Framework

A probabilistic framework for evaluating the structural reliability of timber roof trusses exposed to fire. The simulation integrates thermo-mechanical charring models with stochastic sampling to quantify safety margins and reliability indices (β) per **EN 1995-1-2** and **ISO 834**.

> **Species studied:** *Anogeissus leiocarpa* (White Wood) and *Erythrophleum suaveolens* (Red Wood) — Nigerian hardwoods.

---

## 📂 Project Structure

| File | Description |
|---|---|
| `mcs_simulation.py` | Core simulation engine — charring models, limit states, sampling, and analysis |
| `app.py` | Interactive Streamlit dashboard with Plotly visualisations |
| `requirements.txt` | Python dependencies |

---

## 🛠 Methodology

### 1. Stochastic Material & Load Framework
Input variables are sampled from statistical distributions to capture natural variability:

- **Material Properties** — Bending Strength (*Gumbel*), Compressive Strength (*Lognormal*), Modulus of Elasticity (*Lognormal*), Density (*Normal*), Moisture Content (*Normal*), Shear Strength (*Lognormal*)
- **Correlated Sampling** — A **Gaussian Copula** (Cholesky decomposition) enforces realistic inter-variable correlations per the JCSS Probabilistic Model Code (Eqns 3.78–3.82)
- **Stochastic Loads** — Dead Load (*Normal*) and Live Load (*Gumbel*), scaled by uncertainty factors (θ_R, θ_E, θ_model)

### 2. Multi-Model Charring Dynamics
The effective charring rate (β_eff) is a **weighted hybrid** (40 / 30 / 30):

| Weight | Model | Basis |
|--------|-------|-------|
| 40% | **Experimental** | Site-specific fire tests for Nigerian hardwoods |
| 30% | **Mikkola (1991)** | Net heat flux model — energy for pyrolysis & water evaporation |
| 30% | **Hietaniemi (2005)** | Time-dependent oxygen factor & thermal insulation decay |

### 3. Fire Scenarios

| Key | Name | Type | Duration |
|-----|------|------|----------|
| FTI | Standard ISO 834 | Standard | 60 min |
| FTII | Parametric Kitchen (Low Vent) | Parametric | 43 min |
| FTIII | Parametric Sitting Room (High Vent) | Parametric | 45 min |

### 4. Reduced Cross-Section Method (RCSM) — Numerical Integration
Instead of a simplified fixed zero-strength layer, the residual beam is **discretised into 20 thermal layers**:

- Transient heat conduction maps temperatures within each layer
- Layer-wise reduction factors (`k_mod,fi` for strength, `k_E,fi` for stiffness) are applied per instantaneous temperature
- Shifted neutral axis — dynamically calculates effective section modulus (W_ef) and moment of inertia (I_ef)

### 5. Truss Configurations

| Configuration | Description | Members |
|---|---|---|
| **Double-Howe** | Verified 6 m truss (Chapter 4) | Top Chord, Bottom Chord, Compression Web, Tension Web |
| **Mono-pitch** | Single-slope extension | Top Chord, Vertical Web |

---

## 🏗 Structural Limit States (FM1–FM8)

Eight failure modes are evaluated every minute across all truss members:

| FM | Member | Check | Eurocode Ref |
|---|---|---|---|
| **FM1** | Top Chord | Pure Buckling — Euler critical load & relative slenderness (λ_fi) | Eqn 3.3 |
| **FM2** | Top Chord | Combined Bending & Axial Compression — instability interaction | Eqn 3.14 |
| **FM3** | Bottom Chord | Tension Rupture — effective tension area vs. axial load | Eqn 3.22 |
| **FM4** | Bottom Chord | Pure Bending — strength-weighted W_ef vs. design moment | Eqn 3.28 |
| **FM4a** | Bottom Chord | Combined Tension & Bending — interaction check | Eqn 3.32 |
| **FM5** | Bottom Chord | Lateral Torsional Buckling — out-of-plane instability | Eqn 3.34 |
| **FM6** | Web (Compression) | Compression Buckling — web member stability | Eqn 3.40 |
| **FM7** | Web (Tension) | Tension Rupture — web member capacity | Eqn 3.48 |
| **FM8** | All Members | Shear — geometric residual area vs. shear demand | Eqn 3.52 |

An additional **Burnout** check identifies members that have completely charred away (A_ef ≤ 0).

---

## 📊 Interactive Dashboard (Streamlit)

The dashboard (`app.py`) provides a premium dark-themed UI with five analysis tabs:

1. **Summary Table** — Reliability results across all b × h combinations with CSV export
2. **Failure Mode Distribution** — Horizontal bar & donut charts for FM1–FM8 breakdown
3. **Parametric Heatmap** — β or Pf heat map over the width/depth grid (with 3D surface option)
4. **Convergence** — Running Pf and β vs. iteration count, with convergence check (Eqn 3.90)
5. **Sensitivity** — Spearman rank correlation tornado chart identifying critical design parameters

### Results & Statistics
- **Probability of Failure (Pf)** — failure ratio across N iterations (typically 10,000–100,000)
- **Reliability Index (β)** — derived as β = −Φ⁻¹(Pf)
- **95% Confidence Intervals** — **Clopper-Pearson** exact binomial method
- **Sensitivity Analysis** — **Spearman Rank Correlation** of input variables with failure

---

## 💻 Getting Started

### Prerequisites
- Python 3.10+

### Installation
```bash
pip install -r requirements.txt
```

### Run the Simulation (CLI)
```bash
python3 mcs_simulation.py
```

### Launch the Dashboard
```bash
streamlit run app.py
```

---

*Developed for the research of Fire Reliability of Timber Structural Elements.*
