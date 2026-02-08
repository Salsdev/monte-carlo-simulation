# Reliability Analysis of Nigerian Timber Roof Trusses in Fire

## Probabilistic Assessment using Monte Carlo Simulation (MCS)

This project implements a high-fidelity probabilistic framework to evaluate the structural reliability of Nigerian hardwood roof trusses exposed to fire. By integrating experimental material data with stochastic modeling, the system quantifies the safety gap between indigenous timber species—**African Birch** (*Anogeissus leiocarpa*) and **Tali** (*Erythrophleum suaveolens*)—and international safety standards (EN 1990).

---

## 🛠 Methodology Overview

The project follows a four-phase research design as outlined in the PhD Thesis methodology:

### Phase 1: Material Characterization

Laboratory-determined physical and mechanical properties of Nigerian hardwoods are modeled as random variables.

* **Statistical Distributions:** Normal (Density), Lognormal (MOE), and Gumbel (Bending Strength).
* **Correlation Structure:** Incorporates physical dependencies (e.g., higher density correlates with higher strength and lower charring rates).

### Phase 2: Fire Scenario Modeling

The system evaluates trusses against three distinct fire exposures:

1. **FTI (Standard ISO 834):** Continuous logarithmic heating for 60 minutes.
2. **FTII (Parametric Kitchen):** Low ventilation, long-duration fire with a cooling phase.
3. **FTIII (Parametric Sitting Room):** High ventilation, rapid-growth fire with synthetic fuel loads.

### Phase 3: Physics & Structural Engine

* **Charring Dynamics:** A hybrid model weighting **Mikkola (1991)** heat flux and **Hietaniemi (2005)** probabilistic insulation models.
* **Thermal Profiling:** Solves 1D transient heat conduction using the Complementary Error Function (erfc) to determine core temperatures.
* **RCSM Implementation:** Uses the **EN 1995-1-2 Reduced Cross-Section Method** with a time-dependent zero-strength layer ($d_0 = 7$ mm).

### Phase 4: Reliability Assessment

* **Monte Carlo Engine:** 100,000 iterations per scenario to estimate the Probability of Failure ($P_f$).
* **Safety Metric:** Calculates the Reliability Index ($\beta$), benchmarking against the **EN 1990 target of $\beta = 3.8$**.
* **Limit States:** Evaluates Bending ($M_R$), Buckling ($N_R$), and Shear ($V_R$).

---

## 🚀 Features

* **Exact Statistics:** Uses Clopper-Pearson exact confidence intervals for $P_f$ instead of approximations.
* **Advanced Cooling Phase:** Implements the exact EN 1991-1-2 cooling decay logic based on fire duration.
* **Sensitivity Analysis:** Performs Spearman Rank Correlation to identify the primary drivers of structural failure (e.g., Load vs. Density).
* **Visualization Suite:** Generates publication-quality heatmaps, error-bar reliability plots, and failure mode distributions.

---

## 💻 Technical Implementation

The project is built using Python 3.10+ with the following stack:

* **NumPy & SciPy:** For stochastic sampling and error function mathematics.
* **Pandas:** For high-resolution data management and CSV exports.
* **Matplotlib:** For high-fidelity technical plots and heatmaps.
* **tqdm:** For real-time simulation progress tracking.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- libraries: `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`

### Execution
```bash
python3 mcs_simulation.py
```

---

## 📄 Documentation & References

* **EN 1995-1-2:** Eurocode 5 - Design of timber structures - Part 1-2: General - Structural fire design.
* **EN 1991-1-2:** Eurocode 1 - Actions on structures - Part 1-2: General actions - Actions on structures exposed to fire.
* **Thesis Reference:** *Probabilistic Reliability Analysis of Nigerian Timber Roof Trusses in Fire*, Chapter 3: Materials and Methods.

---
*Created as part of the Timber Roof Truss Fire Reliability Analysis Study.*
