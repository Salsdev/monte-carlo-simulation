# Fire Reliability Analysis of Structural Timber
## High-Fidelity Monte Carlo Simulation (MCS) Framework

This repository contains a professional-grade probabilistic framework for evaluating the structural reliability of timber beams exposed to fire. The simulation integrates advanced thermo-mechanical models with stochastic sampling to quantify safety margins and reliability indices ($\beta$) per international standards (Eurocodes).

---

## 🛠 Project Methodology

The simulation implements a multi-physics approach to model the degradation of timber performance during fire exposure.

### 1. Stochastic Material & Load Framework
To account for natural variability in wood properties and structural loading, the system samples input variables from statistical distributions:
*   **Material Properties**: Bending Strength (Gumbel), Modulus of Elasticity (Lognormal), Density (Normal), and Moisture Content (Normal).
*   **Stochastic Loads**: Design fire loads are calculated using a combination of Normal (Dead Load) and Gumbel (Live Load) distributions, adjusted by uncertainty factors ($\theta$).
*   **Model Probabilities**: Includes multiplicative uncertainty factors for resistance ($\theta_R$), loading ($\theta_E$), and the charring model itself.

### 2. Multi-Model Charring Dynamics
The effective charring rate ($\beta_{eff}$) is calculated using a weighted hybrid approach (40/30/30) to maximize robustness:
*   **Experimental Data (40%)**: Derived from site-specific fire tests for Nigerian hardwoods (*Anogeissus leiocarpa* and *Erythrophleum suaveolens*).
*   **Mikkola Physics Model (30%)**: A net heat flux model based on the energy required for wood pyrolysis and water evaporation.
*   **Hietaniemi Analytical Model (30%)**: Accounts for time-dependent oxygen factors and thermal insulation decay.

### 3. Integrated Reduced Cross-Section Method (RCSM)
Unlike simplified models that use a fixed zero-strength layer, this framework uses **Numerical Integration**:
*   **Discretization**: The residual beam is divided into 20 thermal layers.
*   **Thermal Profiling**: Solves transient heat conduction to map temperatures within each layer.
*   **Layer-Wise Reduction**: Specific reduction factors ($k_{mod,fi}$ for strength and $k_{E,fi}$ for stiffness) are applied to each layer based on its instantaneous temperature.
*   **Shifted Neutral Axis**: Dynamically calculates the section modulus ($W_{ef}$) and moment of inertia ($I_{ef}$) as the beam chars and the thermal gradient shifts.

---

## 🏗 Structural Reliability Engine

The framework evaluates three critical **Limit State Functions (LSF)** every minute:

1.  **Bending (`Bending`)**: Evaluates the strength-weighted section modulus against design moments.
2.  **Buckling (`Buckling`)**: Assesses stability using the Euler critical load and relative slenderness ($\lambda_{fi}$) of the heated residual section.
3.  **Shear (`Shear`)**: Checks local resistance at supports as the cross-section area is reduced.

### Results & Statistics
*   **Probability of Failure ($P_f$)**: Calculated as the failure ratio across $N$ iterations (typically 10,000 to 100,000).
*   **Reliability Index ($\beta$)**: Derived as $\beta = -\Phi^{-1}(P_f)$.
*   **Exact Confidence Intervals**: Uses the **Clopper-Pearson** method (Exact Binomial) to ensure statistical validity for research publication.
*   **Sensitivity Analysis**: Ranks the impact of variables using **Spearman Rank Correlation**.

---

## 📊 Visualizations
The system automatically generates high-impact graphics for thesis inclusion:
*   **Reliability Heatmaps**: Cross-comparison of species, treatments, and fire scenarios.
*   **Exact Stats Plots**: Probabilities of failure with 95% Confidence Interval error bars.
*   **Failure Distributions**: Stacked analysis showing the dominant cause of collapse for each scenario.

---

## 💻 Technical Setup

### Prerequisites
*   Python 3.10+
*   Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`

### Execution
Run the full scenario matrix (Standard & Parametric fires):
```bash
python3 mcs_simulation.py
```

### Outputs
*   `simulation_results.csv`: Comprehensive raw data.
*   `*.png`: High-resolution (300 DPI) plots for documentation.

---
*Developed for the research of Fire Reliability of Timber Structural Elements.*
