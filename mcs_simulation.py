"""
Monte Carlo Simulation Engine for Timber Roof Truss Fire Reliability
=====================================================================
Implements the probabilistic framework from the Research Methodology
(Chapters 3 & 4) for Anogeissus leiocarpa (W) and Erythrophleum suaveolens (R).

This module is the pure simulation engine — no UI code, no matplotlib.
All output is via returned data structures consumed by app.py (Streamlit).
"""

import numpy as np
import scipy.stats as stats
import pandas as pd

# =================================================================
# 1. PARAMETERS & DATA SETUP
# =================================================================

SPECIES_DATA = {
    'W': {  # Anogeissus leiocarpa (White Wood)
        'name': 'Anogeissus leiocarpa',
        'rho': {'dist': 'normal', 'mean': 809.0, 'std': 72.0},       # kg/m³
        'fm':  {'dist': 'lognormal', 'mean': 22.7, 'std': 5.7},      # N/mm²
        'fc0': {'dist': 'lognormal', 'mean': 17.4, 'std': 4.4},      # N/mm²
        'fv':  {'dist': 'lognormal', 'mean': 2.35, 'std': 0.59},     # N/mm²
        'E':   {'dist': 'lognormal', 'mean': 4612.0, 'std': 1247.0}, # N/mm²
        'MC':  {'dist': 'normal', 'mean': 0.1525, 'std': 0.0425},
        'thermal': {'lambda': 0.13, 'cp': 1500},                     # W/mK, J/kgK
        'char_insulation': 12.0,
        # Note: b & h here are the experimental test specimen dimensions used
        # for experimental data validation ONLY — they are NOT varied by the user.
        'b': 75,   # mm — test specimen width (fixed)
        'h': 150,  # mm — test specimen depth (fixed)
        'weights': {'exp': 0.4, 'mikkola': 0.3, 'hietaniemi': 0.3}
    },
    'R': {  # Erythrophleum suaveolens (Red Wood)
        'name': 'Erythrophleum suaveolens',
        'rho': {'dist': 'normal', 'mean': 745.0, 'std': 129.0},
        'fm':  {'dist': 'lognormal', 'mean': 38.9, 'std': 9.3},      
        'fc0': {'dist': 'lognormal', 'mean': 27.1, 'std': 6.8},
        'fv':  {'dist': 'lognormal', 'mean': 4.03, 'std': 1.01},
        'E':   {'dist': 'normal',    'mean': 8935.0, 'std': 1591.0},
        'MC':  {'dist': 'normal', 'mean': 0.1958, 'std': 0.0915},
        'thermal': {'lambda': 0.16, 'cp': 1600},
        'char_insulation': 18.0,
        'b': 75,   # mm — test specimen width (fixed)
        'h': 150,  # mm — test specimen depth (fixed)
        'weights': {'exp': 0.4, 'mikkola': 0.3, 'hietaniemi': 0.3}
    }
}

FIRE_SCENARIOS = {
    'FTI': {
        'name': 'Standard ISO 834',
        'base_duration': 60,
        'O_factor': 1.0,
        'type': 'standard'
    },
    'FTII': {
        'name': 'Parametric Kitchen (Low Vent)',
        'base_duration': 43,
        'opening_factor': 0.028,
        'b_factor': 1160,
        'q_t_d': 158.9,
        'type': 'parametric'
    },
    'FTIII': {
        'name': 'Parametric Sitting Room (High Vent)',
        'base_duration': 45,
        'opening_factor': 0.137,
        'b_factor': 1160,
        'q_t_d': 204.0,
        'type': 'parametric'
    }
}

TRUSS_CONFIGS = {
    'Double-Howe': {
        'description': 'Verified 6m truss from Chapter 4 Deterministic Design',
        'span': 6000,
        'pitch': 18.4,
        'members': {
            'Top Chord': {
                'type': 'compression_bending',
                'L_unbraced_ratio': 0.25,      # 1500mm / 6000mm
                'k_axial': 6.22,
                'k_moment': 0.225,
                'b': 75, 'h': 150
            },
            'Bottom Chord': {
                'type': 'tension_bending',
                'length': 1500,
                'L_unbraced_ratio': 0.25,      # 1500mm / 6000mm
                'k_axial': 5.63,
                'k_moment': 0.05,
                'b': 75, 'h': 150
            },
            'Diagonal Web (Compression)': {
                'type': 'compression',
                'L_unbraced_ratio': 0.30,      # 1800mm / 6000mm
                'k_axial': 2.73,
                'k_moment': 0.02,
                'b': 75, 'h': 150
            },
            'Diagonal Web (Tension)': {
                'type': 'tension',
                'L_unbraced_ratio': 0.30,      # 1800mm / 6000mm
                'k_axial': 2.24,
                'k_moment': 0.0,
                'b': 75, 'h': 150
            },
            'Vertical Web': {
                'type': 'compression',
                'L_unbraced_ratio': 0.1667,    # 1000mm / 6000mm
                'k_axial': 1.85,
                'k_moment': 0.0,
                'b': 75, 'h': 150
            }
        }
    },
    'Mono-pitch': {
        'description': 'Single-slope truss extension (Derived Estimate)',
        'span': 6000,
        'pitch': 18.4,
        'members': {
            'Top Chord': {
                'type': 'compression_bending',
                'L_unbraced_ratio': 0.125,
                'k_axial': 4.5,
                'k_moment': 0.25,
                'b': 75, 'h': 150
            },
            'Vertical Web': {
                'type': 'compression',
                'L_unbraced_ratio': 0.15,
                'k_axial': 2.0,
                'k_moment': 0.0,
                'b': 75, 'h': 150
            }
        }
    }
}

EXPERIMENTAL_CHARRING_DATA = {
    'FTI': {
        'W': {'untreated': {'mean': 0.71, 'cov': 0.08}, 'treated': {'mean': 0.57, 'cov': 0.05}},
        'R': {'untreated': {'mean': 0.65, 'cov': 0.04}, 'treated': {'mean': 0.57, 'cov': 0.03}}
    },
    'FTII': {
        'W': {'untreated': {'mean': 0.85, 'cov': 0.12}, 'treated': {'mean': 0.73, 'cov': 0.08}},
        'R': {'untreated': {'mean': 0.67, 'cov': 0.06}, 'treated': {'mean': 0.52, 'cov': 0.04}}
    },
    'FTIII': {
        'W': {'untreated': {'mean': 1.00, 'cov': 0.15}, 'treated': {'mean': 0.86, 'cov': 0.10}},
        'R': {'untreated': {'mean': 0.79, 'cov': 0.07}, 'treated': {'mean': 0.60, 'cov': 0.04}}
    }
}

# =================================================================
# 2. FIRE TEMPERATURE MODELS
# =================================================================

def get_fire_temperature(t, scenario_key):
    """T_fire(t) from ISO 834 or parametric curves (EN 1991-1-2)."""
    fire = FIRE_SCENARIOS[scenario_key]
    if fire['type'] == 'standard':
        return 20 + 345 * np.log10(8 * t + 1)
    else:
        O = fire['opening_factor']
        b = fire['b_factor']
        q_t_d = fire['q_t_d']
        gamma = (O / 0.04) ** 2 / (b / 1160) ** 2

        t_max = max(0.0002 * q_t_d / O, 0.333)
        t_max_star = t_max * gamma
        t_star = (t / 60.0) * gamma

        def t_gas(ts):
            return 20 + 1325 * (
                1 - 0.324 * np.exp(-0.2 * ts)
                  - 0.204 * np.exp(-1.7 * ts)
                  - 0.472 * np.exp(-19 * ts)
            )

        if t_star <= t_max_star:
            return t_gas(t_star)
        else:
            T_max = t_gas(t_max_star)
            if t_max <= 0.5:
                rate = 625
            elif t_max < 2.0:
                rate = 250 * (3 - t_max)
            else:
                rate = 250
            return max(20, T_max - (rate / 60.0) * (t - t_max * 60))


# =================================================================
# 3. CHARRING RATE MODELS
# =================================================================

def mikkola_charring_rate(t, species_key, scenario_key, rho_sampled, MC_sampled):
    """Mikkola (1991) charring rate model per Thesis Methodology."""
    flux_map = {'FTI': 50000, 'FTII': 35000, 'FTIII': 80000}
    qe = flux_map.get(scenario_key, 50000)
    qL = 15000
    Tp, T0, Tv = 300, 20, 100
    Lv = 2_250_000
    Lvw = 2_260_000
    cw = 4200
    c0 = SPECIES_DATA[species_key]['thermal']['cp']
    denominator = rho_sampled * (
        c0 * (Tp - T0) + Lv + (cw - c0) * (Tv - T0) + Lvw * MC_sampled
    )
    beta_ms = (qe - qL) / denominator if denominator > 0 else 0
    return beta_ms * 1000 * 60


def hietaniemi_charring_rate(t, species_key, scenario_key, w_sampled, rho_sampled):
    """Hietaniemi (2005) analytical charring rate model per Thesis Methodology."""
    tau_map = {
        'FTI':   {'W': 12, 'R': 18},
        'FTII':  {'W': 15, 'R': 22},
        'FTIII': {'W': 8,  'R': 12}
    }
    tau = tau_map[scenario_key][species_key]
    base_o2 = {'FTI': 1.0, 'FTII': 0.85, 'FTIII': 1.15}
    f_o2 = base_o2[scenario_key] * (1 - 0.1 * t / 60.0)
    if t < 10:
        q_std = 30000 + 5000 * t
    elif t < 30:
        q_std = 80000
    else:
        q_std = 70000
    C, p, rho_0, A, B = 0.8, 1.2, 200, 0.5, 0.3
    q_kw = q_std / 1000.0
    numerator = f_o2 * C * q_kw / rho_sampled
    denominator = (p + rho_0) * (A + B * w_sampled)
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * np.exp(-t / tau) * 1000


def combined_charring_rate(t, species_key, scenario_key, rho_sampled, MC_sampled,
                           beta_exp_sampled, theta_model):
    """
    Effective charring rate = weighted average of experimental data and analytical
    models, scaled by model uncertainty factor θ_model.
    """
    beta_M = mikkola_charring_rate(t, species_key, scenario_key, rho_sampled, MC_sampled)
    beta_H = hietaniemi_charring_rate(t, species_key, scenario_key, MC_sampled, rho_sampled)
    w_exp, w_M, w_H = 0.4, 0.3, 0.3
    beta_weighted = w_exp * beta_exp_sampled + w_M * beta_M + w_H * beta_H
    return theta_model * beta_weighted


# =================================================================
# 4. CROSS-SECTION THERMAL & REDUCTION MODELS
# =================================================================

def calculate_internal_temperature(x, t, T_fire, T_init, species_key):
    """
    Temperature profile per Section 3.4.3.1 (Eqn. 3.64).
    x = distance from char front (mm), t = exposure time (min).
    """
    if t <= 0:
        return T_init
    alpha_map = {'W': 1.07e-7, 'R': 1.34e-7}  # m²/s
    alpha = alpha_map[species_key]
    t_seconds = t * 60
    x_m = x / 1000.0
    T = T_init + (T_fire - T_init) * np.exp(-x_m / np.sqrt(4 * alpha * t_seconds))
    return min(T, 300)


def calculate_kmod_fi(temp):
    """
    Strength reduction factor k_mod,fi per Eqn. 3.65 (EN 1995-1-2).
    1 - 0.005*(T - 20) for 20 < T ≤ 300°C.
    """
    if temp <= 20:
        return 1.0
    elif temp < 300:
        return 1.0 - 0.005 * (temp - 20)
    else:
        return 0.0


def calculate_kE_fi(temp):
    """Stiffness reduction factor (MOE). Linear decay 1.0→0.0 over 20–400°C."""
    if temp <= 20:
        return 1.0
    elif temp < 400:
        return 1.0 - (temp - 20) / 380.0
    else:
        return 0.0


def get_effective_section_integrated(b0, h0, d_char, species_key, t):
    """
    Integrated effective section per Section 3.4.3.1 (3-sided exposure).
    Returns:
        A_eff        — Strength-weighted effective area  (mm²)
        W_eff        — Strength-weighted section modulus (mm³)
        I_eff        — Stiffness-weighted I (strong axis, mm⁴)
        y_bar_fi_mod — Strength NA distance from char front / hot face (mm)
        b_residual   — Residual width after charring (mm)
        h_residual   — Residual depth after charring (mm)
        I_eff_z      — Stiffness-weighted I (weak axis, mm⁴)
    """
    if d_char >= h0 or 2 * d_char >= b0:
        return 0, 0, 0, 0, 0, 0, 0

    n_layers = 20
    h_residual = h0 - d_char
    b_residual = b0 - 2 * d_char
    layer_thickness = h_residual / n_layers

    A_eff = 0.0
    sum_kE_dA = 0.0
    I_eff_stiffness = 0.0
    I_eff_strength = 0.0
    I_eff_stiffness_z = 0.0
    stat_moment_E = 0.0
    stat_moment_mod = 0.0
    layer_data = []

    for i in range(n_layers):
        x_dist = (i + 0.5) * layer_thickness
        T_layer = calculate_internal_temperature(x_dist, t, 300.0, 20.0, species_key)
        k_mod = calculate_kmod_fi(T_layer)
        k_E = calculate_kE_fi(T_layer)
        dA = b_residual * layer_thickness
        A_eff += dA * k_mod
        stat_moment_E += (dA * k_E) * x_dist
        sum_kE_dA += dA * k_E
        stat_moment_mod += (dA * k_mod) * x_dist
        layer_data.append((x_dist, dA, k_E, k_mod))

    if A_eff <= 0:
        return 0, 0, 0, 0, 0, 0, 0

    y_bar_fi_E = stat_moment_E / sum_kE_dA if sum_kE_dA > 0 else h_residual / 2
    y_bar_fi_mod = stat_moment_mod / A_eff

    for x_dist, dA, k_E, k_mod in layer_data:
        dist_to_na_E = x_dist - y_bar_fi_E
        I_layer_E = (b_residual * layer_thickness ** 3 / 12) + dA * dist_to_na_E ** 2
        I_eff_stiffness += I_layer_E * k_E
        I_eff_stiffness_z += (layer_thickness * b_residual ** 3 / 12) * k_E
        dist_to_na_mod = x_dist - y_bar_fi_mod
        I_layer_mod = (b_residual * layer_thickness ** 3 / 12) + dA * dist_to_na_mod ** 2
        I_eff_strength += I_layer_mod * k_mod

    y_top = h_residual - y_bar_fi_mod
    y_bottom = y_bar_fi_mod
    W_eff = I_eff_strength / max(y_top, y_bottom)

    return A_eff, W_eff, I_eff_stiffness, y_bar_fi_mod, b_residual, h_residual, I_eff_stiffness_z


# =================================================================
# 5. LIMIT STATE FUNCTIONS (FM1 – FM8)
# =================================================================

def bending_limit_state(M_Ed, fm, W_ef, theta_R):
    """FM4: Pure Bending — G4 = fm·Wef - M_Ed"""
    M_Rd_fi = fm * W_ef
    return theta_R * M_Rd_fi - M_Ed


def tension_limit_state(N_Ed, f_t0, A_ef, theta_R):
    """FM3 / FM7: Tension Rupture — G = ft0·Aef - N_Ed"""
    N_Rd_fi = f_t0 * A_ef
    return theta_R * N_Rd_fi - N_Ed


def combined_tension_bending_limit_state(N_Ed, M_Ed, ft0, fm, A_ef, W_ef, theta_R):
    """FM4a: Combined Tension + Bending — G = 1 - (σt/ft0 + σm/fm)"""
    if A_ef <= 0 or W_ef <= 0:
        return -1.0
    sigma_t = N_Ed / A_ef
    sigma_m = M_Ed / W_ef
    interaction = (sigma_t / ft0) + (sigma_m / fm)
    return theta_R * 1.0 - interaction


def lateral_torsional_buckling_limit_state(M_Ed, fm, E_mod, W_ef, L_ef,
                                           b_ef, h_ef, theta_R):
    """FM5: Lateral Torsional Buckling — G5 = k_crit·fm·Wef - M_Ed"""
    if W_ef <= 0 or b_ef <= 0 or h_ef <= 0:
        return -1.0
    sigma_m_crit = (0.78 * b_ef ** 2 * E_mod) / (h_ef * L_ef)
    lambda_rel_m = np.sqrt(fm / sigma_m_crit) if sigma_m_crit > 0 else 999
    if lambda_rel_m <= 0.75:
        k_crit = 1.0
    elif lambda_rel_m <= 1.4:
        k_crit = 1.56 - 0.75 * lambda_rel_m
    else:
        k_crit = 1.0 / (lambda_rel_m ** 2)
    k_crit = min(k_crit, 1.0)
    M_b_Rd_fi = k_crit * fm * W_ef
    return theta_R * M_b_Rd_fi - M_Ed


def buckling_limit_state(N_Ed, fc0, E_mod, I_ef, A_ef, theta_R, L_cr=2000):
    """
    FM1 / FM6: Compression Buckling per Eqns 3.3–3.10.
    G1 = k_c,y · fc0,d,fi · Aef - N_Ed
    Uses sampled fc0 and E for probabilistic consistency.
    """
    if A_ef <= 0:
        return -1.0
    i_ef = np.sqrt(I_ef / A_ef)
    if i_ef <= 0:
        return -1.0
    lambda_fi = (L_cr / (np.pi * i_ef)) * np.sqrt(fc0 / E_mod)
    beta_c = 0.2  # imperfection factor for solid timber
    k = 0.5 * (1 + beta_c * (lambda_fi - 0.3) + lambda_fi ** 2)
    discriminant = k ** 2 - lambda_fi ** 2
    if discriminant < 0:
        k_c_fi = 0.0
    else:
        k_c_fi = min(1.0 / (k + np.sqrt(discriminant)), 1.0)
    N_Rd_fi = k_c_fi * fc0 * A_ef
    return theta_R * N_Rd_fi - N_Ed


def shear_limit_state(V_Ed, fv, b_residual, h_residual, theta_R):
    """
    FM8: Shear per Eqns 3.52–3.56.
    Shear resistance V_Rd,fi = (2/3) * A_shear * fv.
    A_shear uses geometric b_residual × h_residual, NOT the k_mod-weighted A_ef.
    """
    A_shear = b_residual * h_residual
    V_Rd_fi = (2 / 3) * fv * A_shear
    return theta_R * V_Rd_fi - V_Ed


def combined_bending_compression_limit_state(N_Ed, M_Ed, fc0, fm, A_ef, W_ef,
                                              k_c_y, theta_R):
    """FM2: Combined Bending + Compression (EN 1995-1-1 Eqn 6.23 / Eqn 3.14)."""
    if A_ef <= 0 or W_ef <= 0:
        return -1.0
    if k_c_y <= 1e-6:
        return -1.0
    sigma_c = N_Ed / A_ef
    sigma_m = M_Ed / W_ef
    interaction = (sigma_c / (k_c_y * fc0)) ** 2 + (sigma_m / fm)
    return theta_R * 1.0 - interaction


# =================================================================
# 6. SAMPLING HELPERS
# =================================================================

def sample_variable(params):
    """Sample a single random variable from its distribution."""
    d = params['dist']
    if d == 'normal':
        return np.random.normal(params['mean'], params['std'])
    elif d == 'lognormal':
        mean, std = params['mean'], params['std']
        sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
        mu = np.log(mean) - 0.5 * sigma ** 2
        return np.random.lognormal(mu, sigma)
    elif d == 'gumbel':
        std, mean = params['std'], params['mean']
        beta = std * np.sqrt(6) / np.pi
        mu = mean - np.euler_gamma * beta
        return np.random.gumbel(mu, beta)
    else:
        return params['mean']


def sample_uncertainty(mean=1.0, cov=0.1):
    """Sample θ from Lognormal(mean, cov)."""
    sigma = np.sqrt(np.log(1 + cov ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return np.random.lognormal(mu, sigma)


def _get_lognormal_params(mean, std):
    """Convert physical-space mean/std to lognormal μ_ln, σ_ln."""
    sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return mu, sigma


def _z_to_marginal(z, params):
    """Transform a standard normal sample z to the target marginal distribution."""
    u = stats.norm.cdf(z)  # z → uniform via Φ(z)
    d = params['dist']
    if d == 'normal':
        return stats.norm.ppf(u, loc=params['mean'], scale=params['std'])
    elif d == 'lognormal':
        mu_ln, sigma_ln = _get_lognormal_params(params['mean'], params['std'])
        return stats.lognorm.ppf(u, s=sigma_ln, scale=np.exp(mu_ln))
    else:
        return params['mean']


def _z_to_lognormal(z, mu_ln, sigma_ln):
    """Transform a standard normal sample z to a lognormal marginal."""
    u = stats.norm.cdf(z)
    return stats.lognorm.ppf(u, s=sigma_ln, scale=np.exp(mu_ln))


# --- Correlation matrix for [fm, fc0, E, rho, MC, beta_exp] ---
# Eqns 3.78-3.82 specify 5 direct correlations. Two implied transitive
# correlations are added to ensure positive-definiteness:
#   Corr(fm, rho)  = 0.8 × 0.6 = 0.48  (via fm↔fc0↔rho)
#   Corr(fc0, E)   = 0.8 × 0.7 = 0.56  (via fc0↔fm↔E)
#         fm    fc0    E     rho    MC    beta
_CORR_MATRIX = np.array([
    [1.0,  0.8,  0.7,  0.48,  0.0,  0.0],   # fm
    [0.8,  1.0,  0.56, 0.6,   0.0,  0.0],   # fc0
    [0.7,  0.56, 1.0,  0.0,   0.0,  0.0],   # E
    [0.48, 0.6,  0.0,  1.0,   0.0, -0.5],   # rho
    [0.0,  0.0,  0.0,  0.0,   1.0,  0.3],   # MC
    [0.0,  0.0,  0.0, -0.5,   0.3,  1.0],   # beta_exp
])
_CORR_CHOLESKY_L = np.linalg.cholesky(_CORR_MATRIX)


def sample_correlated(species, mu_ln_char, sigma_ln_char):
    """
    Sample correlated material properties via Gaussian Copula
    per Eqns 3.78–3.82 (JCSS Probabilistic Model Code).

    Correlation structure (in standard-normal space):
        Corr(fm,   fc0)      = 0.8   (Eqn 3.78)
        Corr(fm,   E)        = 0.7   (Eqn 3.79)
        Corr(fc0,  rho)      = 0.6   (Eqn 3.80)
        Corr(rho,  beta_exp) = -0.5  (Eqn 3.81)
        Corr(MC,   beta_exp) = 0.3   (Eqn 3.82)

    Variables: [fm, fc0, E, rho, MC, beta_exp]
    Variables not in the matrix (fv) are sampled independently.

    Returns:
        fm, fc0, E, rho, MC, beta_exp, fv  (all as floats)
    """
    # Generate independent standard normals → correlated via pre-computed Cholesky
    u = np.random.standard_normal(6)
    z = _CORR_CHOLESKY_L @ u  # correlated standard normals

    # Transform to marginals via inverse CDF
    fm       = _z_to_marginal(z[0], species['fm'])
    fc0      = _z_to_marginal(z[1], species['fc0'])
    E        = _z_to_marginal(z[2], species['E'])
    rho      = _z_to_marginal(z[3], species['rho'])
    MC       = _z_to_marginal(z[4], species['MC'])
    beta_exp = _z_to_lognormal(z[5], mu_ln_char, sigma_ln_char)

    # fv is uncorrelated (not in Eqns 3.78–3.82)
    fv = sample_variable(species['fv'])

    return fm, fc0, E, rho, MC, beta_exp, fv


def get_k_vals(p):
    """5th-percentile characteristic value for a distribution."""
    if p['dist'] == 'gumbel':
        beta = p['std'] * np.sqrt(6) / np.pi
        mu = p['mean'] - np.euler_gamma * beta
        return mu - beta * np.log(-np.log(0.05))
    elif p['dist'] == 'lognormal':
        sigma = np.sqrt(np.log(1 + (p['std'] / p['mean']) ** 2))
        mu = np.log(p['mean']) - 0.5 * sigma ** 2
        return np.exp(mu - 1.645 * sigma)
    else:  # normal
        return p['mean'] - 1.645 * p['std']


# =================================================================
# 7. MAIN SIMULATION LOOP
# =================================================================

def run_simulation(
    N=1000,
    species_key='W',
    scenario_key='FTI',
    treatment='untreated',
    truss_type='Double-Howe',
    b_override=None,
    h_override=None,
    rei_duration=None,
    sensitivity=True,
    track_convergence=False,
):
    """
    Monte Carlo Simulation Loop with Truss-Aware Structural Checking.

    Args:
        N             : Number of iterations
        species_key   : 'W' or 'R'
        scenario_key  : 'FTI', 'FTII', or 'FTIII'
        treatment     : 'untreated' or 'borax'
        truss_type    : 'Double-Howe' or 'Mono-pitch'
        b_override    : float — uniform b (mm) for ALL TRUSS_CONFIG members
        h_override    : float — uniform h (mm) for ALL TRUSS_CONFIG members
        rei_duration  : int — REI threshold in minutes (overrides fire duration)
        sensitivity   : bool — collect samples for Spearman sensitivity analysis
        track_convergence : bool — return running Pf estimates

    Returns:
        res_df        : DataFrame of per-iteration results
        samples_df    : DataFrame of sampled variables (if sensitivity=True)
        convergence   : list of (n, pf) tuples (if track_convergence=True)
    """
    fire = FIRE_SCENARIOS[scenario_key]
    species = SPECIES_DATA[species_key]
    truss = TRUSS_CONFIGS[truss_type]

    # Determine simulation duration
    duration = rei_duration if rei_duration is not None else fire['base_duration']


    # Pre-compute lognormal parameters for experimental charring rate
    cond = 'treated' if treatment == 'borax' else 'untreated'
    char_data = EXPERIMENTAL_CHARRING_DATA[scenario_key][species_key][cond]
    char_mean, char_cov = char_data['mean'], char_data['cov']
    v_char = (char_mean * char_cov) ** 2
    sigma_ln_char = np.sqrt(np.log(1 + v_char / char_mean ** 2))
    mu_ln_char = np.log(char_mean) - 0.5 * sigma_ln_char ** 2

    G_params = {'dist': 'normal', 'mean': 2500.0, 'std': 250.0}
    Q_params = {'dist': 'gumbel',  'mean': 1800.0, 'std': 270.0}

    results = []
    samples = []
    convergence = []
    failures_so_far = 0

    for i in range(N):
        # --- Sample Correlated Material Properties (Eqns 3.78–3.82) ---
        fm, fc0, E, rho, MC, beta_exp_sampled, fv = \
            sample_correlated(species, mu_ln_char, sigma_ln_char)
        ft0  = 0.6 * fm

        # --- Sample Model Uncertainties ---
        theta_model = sample_uncertainty(1.0, 0.10)
        theta_R     = sample_uncertainty(1.0, 0.10)
        theta_E     = sample_uncertainty(1.0, 0.05)

        # --- Sample Loads (Eurocode fire combination) ---
        G_load = np.random.normal(G_params['mean'], G_params['std'])
        Q_load = sample_variable(Q_params)
        E_d_fi_base = theta_E * (G_load + 0.2 * Q_load) / 1000.0  # kN/m

        current_samples = {
            'rho': rho, 'fm': fm, 'fc0': fc0, 'fv': fv, 'E': E, 'MC': MC, 'load': E_d_fi_base,
            'theta_mod': theta_model, 'theta_R': theta_R, 'beta_exp': beta_exp_sampled
        }

        mode_survived = {
            'FM1_y': True, 'FM1_z': True,
            'FM2': True,
            'FM3': True,
            'FM4': True, 'FM4a': True,
            'FM5': True,
            'FM6_y': True, 'FM6_z': True,
            'FM7': True,
            'FM8': True
        }
        failed = False
        time_of_failure = duration
        failure_modes_triggered = []
        char_depth = 0.0

        # --- Time-Stepping Loop ---
        for t in range(0, duration + 1):
            T_fire = get_fire_temperature(t, scenario_key)

            if t > 0:
                beta_eff = combined_charring_rate(
                    t, species_key, scenario_key, rho, MC, beta_exp_sampled, theta_model
                )
                char_depth += beta_eff


            # --- Member-by-Member Structural Check ---
            for member_name, props in truss['members'].items():
                # Apply user b/h overrides (only to TRUSS_CONFIG members)
                b_mem = b_override if b_override is not None else props['b']
                h_mem = h_override if h_override is not None else props['h']

                # Species W web width override (per deterministic design validation)
                if species_key == 'W' and 'Web' in member_name and b_override is None:
                    b_mem = 100

                # Simplified Reduced Cross-Section Method (Eqns 3.11-3.12, 3.19)
                # d_ef includes the 7mm zero-strength layer (Eqn 3.12)
                d_ef = char_depth + 7.0 if char_depth > 0 else 0.0
                h_ef = h_mem - d_ef            # Eqn 3.11 (depth, 1-sided charring)
                b_ef = b_mem - 2 * d_ef        # Eqn 3.11 (width, 2-sided charring)
                b_residual = b_mem - 2 * char_depth  # geometric residual (for FM8 shear)
                h_residual = h_mem - char_depth       # geometric residual (for FM8 shear)

                if h_ef <= 0 or b_ef <= 0:
                    A_ef = 0  # triggers burnout logic below
                else:
                    A_ef   = b_ef * h_ef               # Eqn 4.13
                    W_ef   = b_ef * h_ef**2 / 6        # Eqn 3.19
                    I_ef   = b_ef * h_ef**3 / 12       # strong-axis
                    I_ef_z = h_ef * b_ef**3 / 12       # weak-axis

                if A_ef <= 0:
                    # Map complete section loss to the actual failure modes of this member
                    modes_to_fail = ['FM8'] # Everything can fail in shear
                    
                    if 'tension' in props['type']:
                        modes_to_fail.append('FM7' if 'Web' in member_name else 'FM3')
                        if props['k_moment'] > 0:
                            modes_to_fail.extend(['FM4', 'FM4a'])
                        if 'bending' in props['type'] and props['k_moment'] > 0:
                            modes_to_fail.append('FM5')
                            
                    if 'compression_bending' in props['type']:
                        modes_to_fail.append('FM2')
                        
                    if 'compression' in props['type'] or 'compression_bending' in props['type']:
                        prefix = 'FM6' if 'Web' in member_name else 'FM1'
                        modes_to_fail.extend([f"{prefix}_y", f"{prefix}_z"])
                    
                    for m in modes_to_fail:
                        mode_survived[m] = False
                        if m not in failure_modes_triggered:
                            failure_modes_triggered.append(m)

                    if not failed:
                        failed = True
                        time_of_failure = t
                    continue # Skip structural checks if burned out

                # Internal forces from truss factors
                # 15% fire-induced stiffness loss reduction (Table 4.10)
                M_Ed = E_d_fi_base * props['k_moment'] * 1_000_000 * 0.85  # Nmm
                N_Ed = E_d_fi_base * props['k_axial'] * 1_000 * 0.85       # N

                # Fire eccentricity per Eqn 3.18
                # Shift from original NA (h_mem/2 from top) to new NA (h_ef/2 from top)
                e_fire   = abs(h_mem / 2.0 - h_ef / 2.0)
                M_Ed_tot = M_Ed + abs(N_Ed) * e_fire

                # Shear force per Eqn 4.126
                # V_web ≈ V_Ed,fi / 2 = 50% distribution to web members
                span_m   = truss['span'] / 1000.0
                V_support = E_d_fi_base * span_m / 2.0 * 1000.0   # N — max support reaction
                V_Ed = V_support * 0.50   # Eqn 4.126: 50% to web

                # ---- Tension-type members (FM3, FM4, FM4a, FM5, FM7) ----
                if 'tension' in props['type']:
                    mode_key = 'FM7' if 'Web' in member_name else 'FM3'
                    if mode_survived[mode_key] and tension_limit_state(N_Ed, ft0, A_ef, theta_R) <= 0:
                        mode_survived[mode_key] = False
                        if not failed: failed = True; time_of_failure = t
                        if mode_key not in failure_modes_triggered: failure_modes_triggered.append(mode_key)

                    if props['k_moment'] > 0:
                        if mode_survived['FM4a'] and combined_tension_bending_limit_state(
                                N_Ed, M_Ed_tot, ft0, fm, A_ef, W_ef, theta_R) <= 0:
                            mode_survived['FM4a'] = False
                            if not failed: failed = True; time_of_failure = t
                            if 'FM4a' not in failure_modes_triggered: failure_modes_triggered.append('FM4a')

                        if mode_survived['FM4'] and bending_limit_state(M_Ed_tot, fm, W_ef, theta_R) <= 0:
                            mode_survived['FM4'] = False
                            if not failed: failed = True; time_of_failure = t
                            if 'FM4' not in failure_modes_triggered: failure_modes_triggered.append('FM4')

                    if 'bending' in props['type'] and props['k_moment'] > 0:
                        mem_length = props.get('length', truss['span'])
                        L_ef = props['L_unbraced_ratio'] * mem_length
                        if mode_survived['FM5'] and lateral_torsional_buckling_limit_state(
                                M_Ed_tot, fm, E, W_ef, L_ef,
                                b_residual, h_residual, theta_R) <= 0:
                            mode_survived['FM5'] = False
                            if not failed: failed = True; time_of_failure = t
                            if 'FM5' not in failure_modes_triggered: failure_modes_triggered.append('FM5')

                # ---- Top Chord: Combined Bending + Compression (FM2) ----
                if 'compression_bending' in props['type']:
                    L_cr_y = props['L_unbraced_ratio'] * truss['span']
                    i_ef_y = np.sqrt(I_ef / A_ef) if A_ef > 0 else 1.0
                    lambda_fi_y = (L_cr_y / (np.pi * i_ef_y)) * np.sqrt(fc0 / E)
                    k_y    = 0.5 * (1 + 0.2 * (lambda_fi_y - 0.3) + lambda_fi_y ** 2)
                    disc   = max(0, k_y ** 2 - lambda_fi_y ** 2)
                    k_c_y  = min(1.0 / (k_y + np.sqrt(disc)), 1.0) if A_ef > 0 else 0.0
                    if mode_survived['FM2'] and combined_bending_compression_limit_state(
                            N_Ed, M_Ed_tot, fc0, fm, A_ef, W_ef, k_c_y, theta_R) <= 0:
                        mode_survived['FM2'] = False
                        if not failed: failed = True; time_of_failure = t
                        if 'FM2' not in failure_modes_triggered: failure_modes_triggered.append('FM2')

                # ---- Compression-type members: Buckling (FM1, FM6) ----
                if 'compression' in props['type'] or 'compression_bending' in props['type']: # Check both for FM1/6!
                    L_cr_y = props['L_unbraced_ratio'] * truss['span']
                    mode_prefix = 'FM6' if 'Web' in member_name else 'FM1'

                    # Strong-axis buckling
                    mode_key_y = f"{mode_prefix}_y"
                    if mode_survived[mode_key_y] and buckling_limit_state(
                            N_Ed, fc0, E, I_ef, A_ef, theta_R, L_cr=L_cr_y) <= 0:
                        mode_survived[mode_key_y] = False
                        if not failed: failed = True; time_of_failure = t
                        if mode_key_y not in failure_modes_triggered: failure_modes_triggered.append(mode_key_y)
                        
                    # Weak-axis buckling
                    L_cr_z = 0.5 * L_cr_y
                    mode_key_z = f"{mode_prefix}_z"
                    if mode_survived[mode_key_z] and buckling_limit_state(
                            N_Ed, fc0, E, I_ef_z, A_ef, theta_R, L_cr=L_cr_z) <= 0:
                        mode_survived[mode_key_z] = False
                        if not failed: failed = True; time_of_failure = t
                        if mode_key_z not in failure_modes_triggered: failure_modes_triggered.append(mode_key_z)

                # ---- All members: Shear (FM8) ----
                # Uses effective area (b_ef × h_ef) per Section 4.15.4
                if mode_survived['FM8'] and shear_limit_state(V_Ed, fv, b_ef, h_ef, theta_R) <= 0:
                    mode_survived['FM8'] = False
                    if not failed: failed = True; time_of_failure = t
                    if 'FM8' not in failure_modes_triggered: failure_modes_triggered.append('FM8')

            # Only break if ALL modes have failed
            if not any(mode_survived.values()):
                break

        results.append({
            'failed': failed,
            'time': time_of_failure,
            'modes_triggered': ','.join(failure_modes_triggered) if failed else 'none',
            'truss': truss_type
        })

        if sensitivity:
            samples.append({**current_samples, 'failed': 1 if failed else 0})

        if track_convergence:
            if failed:
                failures_so_far += 1
            pf_running = failures_so_far / (i + 1)
            convergence.append((i + 1, pf_running))

    res_df = pd.DataFrame(results)
    samples_df = pd.DataFrame(samples) if sensitivity else pd.DataFrame()
    return res_df, samples_df, convergence


# =================================================================
# 8. RESULTS ANALYSIS
# =================================================================

def calculate_exact_confidence_intervals(k, n, confidence=0.95):
    """Clopper-Pearson exact confidence intervals for Pf (Eqn 3.82)."""
    from scipy.stats import beta
    alpha = 1 - confidence
    pf_lower = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    pf_upper = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    beta_val   = -stats.norm.ppf(k / n) if k > 0 else 5.0
    beta_lower = -stats.norm.ppf(pf_upper) if pf_upper > 0 and pf_upper < 1 else -5.0
    beta_upper = -stats.norm.ppf(pf_lower) if pf_lower > 0 else 5.5
    return (pf_lower, pf_upper), (beta_lower, beta_upper)


def analyze_results(df, scenario_name, species_name, treatment):
    """
    Compute Pf, β, 95% CIs, individual Pf_k, individual β_k, and system metrics.
    Returns a dictionary strictly aligned with Chapter 3 methodology.
    """
    total = len(df)
    failures = df[df['failed']]
    num_failures = len(failures)
    
    # =========================================================
    # NEW LOGIC: strictly adhering to Equations 3.83 to 3.86
    # =========================================================
    
    # Helper: Count total iterations where specific limit states were triggered
    def count_mode_iterations(*mode_names):
        count = 0
        for modes_str in df['modes_triggered']:
            if pd.isna(modes_str) or modes_str == 'none':
                continue
            triggered = [m.strip() for m in modes_str.split(',')]
            # If any of the requested modes are in the triggered list, count the iteration once
            if any(m in triggered for m in mode_names):
                count += 1
        return count

    # Step 3: Compute individual probabilities (Eqn. 3.83: Pf,k = failures[k] / Nsim)
    # Using the 8 distinct limit states defined in Table 3.2
    failures_k = {
        'FM1': count_mode_iterations('FM1_y', 'FM1_z'),
        'FM2': count_mode_iterations('FM2'),
        'FM3': count_mode_iterations('FM3'),
        'FM4': count_mode_iterations('FM4', 'FM4a'), # Combines pure bending and comb. tension/bending
        'FM5': count_mode_iterations('FM5'),
        'FM6': count_mode_iterations('FM6_y', 'FM6_z'),
        'FM7': count_mode_iterations('FM7'),
        'FM8': count_mode_iterations('FM8')
    }
    
    pf_k = {k: v / total for k, v in failures_k.items()}

    # Helper: Calculate Reliability Index
    def calc_beta(pf_val):
        if pf_val <= 0: return 5.0 # Max cap for 0 failures
        if pf_val >= 1: return float('-inf') # Min cap for 100% failures
        return -stats.norm.ppf(pf_val)

    # Step 5: Compute individual reliability indices (Eqn. 3.85: βk = -Φ^-1(Pf,k))
    beta_k = {k: calc_beta(v) for k, v in pf_k.items()}

    # Step 4: System probability of failure (series system) (Eqn. 3.84)
    # Pf,system = 1 - product[1 to 8] (1 - Pf,k)
    survival_prob = 1.0
    for p in pf_k.values():
        survival_prob *= (1.0 - p)
    pf_system = 1.0 - survival_prob

    # Step 5: System reliability index (Eqn. 3.86: βsystem = -Φ^-1(Pf,system))
    beta_system = calc_beta(pf_system)


    # =========================================================
    # LEGACY LOGIC: Preserved for UI and Dashboard Plotting
    # =========================================================
    mode_counts = {}
    for modes_str in failures['modes_triggered']:
        if pd.isna(modes_str) or modes_str == 'none':
            continue
        for mode in modes_str.split(','):
            mode = mode.strip()
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

    def pct(*mode_names):
        """Sum conditional percentages for one or more mode name variants."""
        total_count = sum(mode_counts.get(m, 0) for m in mode_names)
        return (total_count / num_failures * 100) if num_failures > 0 else 0.0

    return {
        'Scenario':              scenario_name,
        'Species':               species_name,
        'Treatment':             treatment,
        'N':                     total,
        'Failures':              num_failures,
        
        # --- EQUATIONS 3.84 & 3.86 (True Series System Metrics) ---
        'Pf_System':             pf_system,
        'Beta_System':           beta_system,
        
        # --- EQUATIONS 3.83 & 3.85 (Individual Mode Metrics) ---
        'Pf_FM1': pf_k['FM1'], 'Beta_FM1': beta_k['FM1'],
        'Pf_FM2': pf_k['FM2'], 'Beta_FM2': beta_k['FM2'],
        'Pf_FM3': pf_k['FM3'], 'Beta_FM3': beta_k['FM3'],
        'Pf_FM4': pf_k['FM4'], 'Beta_FM4': beta_k['FM4'],
        'Pf_FM5': pf_k['FM5'], 'Beta_FM5': beta_k['FM5'],
        'Pf_FM6': pf_k['FM6'], 'Beta_FM6': beta_k['FM6'],
        'Pf_FM7': pf_k['FM7'], 'Beta_FM7': beta_k['FM7'],
        'Pf_FM8': pf_k['FM8'], 'Beta_FM8': beta_k['FM8'],
        
        # --- UI COMPATIBILITY (Conditional % of Failed Runs) ---
        'FM1_Buckling%':         pct('FM1_y', 'FM1_z'),
        'FM1_Buckling_y%':       pct('FM1_y'),
        'FM1_Buckling_z%':       pct('FM1_z'),
        'FM2_CombBendComp%':     pct('FM2'),
        'FM3_Tension%':          pct('FM3'),
        'FM4_Bending%':          pct('FM4', 'FM4a'),
        'FM4a_CombTenBend%':     pct('FM4a'),
        'FM5_LTB%':              pct('FM5'),
        'FM6_WBuckling%':        pct('FM6_y', 'FM6_z'),
        'FM6_WBuckling_y%':      pct('FM6_y'),
        'FM6_WBuckling_z%':      pct('FM6_z'),
        'FM7_WTension%':         pct('FM7'),
        'FM8_Shear%':            pct('FM8'),
    }


def compute_sensitivity(samples_df):
    """
    Sensitivity analysis per Eqn 3.87.
    Spearman rank correlation was utilized as a computationally efficient
    proxy to identify critical parameters instead of the Sobol' indices
    proposed in the methodology (Section 3.15.3).
    Returns a Series sorted by |correlation|.
    """
    df = samples_df.loc[:, samples_df.std() > 0]
    if 'failed' not in df.columns or df['failed'].nunique() < 2:
        return pd.Series(dtype=float)
    corr = df.corr(method='spearman')['failed'].drop('failed')
    return corr.sort_values(key=abs, ascending=False)


# =================================================================
# 9. PARAMETRIC SWEEP
# =================================================================

def run_parametric_sweep(
    N=1000,
    species_key='W',
    scenario_key='FTI',
    treatment='untreated',
    truss_type='Double-Howe',
    rei_duration=30,
    b_values=None,
    h_values=None,
    progress_callback=None,
):
    """
    Run a full b × h parametric sweep for a single fire/species/treatment configuration.

    Args:
        b_values : list of b values (mm) — default [50, 75, 100]
        h_values : list of h values (mm) — default [100, 125, ..., 300]
        progress_callback : callable(current, total) for progress reporting

    Returns:
        sweep_df : DataFrame with columns b, h, Pf, Beta, and all analysis fields
    """
    if b_values is None:
        b_values = list(range(50, 101, 25))
    if h_values is None:
        h_values = list(range(100, 301, 25))

    combinations = [(b, h) for b in b_values for h in h_values]
    total = len(combinations)
    all_rows = []

    for idx, (b, h) in enumerate(combinations):
        res_df, _, _ = run_simulation(
            N=N,
            species_key=species_key,
            scenario_key=scenario_key,
            treatment=treatment,
            truss_type=truss_type,
            b_override=b,
            h_override=h,
            rei_duration=rei_duration,
            sensitivity=False,
            track_convergence=False,
        )
        scenario_name = f"{FIRE_SCENARIOS[scenario_key]['name']} (REI {rei_duration})"
        row = analyze_results(res_df, scenario_name,
                              SPECIES_DATA[species_key]['name'], treatment)
        row['b'] = b
        row['h'] = h
        all_rows.append(row)

        if progress_callback:
            progress_callback(idx + 1, total)

    return pd.DataFrame(all_rows)
