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
        'fm':  {'dist': 'lognormal', 'mean': 22.7, 'std': 5.7},      # N/mm² (Table 4.1)
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
        'fm':  {'dist': 'lognormal', 'mean': 38.9, 'std': 9.3},      # (Table 4.1)
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
                'L_unbraced_ratio': 0.125,
                'k_axial': 6.22,
                'k_moment': 0.225,
                'b': 75, 'h': 150
            },
            'Bottom Chord': {
                'type': 'tension_bending',
                'length': 1500,
                'L_unbraced_ratio': 0.5,
                'k_axial': 5.63,
                'k_moment': 0.05,
                'b': 75, 'h': 150
            },
            'Diagonal Web (Compression)': {
                'type': 'compression',
                'L_unbraced_ratio': 0.15,
                'k_axial': 2.73,
                'k_moment': 0.02,
                'b': 75, 'h': 150
            },
            'Diagonal Web (Tension)': {
                'type': 'tension',
                'L_unbraced_ratio': 0.15,
                'k_axial': 2.24,
                'k_moment': 0.0,
                'b': 75, 'h': 150
            },
            'Vertical Web': {
                'type': 'compression',
                'L_unbraced_ratio': 0.083,
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


def get_effective_section_integrated(b0, h0, d_char, species_key, t, T_fire):
    """
    Integrated effective section per Section 3.4.3.1 (3-sided exposure).
    Returns:
        A_eff        — Strength-weighted effective area  (mm²)
        W_eff        — Strength-weighted section modulus (mm³)
        I_eff        — Stiffness-weighted I (strong axis, mm⁴)
        y_bar_fi_mod — Strength NA distance from residual bottom (mm)
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


def lateral_torsional_buckling_limit_state(M_Ed, fm, E_05_fi, W_ef, L_ef,
                                           b_ef, h_ef, theta_R):
    """FM5: Lateral Torsional Buckling — G5 = k_crit·fm·Wef - M_Ed"""
    if W_ef <= 0 or b_ef <= 0 or h_ef <= 0:
        return -1.0
    sigma_m_crit = (0.78 * b_ef ** 2 * E_05_fi) / (h_ef * L_ef)
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


def buckling_limit_state(N_Ed, f_c0_k, E_05_fi, I_ef, A_ef, theta_R, L_cr=2000):
    """
    FM1 / FM6: Compression Buckling per Eqns 3.3–3.10.
    G1 = k_c,y · fc0,d,fi · Aef - N_Ed
    """
    if A_ef <= 0:
        return -1.0
    i_ef = np.sqrt(I_ef / A_ef)
    if i_ef <= 0:
        return -1.0
    lambda_fi = (L_cr / (np.pi * i_ef)) * np.sqrt(f_c0_k / E_05_fi)
    beta_c = 0.2  # imperfection factor for solid timber
    k = 0.5 * (1 + beta_c * (lambda_fi - 0.3) + lambda_fi ** 2)
    discriminant = k ** 2 - lambda_fi ** 2
    if discriminant < 0:
        k_c_fi = 0.0
    else:
        k_c_fi = min(1.0 / (k + np.sqrt(discriminant)), 1.0)
    N_Rd_fi = k_c_fi * f_c0_k * A_ef
    return theta_R * N_Rd_fi - N_Ed


def shear_limit_state(V_Ed, fv, b_residual, h_residual, theta_R):
    """
    FM8: Shear per Eqns 3.52–3.56.
    Shear area = (2/3) × geometric residual section (EN 1995-1-2).
    Note: A_shear uses geometric b_residual × h_residual, NOT the k_mod-weighted A_ef,
    because shear area is a geometric property of the residual cross-section.
    G8 = fv · A_shear - 1.5 · V_Ed
    """
    A_shear = (2 / 3) * b_residual * h_residual
    V_Rd_fi = fv * A_shear
    return theta_R * V_Rd_fi - 1.5 * V_Ed


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

    # Pre-compute characteristic values (deterministic, once per run)
    f_c0_k  = get_k_vals(species['fc0'])
    E_05_fi = get_k_vals(species['E'])

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
        # --- Sample Random Variables ---
        rho  = sample_variable(species['rho'])
        fm   = sample_variable(species['fm'])
        ft0  = 0.6 * fm
        fc0  = sample_variable(species['fc0'])
        fv   = sample_variable(species['fv'])
        E    = sample_variable(species['E'])
        MC   = sample_variable(species['MC'])

        # --- Sample Model Uncertainties ---
        theta_model = sample_uncertainty(1.0, 0.10)
        theta_R     = sample_uncertainty(1.0, 0.10)
        theta_E     = sample_uncertainty(1.0, 0.05)

        # --- Sample Experimental Charring Rate ---
        beta_exp_sampled = np.random.lognormal(mu_ln_char, sigma_ln_char)

        # --- Sample Loads (Eurocode fire combination) ---
        G_load = np.random.normal(G_params['mean'], G_params['std'])
        Q_load = sample_variable(Q_params)
        E_d_fi_base = theta_E * (G_load + 0.2 * Q_load) / 1000.0  # kN/m

        current_samples = {
            'rho': rho, 'fm': fm, 'fc0': fc0, 'fv': fv, 'E': E, 'MC': MC, 'load': E_d_fi_base,
            'theta_mod': theta_model, 'theta_R': theta_R, 'beta_exp': beta_exp_sampled
        }

        failed = False
        time_of_failure = duration
        failure_mode = 'none'
        char_depth = 0.0

        # --- Time-Stepping Loop ---
        for t in range(0, duration + 1):
            T_fire = get_fire_temperature(t, scenario_key)

            if t > 0:
                beta_eff = combined_charring_rate(
                    t, species_key, scenario_key, rho, MC, beta_exp_sampled, theta_model
                )
                char_depth += beta_eff

            d_ef = char_depth + 7.0 if char_depth > 0 else 0.0

            # --- Member-by-Member Structural Check ---
            for member_name, props in truss['members'].items():
                # Apply user b/h overrides (only to TRUSS_CONFIG members)
                b_mem = b_override if b_override is not None else props['b']
                h_mem = h_override if h_override is not None else props['h']

                # Species W web width override (per deterministic design validation)
                if species_key == 'W' and 'Web' in member_name and b_override is None:
                    b_mem = 100

                A_ef, W_ef, I_ef, y_bar_fi_mod, b_residual, h_residual, I_ef_z = \
                    get_effective_section_integrated(b_mem, h_mem, char_depth, species_key, t, T_fire)

                if A_ef <= 0:
                    failed = True
                    failure_mode = 'Burnout'
                    time_of_failure = t
                    break

                # Internal forces from truss factors
                M_Ed = E_d_fi_base * props['k_moment'] * 1_000_000  # Nmm
                N_Ed = E_d_fi_base * props['k_axial'] * 1_000       # N

                # Fire-induced eccentricity moment contribution
                y_NA_abs  = d_ef + y_bar_fi_mod
                e_fire    = abs(y_NA_abs - h_mem / 2.0)
                M_Ed_tot  = M_Ed + abs(N_Ed) * e_fire

                # Shear force per Eqn 3.53–3.54.
                # V_Ed,max = q_fi [kN/m] × L_span [m] / 2 → [N]
                # Distributed among members sharing the support node (Eqn 3.54).
                # k_dist factors: chords ≈ 0.4, webs ≈ 0.3 (3 members at support node).
                span_m   = truss['span'] / 1000.0
                V_support = E_d_fi_base * span_m / 2.0 * 1000.0   # N — max support reaction
                if 'Web' in member_name:
                    V_Ed = V_support * 0.30   # diagonal/vertical webs take ~30% of support shear
                else:
                    V_Ed = V_support * 0.40   # chords take ~40% of support shear each

                # ---- Tension-type members (FM3, FM4, FM4a, FM5, FM7) ----
                if 'tension' in props['type']:
                    if tension_limit_state(N_Ed, ft0, A_ef, theta_R) <= 0:
                        failed = True; time_of_failure = t; failure_mode = 'Web Tension' if 'Web' in member_name else 'Chord Tension'; break

                    if props['k_moment'] > 0:
                        if combined_tension_bending_limit_state(
                                N_Ed, M_Ed_tot, ft0, fm, A_ef, W_ef, theta_R) <= 0:
                            failed = True; time_of_failure = t
                            failure_mode = 'Comb. Tension+Bending'; break

                        if bending_limit_state(M_Ed_tot, fm, W_ef, theta_R) <= 0:
                            failed = True; time_of_failure = t; failure_mode = 'Bending'; break

                    if 'bending' in props['type'] and props['k_moment'] > 0:
                        mem_length = props.get('length', truss['span'])
                        L_ef = props['L_unbraced_ratio'] * mem_length
                        if lateral_torsional_buckling_limit_state(
                                M_Ed_tot, fm, E_05_fi, W_ef, L_ef,
                                b_residual, h_residual, theta_R) <= 0:
                            failed = True; time_of_failure = t; failure_mode = 'LTB'; break

                # ---- Top Chord: Combined Bending + Compression (FM2) ----
                if 'compression_bending' in props['type']:
                    L_cr_y = props['L_unbraced_ratio'] * truss['span']
                    i_ef_y = np.sqrt(I_ef / A_ef) if A_ef > 0 else 1.0
                    lambda_fi_y = (L_cr_y / (np.pi * i_ef_y)) * np.sqrt(f_c0_k / E_05_fi)
                    k_y    = 0.5 * (1 + 0.2 * (lambda_fi_y - 0.3) + lambda_fi_y ** 2)
                    disc   = max(0, k_y ** 2 - lambda_fi_y ** 2)
                    k_c_y  = min(1.0 / (k_y + np.sqrt(disc)), 1.0) if A_ef > 0 else 0.0
                    if combined_bending_compression_limit_state(
                            N_Ed, M_Ed_tot, fc0, fm, A_ef, W_ef, k_c_y, theta_R) <= 0:
                        failed = True; time_of_failure = t
                        failure_mode = 'Comb. Bending+Comp'; break

                # ---- Compression-type members: Buckling (FM1, FM6) ----
                if 'compression' in props['type']:
                    L_cr = props['L_unbraced_ratio'] * truss['span']
                    if buckling_limit_state(
                            N_Ed, f_c0_k, E_05_fi, I_ef, A_ef, theta_R, L_cr=L_cr) <= 0:
                        failed = True
                        time_of_failure = t          # ← BUG FIX: was missing
                        failure_mode = 'Web Buckling' if 'Web' in member_name else 'Chord Buckling'
                        break

                # ---- All members: Shear (FM8) ----
                if shear_limit_state(V_Ed, fv, b_residual, h_residual, theta_R) <= 0:
                    failed = True; time_of_failure = t; failure_mode = 'Shear'; break

            if failed:
                break

        results.append({
            'failed': failed,
            'time': time_of_failure,
            'mode': failure_mode,
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
    Compute Pf, β, 95% CIs, and failure mode breakdown.
    Returns a dictionary aligned with Table 3.2 / methodology output format.
    """
    total = len(df)
    failures = df[df['failed']]
    num_failures = len(failures)
    pf = num_failures / total
    (pf_l, pf_u), (beta_l, beta_u) = calculate_exact_confidence_intervals(num_failures, total)
    beta = -stats.norm.ppf(pf) if pf > 0 else 5.0

    modes = failures['mode'].value_counts() if num_failures > 0 else {}

    def pct(mode_name):
        return (modes.get(mode_name, 0) / num_failures * 100) if num_failures > 0 else 0.0

    return {
        'Scenario':              scenario_name,
        'Species':               species_name,
        'Treatment':             treatment,
        'N':                     total,
        'Failures':              num_failures,
        'Pf':                    pf,
        'Pf_Low':                pf_l,
        'Pf_High':               pf_u,
        'Beta':                  beta,
        'Beta_Low':              beta_l,
        'Beta_High':             beta_u,
        'FM1_Buckling%':         pct('Chord Buckling'),
        'FM2_CombBendComp%':     pct('Comb. Bending+Comp'),
        'FM3_Tension%':          pct('Chord Tension'),
        'FM4_Bending%':          pct('Bending'),
        'FM4a_CombTenBend%':     pct('Comb. Tension+Bending'),
        'FM5_LTB%':              pct('LTB'),
        'FM6_WBuckling%':        pct('Web Buckling'),
        'FM7_WTension%':         pct('Web Tension'),
        'FM8_Shear%':            pct('Shear'),
        'Burnout%':              pct('Burnout'),
    }


def compute_sensitivity(samples_df):
    """
    Spearman rank correlation of input variables vs. failure outcome.
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
