import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# =================================================================
# 1. PARAMETERS & DATA SETUP (Phase 1)
# =================================================================

SPECIES_DATA = {
    'W': {  # Anogeissus leiocarpa (White Wood)
        'name': 'Anogeissus leiocarpa',
        'rho': {'dist': 'normal', 'mean': 809.0, 'std': 72.0},  # kg/m^3
        # Eqn 3.68: f_m,k ~ Lognormal. mean=22.7 (Table 4.1), COV~25%
        'fm': {'dist': 'lognormal', 'mean': 22.7, 'std': 5.7},
        # Eqn 3.69: f_c,0,k ~ Lognormal. Characteristic = 0.45*fm,k = 0.45*22.7 = 10.2 N/mm^2 (Table 4.2)
        # mean/std chosen so Lognormal 5th percentile ≈ 10.2
        'fc0': {'dist': 'lognormal', 'mean': 17.4, 'std': 4.4},
        # Eqn 3.71: f_v,k ~ Lognormal. Characteristic = 0.067*fm,k = 0.067*22.7 = 1.52 N/mm^2 (Table 4.2)
        # mean=2.35, std=0.59 → Lognormal 5th percentile = 1.52 (corrected from 0.1*fm)
        'fv': {'dist': 'lognormal', 'mean': 2.35, 'std': 0.59},
        'E': {'dist': 'lognormal', 'mean': 4612.0, 'std': 1247.0},
        'MC': {'dist': 'normal', 'mean': 0.1525, 'std': 0.0425}, 
        'thermal': {'lambda': 0.13, 'cp': 1500}, # W/mK, J/kgK
        'char_insulation': 12.0,
        'b': 75, # mm Cross-Section Dimensions: Matched to the experimental fire test specimens
        'h': 150, # mm Essential for validating the charring rate models against experimental data.
        'weights': {'exp': 0.4, 'mikkola': 0.3, 'hietaniemi': 0.3} 
    },
    'R': {  # Erythrophleum suaveolens (Red Wood)
        'name': 'Erythrophleum suaveolens',
        'rho': {'dist': 'normal', 'mean': 745.0, 'std': 129.0},
        # Eqn 3.68: f_m,k ~ Lognormal. mean=38.9 (Table 4.1), COV~24%
        'fm': {'dist': 'lognormal', 'mean': 38.9, 'std': 9.3},
        # Eqn 3.69: f_c,0,k ~ Lognormal. Characteristic = 0.45*fm,k = 0.45*38.9 = 17.5 N/mm^2 (Table 4.3)
        # mean=27.1, std=6.8 → Lognormal 5th percentile = 17.5 (corrected from wrong 30.7)
        'fc0': {'dist': 'lognormal', 'mean': 27.1, 'std': 6.8},
        # Eqn 3.71: f_v,k ~ Lognormal. Characteristic = 0.067*fm,k = 0.067*38.9 = 2.61 N/mm^2 (Table 4.3)
        # mean=4.03, std=1.01 → Lognormal 5th percentile = 2.61 (corrected from 0.1*fm)
        'fv': {'dist': 'lognormal', 'mean': 4.03, 'std': 1.01},
        'E': {'dist': 'normal', 'mean': 8935.0, 'std': 1591.0},  
        'MC': {'dist': 'normal', 'mean': 0.1958, 'std': 0.0915}, 
        'thermal': {'lambda': 0.16, 'cp': 1600},
        'char_insulation': 18.0, 
        'b': 75, # mm Cross-Section Dimensions: Matched to the experimental fire test specimens
        'h': 150, # mm
        'weights': {'exp': 0.4, 'mikkola': 0.3, 'hietaniemi': 0.3} 
    }
}

FIRE_SCENARIOS = {
    'FTI': {
        'name': 'Standard ISO 834',
        'duration': 60,
        'O_factor': 1.0,
        'type': 'standard'
    },
    'FTII': {
        'name': 'Parametric Kitchen (Low Vent)',
        'duration': 43, # Expanded for parametric decay
        'opening_factor': 0.028,
        'b_factor': 1160, # Thermal inertia
        'q_t_d': 158.9,   # Fire load density (MJ/m2)
        'type': 'parametric'
    },
    'FTIII': {
        'name': 'Parametric Sitting Room (High Vent)',
        'duration': 45,
        'opening_factor': 0.137,
        'b_factor': 1160,
        'q_t_d': 204.0,   # Fire load density (MJ/m2)
        'type': 'parametric'
    }
}

TRUSS_CONFIGS = {
    'Double-Howe': {
        'description': 'Verified 6m truss from Chapter 4 Deterministic Design',
        'span': 6000,   # Span L = 6 m
        'pitch': 18.4,  # Height h = L/6 = 1.0m, resulting in an angle of arctan(1/3) approx 18.4 deg
        'members': {
            'Top Chord': {
                'type': 'compression_bending',
                # Lateral restraints provided at 750mm spacing to prevent weak-axis buckling.
                # Ratio relative to truss span: 750 / 6000 = 0.125
                'L_unbraced_ratio': 0.125, 
                # Axial Force at 30 min (-17.8 kN) divided by Design Fire Load (2.86 kN/m) = 6.22
                'k_axial': 6.22,  
                # Bending Moment (0.644 kNm) divided by Design Fire Load (2.86 kN/m) = 0.225
                'k_moment': 0.225, 
                # Recommended size 75x150 mm to satisfy R30 requirements
                'b': 75, 'h': 150 
            },
            'Bottom Chord': {
                'type': 'tension_bending',
                # Explicitly defined member panel length of 1500 mm (Table 4.5)
                'length': 1500, 
                # Lateral restraint at mid-span of the panel (750 mm) for LTB protection.
                # Ratio relative to member length: 750 / 1500 = 0.5
                'L_unbraced_ratio': 0.5, 
                # Axial Force at 30 min (+16.1 kN) divided by Design Fire Load = 5.63
                'k_axial': 5.63,
                # Self-weight moment is negligible (0.005 kNm), but 0.05 retained for eccentricity/tolerances
                'k_moment': 0.05, 
                # Size 75x150 mm required
                'b': 75, 'h': 150 
            },
            'Diagonal Web (Compression)': {
                'type': 'compression',
                # Member length 1800 mm with restraint at mid-length (900 mm).
                # Ratio relative to truss span: 900 / 6000 = 0.15
                'L_unbraced_ratio': 0.15,
                # Axial Force at 30 min (-7.8 kN) divided by Design Fire Load = 2.73
                'k_axial': 2.73,
                'k_moment': 0.02, # Minimal secondary moments
                # Increased from 50x100 to 75x150 mm to prevent Shear Failure (FM8)
                'b': 75, 'h': 150 
            },
            'Diagonal Web (Tension)': {
                'type': 'tension',
                'L_unbraced_ratio': 0.15, 
                # Axial Force at 30 min (+6.4 kN) divided by Design Fire Load = 2.24
                'k_axial': 2.24,
                'k_moment': 0.0,
                # Increased to 75x150 mm to prevent Shear Failure
                'b': 75, 'h': 150 
            },
            'Vertical Web': {
                'type': 'compression',
                # Member length 1000 mm with restraint at mid-height (500 mm).
                # Ratio relative to truss span: 500 / 6000 = 0.083
                'L_unbraced_ratio': 0.083,
                # Axial Force at 30 min (-5.3 kN) divided by Design Fire Load = 1.85
                'k_axial': 1.85,
                'k_moment': 0.0,
                # Sized 75x150 mm for consistency and shear capacity
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
                'L_unbraced_ratio': 0.125, # Maintain 750mm spacing
                'k_axial': 4.5, # Estimated lower axial load for mono-pitch
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

# Expanded to include COV and Treatment
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
# 2. CORE PHYSICS & MODELS (Phase 2)
# =================================================================

def get_fire_temperature(t, scenario_key):
    """
    T_fire(t) from ISO 834 or parametric curves.
    """
    fire = FIRE_SCENARIOS[scenario_key]
    if fire['type'] == 'standard':
        return 20 + 345 * np.log10(8 * t + 1)
    else:
        # Parametric Fire (EN 1991-1-2)
        O = fire['opening_factor']
        b = fire['b_factor']
        q_t_d = fire['q_t_d']
        gamma = (O / 0.04)**2 / (b / 1160)**2
        
        # t_max = max(0.0002 * q_t_d / O, t_lim)
        # t_lim = 20 mins = 0.333 hours
        t_max = max(0.0002 * q_t_d / O, 0.333) 
        t_max_star = t_max * gamma
        
        t_star = (t / 60.0) * gamma  # Normalized time (hours)
        
        def t_gas(ts):
            return 20 + 1325 * (1 - 0.324 * np.exp(-0.2*ts) - 0.204 * np.exp(-1.7*ts) - 0.472 * np.exp(-19*ts))
        
        if t_star <= t_max_star:
            return t_gas(t_star)
        else:
            # Exact Cooling Phase (EN 1991-1-2 Annex A)
            T_max = t_gas(t_max_star)
            # Cooling rates depend on fire duration (t_max)
            if t_max <= 0.5:
                rate = 625 # deg/h
            elif t_max < 2.0:
                rate = 250 * (3 - t_max) # deg/h
            else:
                rate = 250 # deg/h
            
            # rate is in deg/hour. Convert to deg/min for the decay relative to peak
            return max(20, T_max - (rate / 60.0) * (t - t_max * 60))

def mikkola_charring_rate(t, species_key, scenario_key, rho_sampled, MC_sampled):
    """
    Impliment Mikkola (1991) Model per Thesis Methodology.
    """
    # 1. Scenario-dependent net heat flux (qe - qL)
    flux_map = {'FTI': 50000, 'FTII': 35000, 'FTIII': 80000}
    qe = flux_map.get(scenario_key, 50000)
    qL = 15000  # Heat losses 
    
    # 2. Species-specific & Global Constants
    Tp, T0, Tv = 300, 20, 100
    Lv = 2250000    # J/kg 
    Lvw = 2260000   # J/kg 
    cw = 4200       # J/kgK 
    
    # Use species-specific cp from the table
    c0 = SPECIES_DATA[species_key]['thermal']['cp'] 
    
    # 3. Explicit Denominator calculation 
    # Energy to heat wood + Latent heat + Energy to heat/evap water
    denominator = rho_sampled * (
        c0 * (Tp - T0) + 
        Lv + 
        (cw - c0) * (Tv - T0) + 
        Lvw * MC_sampled
    )
    
    # 4. Result conversion to mm/min
    beta_ms = (qe - qL) / denominator if denominator > 0 else 0
    return beta_ms * 1000 * 60

def hietaniemi_charring_rate(t, species_key, scenario_key, w_sampled, rho_sampled):
    """
    Corrected Hietaniemi (2005) Analytical Model per Thesis Methodology.
    """
    # 1. Scenario and Species-dependent Char Insulation Constant (tau)
    tau_map = {
        'FTI':   {'W': 12, 'R': 18},
        'FTII':  {'W': 15, 'R': 22},
        'FTIII': {'W': 8,  'R': 12}
    }
    tau = tau_map[scenario_key][species_key]
    
    # 2. Oxygen Factor f(chi_O2) - CORRECTION APPLIED HERE
    # The source code includes a time-decay factor: (1 - 0.1 * t / 60)
    base_o2 = {'FTI': 1.0, 'FTII': 0.85, 'FTIII': 1.15}
    f_o2 = base_o2[scenario_key] * (1 - 0.1 * t / 60.0)
    
    # 3. Standard Heat Flux q_std(t) (W/m^2)
    if t < 10:
        q_std = 30000 + 5000 * t
    elif t < 30:
        q_std = 80000
    else:
        q_std = 70000
        
    # 4. Model Constants 
    C, p, rho_0, A, B = 0.8, 1.2, 200, 0.5, 0.3
    
    # 5. Analytical Equation
    q_kw = q_std / 1000.0  # Convert W/m^2 to kW/m^2
    numerator = f_o2 * C * q_kw / rho_sampled
    denominator = (p + rho_0) * (A + B * w_sampled)
    
    if denominator <= 0:
        return 0.0
    # Conversion to mm/min with exponential decay 
    beta_H = (numerator / denominator) * np.exp(-t / tau) * 1000
    
    return beta_H

def combined_charring_rate(t, species_key, scenario_key, rho_sampled, MC_sampled, beta_exp_sampled, theta_model):
    """
    Calculates the Effective Charring Rate per Methodology
    
    The effective rate is a weighted average of experimental data and analytical models,
    scaled by a model uncertainty factor.
    
    Args:
        beta_exp_sampled: The experimental rate sampled once at the start of the simulation.
        theta_model: Model uncertainty factor sampled from Lognormal(1.0, 0.1).
    """
    # 1. Analytical Mikkola Calculation (Time-dependent)
    beta_M = mikkola_charring_rate(t, species_key, scenario_key, rho_sampled, MC_sampled)
    
    # 2. Analytical Hietaniemi Calculation (Time-dependent)
    beta_H = hietaniemi_charring_rate(t, species_key, scenario_key, MC_sampled, rho_sampled)
    
    # 3. Apply Weighting Factors (0.4, 0.3, 0.3)
    # Experimental data (0.4) is prioritized, with analytical models (0.3 each) providing 
    # physics-based adjustments for time-dependent behavior.
    w_exp, w_M, w_H = 0.4, 0.3, 0.3
    
    beta_weighted = (w_exp * beta_exp_sampled) + (w_M * beta_M) + (w_H * beta_H)
    
    # 4. Apply Model Uncertainty Factor
    beta_eff = theta_model * beta_weighted
    
    return beta_eff

def calculate_internal_temperature(x, t, T_fire, T_init, species_key):
    """
    Corrected Temperature Profile per Section 3.4.3.1 of your Thesis.
    """
    if t <= 0: return T_init

    # 1. Use pre-calculated alpha from Table 3.9 / 4.10
    alpha_map = {'W': 1.07e-7, 'R': 1.34e-7}
    alpha = alpha_map[species_key]

    # 2. t must be in seconds for alpha (m^2/s)
    t_seconds = t * 60
    x_m = x / 1000.0 

    # 3. Exponential decay formula from your methodology
    # x is the distance from the heat source/char line
    T = T_init + (T_fire - T_init) * np.exp(-x_m / np.sqrt(4 * alpha * t_seconds))

    return min(T, 300) # Cap at pyrolysis temperature for stiffness calculations

def calculate_kmod_fi(temp):
    """
    Strength reduction factor per Methodology.
    Linearly decreases from 1.0 at 20C to 0.0 at 300C.
    """
    if temp <= 20:
        return 1.0
    elif temp < 300:
        return 1.0 - (temp - 20) / 280.0
    else:
        # Zero strength above 300C (Pyrolysis temperature)
        return 0.0

def calculate_kE_fi(temp):
    """
    Stiffness reduction factor (MOE) per Methodology.
    
    Linearly decreases from 1.0 at 20C to 0.0 at 400C.
    """
    if temp <= 20:
        return 1.0
    elif temp < 400:
        # Single linear decay in the methodology
        return 1.0 - (temp - 20) / 380.0
    else:
        # Zero stiffness above 400C
        return 0.0

def get_effective_section_integrated(b0, h0, d_char, species_key, t, T_fire):
    """
    Refined Integrated Effective Section per Methodology Section 3.4.3.1.
    Accounts for 3-sided exposure and shifting neutral axis.
    
    Returns:
        A_ef: Strength-weighted effective area (integrated with k_mod)
        W_ef: Strength-weighted section modulus (derived from integrated k_mod profile)
        I_ef: Stiffness-weighted moment of inertia (integrated with k_E for buckling)
    """
    if d_char >= h0 or 2*d_char >= b0:
        return 0, 0, 0, 0, 0, 0, 0
        
    n_layers = 20
    h_residual = h0 - d_char
    b_residual = b0 - 2 * d_char
    layer_thickness = h_residual / n_layers
    
    A_eff = 0
    sum_kE_dA = 0.0
    I_eff_stiffness = 0
    I_eff_strength = 0
    stat_moment_E = 0 
    stat_moment_mod = 0
    
    layer_data = []
    for i in range(n_layers):
        x_dist = (i + 0.5) * layer_thickness
        T_layer = calculate_internal_temperature(x_dist, t, 300.0, 20.0, species_key)
        
        k_mod = calculate_kmod_fi(T_layer)
        k_E = calculate_kE_fi(T_layer)
        
        dA = b_residual * layer_thickness
        A_eff += dA * k_mod
        
        stat_moment_E += (dA * k_E) * x_dist
        sum_kE_dA      += dA * k_E
        stat_moment_mod += (dA * k_mod) * x_dist
        layer_data.append((x_dist, dA, k_E, k_mod))

    if A_eff <= 0: return 0, 0, 0, 0, 0, 0, 0

    # Section properties depends on the property being analyzed
    # Stiffness centroid (for buckling/deflection)
    y_bar_fi_E = stat_moment_E / sum_kE_dA if sum_kE_dA > 0 else h_residual / 2
    
    # Strength centroid (for bending)
    y_bar_fi_mod = stat_moment_mod / A_eff if A_eff > 0 else h_residual / 2
    
    # We also need I_eff about the weak axis (z-z) for Lateral Torsional Buckling
    # And we need to explicitly return the residual b and h dimensions
    I_eff_stiffness_z = 0
    for x_dist, dA, k_E, k_mod in layer_data:
        # Stiffness Calculation strong axis (y-y)
        dist_to_na_E = x_dist - y_bar_fi_E
        I_layer_E = (b_residual * layer_thickness**3 / 12) + dA * dist_to_na_E**2
        I_eff_stiffness += I_layer_E * k_E
        
        # Stiffness Calculation weak axis (z-z)
        # Assuming the neutral axis for weak axis is exactly at mid-width
        # I_z of a layer = (layer_thickness * b_residual^3) / 12
        I_layer_E_z = (layer_thickness * b_residual**3 / 12) 
        I_eff_stiffness_z += I_layer_E_z * k_E
        
        # Strength Calculation
        dist_to_na_mod = x_dist - y_bar_fi_mod
        I_layer_mod = (b_residual * layer_thickness**3 / 12) + dA * dist_to_na_mod**2
        I_eff_strength += I_layer_mod * k_mod
        
    # Distance to extreme fibers from strength-weighted NA
    # y_bar_fi_mod is the distance from the residual bottom to the NA
    y_top = h_residual - y_bar_fi_mod
    y_bottom = y_bar_fi_mod
    W_eff = I_eff_strength / max(y_top, y_bottom)
    
    return A_eff, W_eff, I_eff_stiffness, y_bar_fi_mod, b_residual, h_residual, I_eff_stiffness_z

def bending_limit_state(M_Ed, fm, W_ef, theta_R):
    """
    G = theta_R * M_Rd,fi - M_Ed
    Integrated properties already account for k_mod.
    """
    gamma_M_fi = 1.0 
    M_Rd_fi = fm * W_ef / gamma_M_fi
    return theta_R * M_Rd_fi - M_Ed

def tension_limit_state(N_Ed, f_t0, A_ef, theta_R):
    """
    Failure Modes 3 & 7: Tension Rupture.
    """
    gamma_M_fi = 1.0
    N_Rd_fi = f_t0 * A_ef / gamma_M_fi
    return theta_R * N_Rd_fi - N_Ed

def combined_tension_bending_limit_state(N_Ed, M_Ed, ft0, fm, A_ef, W_ef, theta_R):
    """
    Failure Mode 4a: Combined Tension and Bending.
    """
    if A_ef <= 0 or W_ef <= 0:
        return -1.0
        
    sigma_t = N_Ed / A_ef
    sigma_m = M_Ed / W_ef
    
    # Interaction check
    interaction = (sigma_t / ft0) + (sigma_m / fm)
    
    return theta_R * 1.0 - interaction

def lateral_torsional_buckling_limit_state(M_Ed, fm, E_05_fi, W_ef, L_ef, b_ef, h_ef, theta_R):
    """
    Failure Mode 5: Lateral Torsional Buckling.
    L_ef relates to the distance between restraints (mm).
    """
    if W_ef <= 0 or b_ef <= 0 or h_ef <= 0:
        return -1.0
        
    # 1. Critical Bending Stress (sigma_m_crit)
    # Using the simplified formula for rectangular solid timber:
    # sigma_m_crit = 0.78 * b^2 * E_05 / (h * L_ef)
    sigma_m_crit = (0.78 * b_ef**2 * E_05_fi) / (h_ef * L_ef)
    
    # 2. Relative Slenderness for Bending (lambda_rel_m)
    lambda_rel_m = np.sqrt(fm / sigma_m_crit) if sigma_m_crit > 0 else 999
    
    # 3. Lateral Torsional Buckling Factor (k_crit)
    if lambda_rel_m <= 0.75:
        k_crit = 1.0
    elif lambda_rel_m <= 1.4:
        k_crit = 1.56 - 0.75 * lambda_rel_m
    else:
        k_crit = 1.0 / (lambda_rel_m**2)
        
    # Cap k_crit at 1.0
    k_crit = min(k_crit, 1.0)
        
    # 4. Limit State Evaluation
    gamma_M_fi = 1.0
    M_b_Rd_fi = k_crit * fm * W_ef / gamma_M_fi
    
    return theta_R * M_b_Rd_fi - M_Ed

def buckling_limit_state(N_Ed, f_c0_k, E_05_fi, I_ef, A_ef, theta_R, L_cr=2000):
    """
    Corrected Buckling Limit State per Methodology.
    
    Args:
        N_Ed: Design axial load in fire (N)
        f_c0_k: Characteristic compressive strength (N/mm^2)
        E_05_fi: 5th percentile Stiffness (N/mm^2)
        I_ef: Effective second moment of area (mm^4) (includes stiffness reduction kE)
        A_ef: Effective area (mm^2) (includes strength reduction k_mod)
        theta_R: Model uncertainty factor
        L_cr: Effective buckling length (mm)
    """
    # 1. Calculate Radius of Gyration (i_ef)
    if A_ef <= 0:
        return -1.0  # Failure if section is gone
    i_ef = np.sqrt(I_ef / A_ef)
    
    # 2. Calculate Relative Slenderness (lambda_fi)
    # L_cr is the effective buckling length
    lambda_fi = (L_cr / (np.pi * i_ef)) * np.sqrt(f_c0_k / E_05_fi)
    
    # 3. Calculate Instability Factor k
    # beta_c = 0.1 for the specific methodology/glulam
    beta_c = 0.2
    k = 0.5 * (1 + beta_c * (lambda_fi - 0.3) + lambda_fi**2)
    
    # 4. Calculate Buckling Reduction Factor k_c,fi
    # This reduces capacity based on slenderness
    denom = k + np.sqrt(k**2 - lambda_fi**2)
    if k**2 < lambda_fi**2:  # Catch math domain error if very slender
        k_c_fi = 0.0
    else:
        k_c_fi = 1.0 / denom
    k_c_fi = min(k_c_fi, 1.0)  # Cap at 1.0
    
    # 5. Calculate Resistance N_b,Rd,fi
    # Note: A_ef is already integrated with k_mod strength reduction
    N_Rd_fi = f_c0_k * A_ef * k_c_fi
    
    # 6. Limit State Function
    G = theta_R * N_Rd_fi - N_Ed
    return G

def shear_limit_state(V_Ed, fv, A_ef, theta_R):
    """
    Shear limit state for rectangular timber sections.
    A_ef already accounts for k_mod strength reduction.
    """
    V_Rd_fi = (fv * A_ef) / 1.5 
    
    return theta_R * V_Rd_fi - V_Ed

def combined_bending_compression_limit_state(N_Ed, M_Ed, fc0, fm, A_ef, W_ef, k_c_y, theta_R):
    """
    Failure Mode 2: Combined Bending and Compression (EN 1995-1-1 Eqn 6.23).
    Matches Eqn 3.14 and Eqn 4.116.
    
    Args:
        k_c_y: Instability factor (from Buckling check FM1)
    """
    if A_ef <= 0 or W_ef <= 0:
        return -1.0
        
    sigma_c = N_Ed / A_ef
    sigma_m = M_Ed / W_ef
    
    # CRITICAL FIX: Include k_c_y in the denominator
    # Ensure k_c_y is not 0 to avoid division by zero
    if k_c_y <= 1e-6: 
        return -1.0 # Failed by buckling already
        
    term_compression = (sigma_c / (k_c_y * fc0))**2
    term_bending = (sigma_m / fm)
    
    # Interaction check
    interaction = term_compression + term_bending
    
    return theta_R * 1.0 - interaction

def sample_variable(params):
    """
    Helper to sample from distributions.
    """
    if params['dist'] == 'normal':
        return np.random.normal(params['mean'], params['std'])
    elif params['dist'] == 'lognormal':
        # Scipy lognorm uses sigma (shape) and scale=exp(mu)
        # mean = exp(mu + sigma^2/2), var = [exp(sigma^2)-1]*mean^2
        mean = params['mean']
        std = params['std']
        sigma = np.sqrt(np.log(1 + (std/mean)**2))
        mu = np.log(mean) - 0.5 * sigma**2
        return np.random.lognormal(mu, sigma)
    elif params['dist'] == 'gumbel':
        # mean = mu + 0.577 * beta, var = pi^2 * beta^2 / 6
        std = params['std']
        mean = params['mean']
        beta = std * np.sqrt(6) / np.pi
        mu = mean - np.euler_gamma * beta
        return np.random.gumbel(mu, beta)
    else:
        return params['mean']

def sample_uncertainty(mean=1.0, cov=0.1):
    """
    Samples theta from a lognormal distribution.
    """
    sigma = np.sqrt(np.log(1 + cov**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return np.random.lognormal(mu, sigma)

# Compute Characteristic values for Buckling (once per species)
def get_k_vals(p):
    if p['dist'] == 'gumbel':
        beta = p['std'] * np.sqrt(6) / np.pi
        mu = p['mean'] - np.euler_gamma * beta
        return mu - beta * np.log(-np.log(0.05))
    elif p['dist'] == 'lognormal':
        sigma = np.sqrt(np.log(1 + (p['std']/p['mean'])**2))
        mu = np.log(p['mean']) - 0.5 * sigma**2
        return np.exp(mu - 1.645 * sigma)
    else: # normal
        return p['mean'] - 1.645 * p['std']

def run_simulation(N=1000, species_key='W', scenario_key='FTI', treatment='untreated', truss_type='Double-Howe', sensitivity=True):
    """
    Main Monte Carlo Simulation Loop with Truss-Aware Logic.
    Integrates Material, Fire, and Structural models per Methodology.
    """
    fire = FIRE_SCENARIOS[scenario_key]
    species = SPECIES_DATA[species_key]
    
    # LOAD TRUSS CONFIGURATION
    # This determines member dimensions (b, h) and load distribution factors (k)
    truss = TRUSS_CONFIGS[truss_type]
    
    results = []
    samples = []
    
    G_params = {'dist': 'normal', 'mean': 2500.0, 'std': 250.0}
    Q_params = {'dist': 'gumbel', 'mean': 1800.0, 'std': 270.0}
            
    f_c0_k = get_k_vals(species['fc0'])
    f_m_k = get_k_vals(species['fm'])
    # E_0,05 per Eqn 3.72 (5th percentile of MOE). No upward adjustment is specified in the methodology.
    E_05_fi = get_k_vals(species['E'])
    
    print(f"Running MCS: {N} iter | {species['name']} | {truss_type} | {treatment}...")
    print(f"  Characteristic: fc0,k={f_c0_k:.2f} N/mm² | E0,05={E_05_fi:.0f} N/mm²")
    # Pre-calculate Lognormal params for Charring Rate to save time inside loop
    # Map 'borax' to 'treated' for data lookup
    cond = 'treated' if treatment == 'borax' else 'untreated'
    
    char_data = EXPERIMENTAL_CHARRING_DATA[scenario_key][species_key][cond]
    char_mean = char_data['mean']
    char_cov = char_data['cov']
    
    # Convert arithmetic statistics to Lognormal underlying parameters
    v_char = (char_mean * char_cov)**2
    sigma_ln_char = np.sqrt(np.log(1 + v_char / char_mean**2))
    mu_ln_char = np.log(char_mean) - 0.5 * sigma_ln_char**2

    for _ in tqdm(range(N), disable=True): # Disable tqdm for cleaner logs if needed
        # 1. Sample Random Variables
        rho = sample_variable(species['rho'])
        fm = sample_variable(species['fm'])
        ft0 = 0.6 * fm # Derived directly from sampled bending strength
        fc0 = sample_variable(species['fc0'])
        fv = sample_variable(species['fv'])
        E = sample_variable(species['E'])
        MC = sample_variable(species['MC'])
        
        # 2. Sample Uncertainties
        theta_model = sample_uncertainty(1.0, 0.10)
        theta_R = sample_uncertainty(1.0, 0.10)
        theta_E = sample_uncertainty(1.0, 0.05)
        
        # 3. Sample Experimental Charring Rate (Lognormal)
        beta_exp_sampled = np.random.lognormal(mu_ln_char, sigma_ln_char)
        
        # 4. Sample Loads (Structural Analysis Logic)
        G_load = np.random.normal(G_params['mean'], G_params['std'])
        Q_load  = sample_variable(Q_params)
        # Fire load combination (Eurocode: G + psi_2,1 * Q) 
        # theta_E is model uncertainty for the load effect.
        # Combined load result converted to kN/m
        E_d_fi_base = theta_E * (G_load + 0.2 * Q_load) / 1000.0
        
        current_samples = {'rho': rho, 'fm': fm, 'fv': fv, 'E': E, 'MC': MC, 'load': E_d_fi_base}
        
        failed = False
        time_of_failure = fire['duration']
        failure_mode = 'none'
        char_depth = 0.0
        
        # TIME STEPPING LOOP
        for t in range(0, fire['duration'] + 1):
            T_fire = get_fire_temperature(t, scenario_key)
            
            # A. Update char depth (mm)
            if t > 0:
                beta_eff = combined_charring_rate(t, species_key, scenario_key, rho, MC, beta_exp_sampled, theta_model)
                char_depth += beta_eff 
            
            if char_depth > 0.0:
                d_ef = char_depth + 7.0
            else:
                d_ef = 0.0
            # B. TRUSS-AWARE STRUCTURAL CHECK 
            # Iterate through critical members defined in TRUSS_CONFIGS
            for member_name, props in truss['members'].items():
                
                # 1. Get Member Dimensions (Structural, not Experimental)
                # IMPORTANT: Using TRUSS dimensions, not SPECIES_DATA dimensions
                
                b_mem = props['b']
                h_mem = props['h']
                
                # 2. CRITICAL FIX: Species W Web Sizing
                # Override width to 100mm if Species is W AND it is a Web member
                # Using "in" catches 'Web' (Double-Howe) and 'Vertical Web' (Mono-pitch)
                if species_key == 'W' and 'Web' in member_name:
                    b_mem = 100 
                
                # 3. Calculate Section Properties with the correct b_mem
                A_ef, W_ef, I_ef, y_bar_fi_mod, b_residual, h_residual, I_ef_z = get_effective_section_integrated(b_mem, h_mem, char_depth, species_key, t, T_fire)
                
                # Check if member is completely charred
                if A_ef <= 0:
                    failed = True 
                    failure_mode = 'Burnout'
                    time_of_failure = t
                    break

                # 4. Calculate Internal Forces based on Truss Factors
                # M_Ed: kNm -> Nmm
                M_Ed = E_d_fi_base * props['k_moment'] * 1_000_000
                
                # N_Ed: kN -> N
                N_Ed = E_d_fi_base * props['k_axial'] * 1_000
                
                # Bending Moment with Fire Eccentricity (FM2)
                # y_NA_abs is distance of NA from the original bottom
                y_NA_abs = d_ef + y_bar_fi_mod
                e_fire = abs(y_NA_abs - h_mem / 2.0)
                M_Ed_tot = M_Ed + abs(N_Ed) * e_fire 
                
                # V_Ed: Calculate Global Max Shear (N)
                V_global = E_d_fi_base * (truss['span'] / 2.0)
                
                # DISTRIBUTE SHEAR per Source Eqn 4.126
                if 'Web' in member_name:
                    V_Ed = V_global * 0.5  # Webs take ~50% of support shear
                else:
                    V_Ed = V_global        # Conservative max for Chords
                
                A_shear = b_residual * h_residual  
                # 5. Limit State Evaluation based on Member Type
                
                # Tension Members -> Bottom Chord and Tension Webs (FM3, FM4, FM4a, FM5, FM7)
                if 'tension' in props['type']: 
                    
                    # 1. Pure Tension Rupture (FM3 / FM7)
                    if tension_limit_state(N_Ed, ft0, A_ef, theta_R) <= 0:
                        failed = True; time_of_failure = t; failure_mode = 'Tension'; break
                        
                    # 2. Combined Tension and Bending (FM4a)
                    # Only calculate if there's actually a bending moment defined
                    if props['k_moment'] > 0:
                        if combined_tension_bending_limit_state(N_Ed, M_Ed_tot, ft0, fm, A_ef, W_ef, theta_R) <= 0:
                            failed = True; time_of_failure = t; failure_mode = 'Comb. Tension+Bending'; break
                            
                    # 3. Pure Bending (FM4) - Secondary check requested previously
                    if props['k_moment'] > 0:
                        if bending_limit_state(M_Ed_tot, fm, W_ef, theta_R) <= 0:
                            failed = True; time_of_failure = t; failure_mode = 'Bending'; break
                            
                    # 4. Lateral Torsional Buckling - Bottom Chord Only (FM5)
                    if 'bending' in props['type'] and props['k_moment'] > 0:
                        # For Double-Howe, length is directly defined. Fallback to span if missing.
                        mem_length = props.get('length', truss['span']) 
                        L_ef = props['L_unbraced_ratio'] * mem_length
                        
                        if lateral_torsional_buckling_limit_state(M_Ed_tot, fm, E_05_fi, W_ef, L_ef, b_residual, h_residual, theta_R) <= 0:
                            failed = True; time_of_failure = t; failure_mode = 'LTB'; break
                        
                # Top Chord -> Combined Bending + Compression (FM2)
                if 'compression_bending' in props['type']:
                    # Calculate k_c_y (instability factor)
                    L_cr_y = props['L_unbraced_ratio'] * truss['span'] 
                    i_ef_y = np.sqrt(I_ef / A_ef) if A_ef > 0 else 1.0
                    lambda_fi_y = (L_cr_y / (np.pi * i_ef_y)) * np.sqrt(f_c0_k / E_05_fi)
                    k_y = 0.5 * (1 + 0.2 * (lambda_fi_y - 0.3) + lambda_fi_y**2)
                    k_c_y = min(1.0 / (k_y + np.sqrt(max(0, k_y**2 - lambda_fi_y**2))), 1.0) if A_ef > 0 else 0.0
                    
                    if combined_bending_compression_limit_state(N_Ed, M_Ed_tot, fc0, fm, A_ef, W_ef, k_c_y, theta_R) <= 0:
                        failed = True; time_of_failure = t; failure_mode = 'Comb. Bending+Comp'; break

                # Top Chord / Webs -> Buckling (FM1, FM6)
                if 'compression' in props['type']:
                    # Use member specific buckling length
                    L_cr = props['L_unbraced_ratio'] * truss['span'] 
                    
                    # Note: We pass buckling length L_cr to the limit state function
                    if buckling_limit_state(N_Ed, f_c0_k, E_05_fi, I_ef, A_ef, theta_R, L_cr=L_cr) <= 0:
                        failed = True; failure_mode = 'Buckling'; break
                
                # All Members -> Shear (FM8)
                # With V_Ed * 0.5, Species W (100mm) should now PASS this check
                if shear_limit_state(V_Ed, fv, A_ef, theta_R) <= 0:
                    failed = True; time_of_failure = t; failure_mode = 'Shear'; break

            # Once any member fails, record the actual time and exit the time loop
            if failed:
                break

        res = {'failed': failed, 'time': time_of_failure, 'mode': failure_mode, 'truss': truss_type}
        results.append(res)
        if sensitivity:
            samples.append({**current_samples, 'failed': 1 if failed else 0})
            
    res_df = pd.DataFrame(results)
    
    # Sensitivity Analysis Output
    if sensitivity and any(res_df['failed']):
        samp_df = pd.DataFrame(samples)
        # Drop columns with zero variance to avoid NaNs
        samp_df = samp_df.loc[:, samp_df.std() > 0] 
        if 'failed' in samp_df:
            corr = samp_df.corr(method='spearman')['failed'].drop('failed')
            print(f"\nSensitivity (Spearman) for {species['name']} [{truss_type}]:")
            print(corr.sort_values(key=abs, ascending=False).head(5)) # Top 5 factors
        
    return res_df

# =================================================================
# 4. OUTPUT & VISUALIZATION (Phase 4)
# =================================================================

def calculate_exact_confidence_intervals(k, n, confidence=0.95):
    """
    Clopper-Pearson Exact Confidence Intervals for Pf.
    """
    from scipy.stats import beta
    alpha = 1 - confidence
    pf_lower = beta.ppf(alpha/2, k, n - k + 1) if k > 0 else 0.0
    pf_upper = beta.ppf(1 - alpha/2, k + 1, n - k) if k < n else 1.0
    
    # Calculate Beta Index with CI directly from Pf bounds
    # beta = -norm.ppf(pf)
    beta_val = -stats.norm.ppf(k/n) if k > 0 else 5.0
    beta_lower = -stats.norm.ppf(pf_upper) # Lower Pf -> Higher Beta, so upper Pf -> lower Beta
    beta_upper = -stats.norm.ppf(pf_lower) if pf_lower > 0 else 5.5
    
    return (pf_lower, pf_upper), (beta_lower, beta_upper)

def analyze_results(df, scenario_name, species_name, treatment):
    total = len(df)
    failures = df[df['failed']]
    num_failures = len(failures)
    
    pf = num_failures / total
    (pf_l, pf_u), (beta_l, beta_u) = calculate_exact_confidence_intervals(num_failures, total)
    beta = -stats.norm.ppf(pf) if pf > 0 else 5.0
    
    print(f"\n--- Professional Summary for {species_name} ({treatment}) ---")
    print(f"Fire Scenario: {scenario_name}")
    print(f"N = {total:,} iterations")
    print(f"Pf: {pf:.5e} [95% CI: {pf_l:.2e}, {pf_u:.2e}]")
    print(f"Beta: {beta:.3f} [95% CI: {beta_l:.3f}, {beta_u:.3f}]")
    
    modes = failures['mode'].value_counts() if num_failures > 0 else {}
    return {
        'Scenario': scenario_name,
        'Species': species_name,
        'Treatment': treatment,
        'Pf': pf,
        'Pf_Low': pf_l,
        'Pf_High': pf_u,
        'Beta': beta,
        'Beta_Low': beta_l,
        'Beta_High': beta_u,
        'Bending%': (modes.get('Bending', 0) / num_failures * 100) if num_failures > 0 else 0,
        'CombBendingComp%': (modes.get('Comb. Bending+Comp', 0) / num_failures * 100) if num_failures > 0 else 0,
        'Tension%': (modes.get('Tension', 0) / num_failures * 100) if num_failures > 0 else 0,
        'CombTensionBending%': (modes.get('Comb. Tension+Bending', 0) / num_failures * 100) if num_failures > 0 else 0,
        'LTB%': (modes.get('LTB', 0) / num_failures * 100) if num_failures > 0 else 0,
        'Buckling%': (modes.get('Buckling', 0) / num_failures * 100) if num_failures > 0 else 0,
        'Shear%': (modes.get('Shear', 0) / num_failures * 100) if num_failures > 0 else 0
    }

def generate_professional_plots(summary_df):
    """
    High-impact visualizations for the thesis using pure Matplotlib.
    """
    plt.rcParams.update({'font.size': 10, 'figure.facecolor': 'white'})
    
    # 1. Reliability Index Heatmap (Pure Matplotlib approach)
    plt.figure(figsize=(10, 6))
    pivot = summary_df.pivot_table(index='Scenario', columns=['Species', 'Treatment'], values='Beta')
    data = pivot.values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap="YlGnBu")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Reliability Index (Beta)", rotation=-90, va="bottom")
    
    # Labels
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", 
                    color="white" if data[i, j] > pivot.values.max()*0.7 else "black")
            
    ax.set_title('Reliability Index Heatmap (Exact Stats)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('reliability_heatmap.png', dpi=300)
    plt.close()
    
    # 2. Probability of Failure with Exact CIs
    plt.figure(figsize=(12, 6))
    # Create descriptive labels
    labels = [f"{r['Scenario']}\n{r['Species']} ({r['Treatment']})" for _, r in summary_df.iterrows()]
    summary_df['TempLabels'] = labels
    df_sorted = summary_df.sort_values('Pf')
    
    plt.errorbar(range(len(df_sorted)), df_sorted['Pf'], 
                 yerr=[df_sorted['Pf'] - df_sorted['Pf_Low'], df_sorted['Pf_High'] - df_sorted['Pf']],
                 fmt='o', color='crimson', capsize=5, elinewidth=1.5, markeredgewidth=1.5, label='Pf (Exact 95% CI)')
    
    plt.yscale('log')
    plt.xticks(range(len(df_sorted)), df_sorted['TempLabels'], rotation=45, ha='right')
    plt.ylabel('Probability of Failure (Log Scale)')
    plt.title('Safety Margin Analysis: Pf with Exact Confidence Intervals', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig('pf_confidence_intervals.png', dpi=300)
    plt.close()

    # 3. Failure Mode Distribution (Stacked Bar)
    plt.figure(figsize=(10, 6))
    cols = ['Bending%', 'CombBendingComp%', 'Tension%', 'CombTensionBending%', 'LTB%', 'Buckling%', 'Shear%']
    available_cols = [c for c in cols if c in df_sorted.columns]
    plot_df = df_sorted[['TempLabels'] + available_cols].set_index('TempLabels')
    
    plot_df.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Failure Mode Distribution across Scenarios', fontsize=14)
    plt.ylabel('Relative Frequency (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('failure_modes_distribution.png', dpi=300)
    plt.close()

def run_all_scenarios(N=30): # Reduced to 30 for testing, change to 100000 for final
    all_res = []
    
    # 1. NEW LOOP: Iterate through Truss Types (Double-Howe, Mono-pitch)
    for truss_key in TRUSS_CONFIGS:
        
        for scenario in FIRE_SCENARIOS:
            for species in SPECIES_DATA:
                for treatment in ['untreated', 'borax']:
                    
                    # 2. Pass the 'truss_key' to the simulation
                    df = run_simulation(
                        N=N, 
                        species_key=species, 
                        scenario_key=scenario, 
                        treatment=treatment,
                        truss_type=truss_key, # <--- CRITICAL FIX
                        sensitivity=True
                    )
                    
                    # 3. Analyze results
                    # Pass truss_key so it appears in the output text
                    res = analyze_results(df, f"{scenario} [{truss_key}]", species, treatment)
                    
                    # Add truss type to the results dictionary for CSV
                    res['Truss_Type'] = truss_key 
                    all_res.append(res)
                    
                    # Incremental Save
                    summary_df = pd.DataFrame(all_res)
                    summary_df.to_csv('simulation_results.csv', index=False)
    
    # Generate Plots
    generate_professional_plots(summary_df)
    
    # Display Ranked Summary
    print("\n" + "="*60)
    print("RANKED RELIABILITY SUMMARY (Conservative Ranking)")
    print("="*60)
    
    # Sort by Reliability (Beta)
    ranked = summary_df.sort_values('Beta', ascending=False)
    
    # Clean output columns
    print(ranked[['Truss_Type', 'Scenario', 'Species', 'Treatment', 'Beta', 'Pf', 'Buckling%']])
    
    return summary_df

if __name__ == "__main__":
    # Recommended: Test with N=30 first to ensure Mono-pitch runs
    # Then increase to N=100,000 for the thesis run
    summary = run_all_scenarios(N=30)
