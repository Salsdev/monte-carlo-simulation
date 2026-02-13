import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

# =================================================================
# 1. PARAMETERS & DATA SETUP (Phase 1)
# =================================================================

SPECIES_DATA = {
    'W': {  # Anogeissus leiocarpa (White Wood)
        'name': 'Anogeissus leiocarpa',
        'rho': {'dist': 'normal', 'mean': 809.0, 'std': 72.0},  # kg/m^3
        'fm': {'dist': 'gumbel', 'mean': 22.7, 'std': 5.7},     # N/mm^2
        'fv': {'dist': 'normal', 'mean': 2.27, 'std': 0.57},    # Shear strength (Derived as 0.1*fm)
        'E': {'dist': 'lognormal', 'mean': 4612.0, 'std': 1247.0},
        'MC': {'dist': 'normal', 'mean': 0.1525, 'std': 0.0425}, 
        'thermal': {'lambda': 0.13, 'cp': 1500}, # W/mK, J/kgK
        'char_insulation': 12.0,
        'b': 75, # mm Cross-Section Dimensions: Matched to the experimental fire test specimens
        'h': 125, # mm Essential for validating the charring rate models against experimental data.
        'weights': {'exp': 0.4, 'mikkola': 0.3, 'hietaniemi': 0.3} 
    },
    'R': {  # Erythrophleum suaveolens (Red Wood)
        'name': 'Erythrophleum suaveolens',
        'rho': {'dist': 'normal', 'mean': 745.0, 'std': 129.0}, 
        'fm': {'dist': 'normal', 'mean': 38.9, 'std': 9.3},     
        'fv': {'dist': 'normal', 'mean': 3.89, 'std': 0.93},    # Shear strength (Derived as 0.1*fm)
        'E': {'dist': 'normal', 'mean': 8935.0, 'std': 1591.0},  
        'MC': {'dist': 'normal', 'mean': 0.1958, 'std': 0.0915}, 
        'thermal': {'lambda': 0.16, 'cp': 1600},
        'char_insulation': 18.0, 
        'b': 75, # mm Cross-Section Dimensions: Matched to the experimental fire test specimens
        'h': 125, # mm
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

    # 3. Exponential decay formula from your methodology
    # x is the distance from the heat source/char line
    T = T_init + (T_fire - T_init) * np.exp(-x / np.sqrt(4 * alpha * t_seconds))

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

def calculate_average_temperature(d_char, t, T_fire, species_key):
    """
    Estimates the temperature in the heated zone of the residual core.
    We take a point 30mm from the char line into the residual core
    to represent the average state of the load-bearing zone.
    """
    if t <= 0: return 20.0
    # Depth x is distance from the char line (m)
    depth_x = 30.0e-3 
    # The temperature AT the char line is always 300C (isotherm definition)
    # We use 300C as boundary instead of T_fire for the internal gradient.
    return calculate_internal_temperature(depth_x, t, 300.0, 20, species_key)

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
        return 0, 0, 0 
        
    n_layers = 20
    h_residual = h0 - d_char
    b_residual = b0 - 2 * d_char
    layer_thickness = h_residual / n_layers
    
    A_eff = 0
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
        stat_moment_mod += (dA * k_mod) * x_dist
        layer_data.append((x_dist, dA, k_E, k_mod))

    if A_eff <= 0: return 0, 0, 0

    # Section properties depends on the property being analyzed
    # Stiffness centroid (for buckling/deflection)
    sum_kE_dA = sum(d[1] * d[2] for d in layer_data)
    y_bar_fi_E = stat_moment_E / sum_kE_dA if sum_kE_dA > 0 else h_residual / 2
    
    # Strength centroid (for bending)
    y_bar_fi_mod = stat_moment_mod / A_eff if A_eff > 0 else h_residual / 2
    
    for x_dist, dA, k_E, k_mod in layer_data:
        # Stiffness Calculation
        dist_to_na_E = x_dist - y_bar_fi_E
        I_layer_E = (b_residual * layer_thickness**3 / 12) + dA * dist_to_na_E**2
        I_eff_stiffness += I_layer_E * k_E
        
        # Strength Calculation
        dist_to_na_mod = x_dist - y_bar_fi_mod
        I_layer_mod = (b_residual * layer_thickness**3 / 12) + dA * dist_to_na_mod**2
        I_eff_strength += I_layer_mod * k_mod
        
    # Distance to extreme fibers from strength-weighted NA
    y_top = h_residual - y_bar_fi_mod
    y_bottom = y_bar_fi_mod
    W_eff = I_eff_strength / max(y_top, y_bottom)
    
    return A_eff, W_eff, I_eff_stiffness

def bending_limit_state(M_Ed, fm, W_ef, theta_R):
    """
    G = theta_R * M_Rd,fi - M_Ed
    Integrated properties already account for k_mod.
    """
    gamma_M_fi = 1.0 
    M_Rd_fi = fm * W_ef / gamma_M_fi
    return theta_R * M_Rd_fi - M_Ed

def buckling_limit_state(N_Ed, f_c0_k, E_05_fi, I_ef, A_ef, theta_R, L_unbraced=2000):
    """
    Corrected Buckling Limit State per Methodology.
    
    Args:
        N_Ed: Design axial load in fire (N)
        f_c0_k: Characteristic compressive strength (N/mm^2)
        E_05_fi: 5th percentile Stiffness (N/mm^2)
        I_ef: Effective second moment of area (mm^4) (includes stiffness reduction kE)
        A_ef: Effective area (mm^2) (includes strength reduction k_mod)
        theta_R: Model uncertainty factor
    """
    # 1. Calculate Radius of Gyration (i_ef)
    if A_ef <= 0: return -1.0 # Failure if section is gone
    i_ef = np.sqrt(I_ef / A_ef)
    
    # 2. Calculate Relative Slenderness (lambda_fi)
    # L_cr is the effective buckling length
    lambda_fi = (L_unbraced / (np.pi * i_ef)) * np.sqrt(f_c0_k / E_05_fi)
    
    # 3. Calculate Instability Factor k
    # beta_c = 0.1 for the specific methodology/glulam
    beta_c = 0.1 
    k = 0.5 * (1 + beta_c * (lambda_fi - 0.3) + lambda_fi**2)
    
    # 4. Calculate Buckling Reduction Factor k_c,fi
    # This reduces capacity based on slenderness
    denom = k + np.sqrt(k**2 - lambda_fi**2)
    if k**2 < lambda_fi**2: # Catch math domain error if very slender
        k_c_fi = 0.0
    else:
        k_c_fi = 1.0 / denom
        k_c_fi = min(k_c_fi, 1.0) # Cap at 1.0
    
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
    elif params['dist'] == 'static':
        return params['mean']
    return params['mean']

def sample_uncertainty(mean=1.0, cov=0.1):
    """
    Samples theta from a lognormal distribution.
    """
    sigma = np.sqrt(np.log(1 + cov**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return np.random.lognormal(mu, sigma)

def run_simulation(N=1000, species_key='W', scenario_key='FTI', treatment='untreated', sensitivity=True):
    """
    Main Monte Carlo Simulation Loop with Full Fidelity.
    """
    fire = FIRE_SCENARIOS[scenario_key]
    species = SPECIES_DATA[species_key]
    results = []
    samples = []
    
    # Stochastic Loads Parameters
    G_params = {'mean': 1.0e6, 'std': 0.1e6} # Dead Load (N-mm) proxy
    Q_params = {'mean': 0.5e6, 'std': 0.15e6} # Live Load (N-mm) proxy
    
    # Compute Characteristic values for Buckling (once per species)
    # Using 5th percentile of bending and stiffness as proxy for fc0,k and E05
    def get_k_vals(p):
        if p['dist'] == 'gumbel':
            beta = p['std'] * np.sqrt(6) / np.pi
            mu = p['mean'] - np.euler_gamma * beta
            return mu - beta * np.log(-np.log(0.05))
        elif p['dist'] == 'lognormal':
            sigma = np.sqrt(np.log(1 + (p['std']/p['mean'])**2))
            mu = np.log(p['mean']) - 0.5 * sigma**2
            return np.exp(mu - 1.645 * sigma)
        else: # normal or static
            return p['mean'] - 1.645 * p['std']
            
    f_c0_k = get_k_vals(species['fm'])
    # Eurocode fire factor k_fi (approx 1.25 for glulam)
    E_05_fi = get_k_vals(species['E']) * 1.25 
    
    print(f"Running Full Fidelity MCS: {N} iterations for {species['name']}...")
    
    for _ in tqdm(range(N)):
        # 1. Sample Random Variables
        rho = sample_variable(species['rho'])
        fm = sample_variable(species['fm'])
        fv = sample_variable(species['fv'])
        E = sample_variable(species['E'])
        MC = sample_variable(species['MC'])
        
        # 2. Sample Uncertainties
        theta_model = sample_uncertainty(1.0, 0.10)
        theta_R = sample_uncertainty(1.0, 0.10)
        theta_E = sample_uncertainty(1.0, 0.05)
        
        # 3. Sample Experimental Charring Rate (Lognormal)
        # Map 'borax' to 'treated' to match the experimental data keys
        cond = 'treated' if treatment == 'borax' else 'untreated'
        char_params = EXPERIMENTAL_CHARRING_DATA[scenario_key][species_key][cond]

        # Retrieve arithmetic Mean and COV from data
        m = char_params['mean']
        cov = char_params['cov']

        # Convert arithmetic statistics to underlying Normal distribution parameters
        # Variance v = (mean * cov)^2
        v = (m * cov)**2 

        # sigma_ln = sqrt(ln(1 + v / m^2)) = sqrt(ln(1 + cov^2))
        sigma_ln = np.sqrt(np.log(1 + v / m**2))

        # mu_ln = ln(mean) - 0.5 * sigma_ln^2
        mu_ln = np.log(m) - 0.5 * sigma_ln**2

        # Sample from Lognormal distribution
        beta_exp_sampled = np.random.lognormal(mu_ln, sigma_ln)
        
        # 4. Sample Loads (Structural Analysis Logic)
        G_load = np.random.normal(G_params['mean'], G_params['std'])
        # FIX: Correct Gumbel sampling using the utility logic
        Q_load = sample_variable({'dist': 'gumbel', 'mean': Q_params['mean'], 'std': Q_params['std']})
        
        E_d_fi = theta_E * (G_load + 0.3 * Q_load) # Fire load combination
        
        current_samples = {'rho': rho, 'fm': fm, 'fv': fv, 'E': E, 'MC': MC, 'load': E_d_fi}
        
        failed = False
        time_of_failure = fire['duration']
        failure_mode = 'none'
        char_depth = 0.0
        
        for t in range(0, fire['duration'] + 1):
            T_fire = get_fire_temperature(t, scenario_key)
            
            # 1. Update char depth (mm)
            if t > 0:
                # Calculate rate based on state at previous step
                beta_eff = combined_charring_rate(t, species_key, scenario_key, rho, MC, beta_exp_sampled, theta_model)
                char_depth += beta_eff 
            
            # 2. Section Analysis
            A_ef, W_ef, I_ef = get_effective_section_integrated(species['b'], species['h'], char_depth, species_key, t, T_fire)
            
            # 3. Limit State Checks (Redundant thermal analysis removed)
            M_Ed = E_d_fi
            N_Ed = 0.02 * E_d_fi 
            V_Ed = 0.001 * E_d_fi 
            
            if bending_limit_state(M_Ed, fm, W_ef, theta_R) <= 0:
                failure_mode = 'Bending'; failed = True
            elif buckling_limit_state(N_Ed, f_c0_k, E_05_fi, I_ef, A_ef, theta_R) <= 0:
                failure_mode = 'Buckling'; failed = True
            elif shear_limit_state(V_Ed, fv, A_ef, theta_R) <= 0:
                failure_mode = 'Shear'; failed = True
                
            if failed:
                time_of_failure = t; break
        
        res = {'failed': failed, 'time': time_of_failure, 'mode': failure_mode}
        results.append(res)
        if sensitivity:
            samples.append({**current_samples, 'failed': 1 if failed else 0})
            
    res_df = pd.DataFrame(results)
    if sensitivity and any(res_df['failed']):
        samp_df = pd.DataFrame(samples)
        corr = samp_df.corr(method='spearman')['failed'].drop('failed')
        print(f"\nSensitivity Analysis (Spearman) for {species['name']}:")
        print(corr)
        
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
    plot_df = df_sorted[['TempLabels', 'Bending%', 'Buckling%', 'Shear%']].set_index('TempLabels')
    
    plot_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#4c72b0', '#55a868', '#c44e52'])
    plt.title('Failure Mode Distribution across Scenarios', fontsize=14)
    plt.ylabel('Relative Frequency (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('failure_modes_distribution.png', dpi=300)
    plt.close()

def run_all_scenarios(N=1000):
    all_res = []
    
    for scenario in FIRE_SCENARIOS:
        for species in SPECIES_DATA:
            for treatment in ['untreated', 'borax']:
                df = run_simulation(N, species, scenario, treatment)
                res = analyze_results(df, scenario, species, treatment)
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
    # Sorting by Beta_Low for safety (worst-case reliable scenario)
    ranked = summary_df.sort_values('Beta_Low', ascending=False)
    print(ranked[['Scenario', 'Species', 'Treatment', 'Beta', 'Beta_Low', 'Pf']])
    
    return summary_df

if __name__ == "__main__":
    # Trial run with N=10,000
    summary = run_all_scenarios(N=1000)
