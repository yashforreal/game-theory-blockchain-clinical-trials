import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import itertools

# --- 1. Define Baseline Parameters ---
BASE_PARAMS = {
    'R': 10.0,          # Reward/Benefit for Honest participation
    'C': 5.0,           # Cost of Honest participation
    # B_collude and P_collude will be varied on the axes for the plots.
    # Attacker-related parameters are not needed for this specific shareholder analysis.
    'N_base': 2.0,      # Reference N for p_detect scaling (if used in a more complex p_detect)
    't_base': 2.0,      # Reference t for p_detect scaling (if used in a more complex p_detect)
    'base_p_detect': 0.1, # Baseline detection probability
    'gamma_detect': 0.8,  # Sensitivity of p_detect to the proportion t/N
}

# --- 2. Define Needed Parameter Functions ---
def p_detect(t, N, params=BASE_PARAMS):
    """Calculates estimated probability of detecting collusion."""
    if N <= 0 or t < 1: 
        return params.get('base_p_detect', 0.1) # Default for invalid inputs

    base_p = params.get('base_p_detect', 0.1)
    gamma = params.get('gamma_detect', 0.8)

    # Effective N ensures denominator is valid and at least t
    N_eff = max(N, t, 1) 
    prob = base_p + gamma * (t / N_eff)
    # Clamp result slightly below 1.0 to avoid division by zero in critical benefit calculation
    return max(0.0, min(0.99999, prob)) 

# --- 3. Function to Calculate Critical Benefit ---
def calculate_critical_b_collude(P_collude_val, N, t, params=BASE_PARAMS):
    """
    Calculates the critical B_collude value where a shareholder is
    indifferent between Honest and Collude, given P_collude, N, t.
    """
    if not isinstance(N, int) or not isinstance(t, int) or N < 1 or t < 1 or t > N:
        warnings.warn(f"Invalid N={N}, t={t} combination.")
        return np.nan 

    prob_detect = p_detect(t, N, params)

    # Handle edge case where detection is certain (1 - prob_detect would be zero)
    if np.isclose(prob_detect, 1.0):
        # If detection is certain, collusion is rational only if R + P_collude < 0, 
        # which is impossible if R and P_collude are non-negative.
        # Thus, an infinite benefit would be required to make collusion rational.
        return np.inf 

    R_val = params.get('R', 10.0)
    # Note: C_val (Cost of Honesty) cancels out in the indifference equation: R-C vs E[Collude]-C

    # Indifference: R = (1 - prob_detect) * (B_collude_critical / t) - prob_detect * P_collude
    # Solved for B_collude_critical:
    b_critical = (R_val + prob_detect * P_collude_val) * t / (1 - prob_detect)
    return max(0.0, b_critical) # Benefit should be non-negative


# --- 4. Define Scenarios and Parameters to Vary for Plots ---
thresholds_to_plot = [5, 10, 15, 20]
max_N_plot = 40 # Maximum N to show on plots

# P_collude values for the x-axis
p_collude_values = np.linspace(0, 10000, 100) 

# --- 5. Generate Data for Plots ---
plot_data = []

for t_fixed in thresholds_to_plot:
    # Generate N values for each t_fixed: t_fixed itself, and then in steps
    n_values_for_t = sorted(list(set([t_fixed] + list(range(t_fixed + 5, max_N_plot + 1, 10)))))
    if t_fixed > max_N_plot : continue 

    for n_val in n_values_for_t:
        if n_val < t_fixed: continue 

        for p_val in p_collude_values:
            b_crit = calculate_critical_b_collude(p_val, n_val, t_fixed, BASE_PARAMS)
            if not np.isinf(b_crit): # Store only finite critical benefit values
                plot_data.append({'t_fixed': t_fixed, 'N': n_val, 'P_collude': p_val, 'B_collude_critical': b_crit})

plot_df = pd.DataFrame(plot_data)

# --- 6. Create the Benefit vs Penalty Plots ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
axes = axes.flatten() 

# Define a color map for different N values for consistency across subplots
unique_n_plotting = sorted(plot_df['N'].unique())
plot_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_n_plotting)))
n_color_map = {n: color for n, color in zip(unique_n_plotting, plot_colors)}

for i, t_fixed in enumerate(thresholds_to_plot):
    if i >= len(axes): break 
    ax = axes[i]
    subset_t = plot_df[plot_df['t_fixed'] == t_fixed]

    if subset_t.empty:
        ax.set_title(f'Benefit vs Penalty (t = {t_fixed}) - No Data')
        continue
    
    for n_val in sorted(subset_t['N'].unique()):
        subset_n = subset_t[subset_t['N'] == n_val].sort_values(by='P_collude')
        if not subset_n.empty:
            ax.plot(subset_n['P_collude'], subset_n['B_collude_critical'],
                    label=f'N={n_val}', color=n_color_map.get(n_val))

    # Annotate regions for clarity
    ax.text(0.05, 0.9, 'Collude Preferred Region\n(Above Lines)', transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', color='red', alpha=0.7)
    ax.text(0.95, 0.1, 'Honest Preferred Region\n(Below Lines)', transform=ax.transAxes, 
            fontsize=9, verticalalignment='bottom', horizontalalignment='right', color='green', alpha=0.7)

    ax.set_title(f'Benefit vs Penalty Boundary (t = {t_fixed})')
    ax.set_xlabel("Penalty if Caught Colluding (P_collude)")
    ax.set_ylabel("Critical Benefit of Collusion (B_collude)")
    ax.legend(title='N values', loc='upper left')
    ax.grid(True)
    # Consider ax.set_yscale('log') or ax.set_xscale('log') if ranges are very wide

fig.suptitle('Shareholder Strategy Boundary: Critical Collusion Benefit vs. Penalty for Fixed Thresholds', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to prevent suptitle overlap
plt.show()