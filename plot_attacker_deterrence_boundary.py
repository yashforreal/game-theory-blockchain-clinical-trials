import numpy as np
import nashpy as nash
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import itertools

# --- 1. Define Baseline Parameters ---
BASE_PARAMS = {
    'B_attack': 200.0,      # Benefit for successful external Attack
    'base_C_attack': 10.0, # Baseline Cost to initiate Attack (e.g., for N_base=2, t_base=2)
    'N_base': 2.0,         # Reference N for scaling cost calculations
    't_base': 2.0,         # Reference t for scaling cost calculations
    'alpha_cost': 0.5,     # C_attack: Linear sensitivity to N increase from N_base
    'beta_cost': 1.7,      # C_attack: Exponential base for sensitivity to t increase from t_base
    # Note: Shareholder/detection parameters are omitted as this script focuses on Attacker strategy.
}

# --- 2. Define Parameter Functions ---

def C_attack(N, t, params=BASE_PARAMS):
    """Calculates estimated cost for an attacker based on N and t."""
    base_C = params.get('base_C_attack', 10.0)
    alpha = params.get('alpha_cost', 0.5)
    beta = params.get('beta_cost', 1.7)
    N_base = params.get('N_base', 2.0)
    t_base = params.get('t_base', 2.0)
    
    N_base_eff = max(1.0, N_base)
    t_base_eff = max(1.0, t_base)

    n_factor = 1 + alpha * max(0.0, N - N_base_eff)
    t_factor = beta ** max(0.0, t - t_base_eff)
    cost = base_C * n_factor * t_factor
    return cost

def p_success_attack(N, t, Ps):
    """
    Calculates probability of successful attack based on Binomial distribution B(N, Ps).
    Attack succeeds if k (number of compromised nodes) >= t.
    """
    if not isinstance(N, int) or N < 1 or not isinstance(t, int) or t < 1 or t > N:
        return 0.0
    if not (0 <= Ps <= 1):
        warnings.warn(f"Ps should be between 0 and 1. Got Ps={Ps}. Clamping.")
        Ps = max(0.0, min(1.0, Ps))

    try:
        # P(k >= t) is equivalent to P(k > t-1) for discrete distribution.
        # binom.sf(k_val, n_trials, p_success_trial) calculates P(X > k_val).
        prob = binom.sf(k=t-1, n=N, p=Ps)
    except TypeError:
        warnings.warn(f"Type error during binomial calculation for N={N}, t={t}, Ps={Ps}")
        return 0.0
    return prob

# --- 3. Function to Calculate Payoff Matrices ---
def calculate_payoffs_attacker_focus(N, t, Ps, params=BASE_PARAMS):
    """
    Calculates Attacker's payoff matrix (payoffs_A).
    Returns a dummy Shareholder payoff matrix (dummy_S) as Nashpy requires two matrices.
    """
    if not isinstance(N, int) or not isinstance(t, int) or N < 1 or t < 1 or t > N:
        return None, None # Invalid input combination

    try:
        p_success = p_success_attack(N, t, Ps)
        cost_attack = C_attack(N, t, params)
    except Exception as e:
        print(f"Error calculating dynamic values for N={N}, t={t}, Ps={Ps}: {e}")
        return None, None

    B_attack_val = params.get('B_attack', 200.0)
    not_attack_payoff_A = 0.0
    attack_payoff_A = p_success * B_attack_val - cost_attack

    # Attacker's payoffs are independent of Shareholder's strategy in this model
    payoffs_A = np.array([
        [not_attack_payoff_A, attack_payoff_A], 
        [not_attack_payoff_A, attack_payoff_A]
    ])

    # Nashpy requires two payoff matrices for a 2-player game definition
    dummy_payoffs_S = np.zeros((2, 2)) 

    return dummy_payoffs_S, payoffs_A

# --- 4. Define Scenarios and Parameters to Vary ---
max_N = 20 
scenarios_Nt = []
for n_iter in range(2, max_N + 1):
    for t_iter in range(2, n_iter + 1):
         scenarios_Nt.append((n_iter, t_iter))

# Ps: Probability of compromising a single shareholder node
ps_values = [0.05, 0.20, 0.35, 0.50, 0.65, 0.8, 0.95]

print(f"Generated {len(scenarios_Nt) * len(ps_values)} total scenarios (N, t, Ps).")

# --- 5. Run Simulation and Store Results (Attacker Focus) ---
results_list = []

for (N_val, t_val), ps_val in itertools.product(scenarios_Nt, ps_values):
    dummy_S, payoffs_A = calculate_payoffs_attacker_focus(N_val, t_val, ps_val, BASE_PARAMS)

    scenario_result = {
        'N': N_val, 't': t_val, 'Ps': ps_val,
        'A_NotAttack_PO': np.nan, 'A_Attack_PO': np.nan,
        'Equilibria_Count': 0,
        'A_Strategy_NotAttack': np.nan,
        'Error': None
    }

    if dummy_S is None: 
        scenario_result['Error'] = 'Invalid N/t in payoff calc'
    else:
        scenario_result['A_NotAttack_PO'] = payoffs_A[0, 0]
        scenario_result['A_Attack_PO'] = payoffs_A[0, 1]

        try:
            game = nash.Game(dummy_S, payoffs_A)
            equilibria = list(game.support_enumeration(tol=1e-7))
            scenario_result['Equilibria_Count'] = len(equilibria)

            if len(equilibria) > 0:
                eq1 = equilibria[0] # Analyze the first equilibrium found
                is_A_mixed = not np.allclose(eq1[1], eq1[1].round())
                if is_A_mixed:
                    scenario_result['Error'] = 'Mixed Strategy NE for Attacker'
                
                scenario_result['A_Strategy_NotAttack'] = eq1[1][0] # Prob Attacker plays "Not Attack"
            else:
                scenario_result['Error'] = 'No NE found'
        except Exception as e:
            scenario_result['Error'] = f'Computation Error: {type(e).__name__}'

    results_list.append(scenario_result)

results_df = pd.DataFrame(results_list)
float_cols = results_df.select_dtypes(include=['float64']).columns
results_df[float_cols] = results_df[float_cols].round(4)


# --- 6. Analyze and Visualize Results (Attacker Focus) ---
print("\n--- Simulation Results Summary (Attacker Focus) ---")
print(results_df.head())

# --- Calculate Critical Threshold t* ---
critical_t_data = []
unique_n_values = sorted(results_df['N'].unique())
unique_ps_values = sorted(results_df['Ps'].unique())

for n_val in unique_n_values:
    for ps_val in unique_ps_values:
        subset = results_df[(results_df['N'] == n_val) & (results_df['Ps'] == ps_val)].sort_values(by='t')
        
        t_critical = np.nan
        
        if not subset.empty:
            # Find the first t where Attacker chooses "Not Attack" (A_Strategy_NotAttack >= 0.9)
            switched_to_not_attack_df = subset[subset['A_Strategy_NotAttack'] >= 0.9] 
            
            if not switched_to_not_attack_df.empty:
                t_critical = switched_to_not_attack_df['t'].min()
            else:
                # If attacker consistently attacks for all tested t for this (N, Ps)
                max_t_for_scenario = subset['t'].max()
                if subset['A_Strategy_NotAttack'].max() < 0.1: 
                     t_critical = max_t_for_scenario + 1 # Signifies deterrence not achieved within t<=N
        
        critical_t_data.append({'N': n_val, 'Ps': ps_val, 't_critical': t_critical})

critical_t_df = pd.DataFrame(critical_t_data)
print("\n--- Critical Threshold (t*) Data ---")
print(critical_t_df)

# --- Generate the Consolidated Attacker Strategy Plot ---
plt.figure(figsize=(10, 7))

markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', '<', '>', 'H', 'd'] # Expanded markers
line_styles = ['-', '--', '-.', ':']
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, n_val in enumerate(unique_n_values):
    subset_n = critical_t_df[critical_t_df['N'] == n_val].sort_values(by='Ps')
    plot_subset_n = subset_n.dropna(subset=['t_critical']) # Ensure t_critical is not NaN for plotting

    if not plot_subset_n.empty:
        plt.plot(plot_subset_n['Ps'], plot_subset_n['t_critical'],
                 label=f'N={n_val}',
                 marker=markers[i % len(markers)],
                 linestyle=line_styles[(i // len(markers)) % len(line_styles)], 
                 color=color_cycle[i % len(color_cycle)])

plt.xlabel("Probability of Single Share Compromise (Ps)")
plt.ylabel("Critical Threshold (t*) to Deter Attack")
plt.title("Attacker Deterrence: Critical Threshold t* vs. Ps for different Network Sizes (N)")
plt.legend(title="N values", bbox_to_anchor=(1.05, 1), loc='upper left') # Legend outside plot
plt.grid(True)

if not critical_t_df['t_critical'].dropna().empty:
    min_y_plot = 1.5 
    max_y_plot = critical_t_df['t_critical'].replace(np.inf, np.nan).max() 
    if pd.notna(max_y_plot):
         plt.ylim(min_y_plot, max_y_plot + 1) 
    else: 
         plt.ylim(min_y_plot, scenarios_Nt[-1][1] + 1 if scenarios_Nt else 21)
else:
    plt.ylim(1.5, 21)

plt.xticks(ps_values) # Ensure all tested Ps values are marked
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect for legend
plt.show()