import numpy as np
import nashpy as nash
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import itertools # To iterate over multiple parameters

# --- 1. Define Baseline Parameters ---
# Focused on parameters influencing the Attacker's decision
BASE_PARAMS = {
    # Attacker-related Payoffs/Costs
    'B_attack': 200.0,      # Benefit for successful external Attack
    'base_C_attack': 10.0, # Baseline Cost to initiate Attack (e.g., for N_base=2, t_base=2)

    # Parameters for Scaling C_attack
    'N_base': 2.0,         # Reference N for scaling cost calculations
    't_base': 2.0,         # Reference t for scaling cost calculations
    'alpha_cost': 0.5,     # C_attack: Linear sensitivity to N increase from N_base
    'beta_cost': 1.7,      # C_attack: Exponential base for sensitivity to t increase from t_base
}

# --- 2. Define Parameter Functions ---

def C_attack(N, t, params=BASE_PARAMS):
    """Calculates estimated cost for an attacker based on N and t."""
    base_C = params.get('base_C_attack', 10.0)
    alpha = params.get('alpha_cost', 0.5)
    beta = params.get('beta_cost', 1.7)
    N_base = params.get('N_base', 2.0)
    t_base = params.get('t_base', 2.0)
    
    N_base_eff = max(1.0, N_base) # Ensure effective base N is at least 1
    t_base_eff = max(1.0, t_base) # Ensure effective base t is at least 1

    n_factor = 1 + alpha * max(0.0, N - N_base_eff)
    t_factor = beta ** max(0.0, t - t_base_eff)
    cost = base_C * n_factor * t_factor
    return cost

def p_success_attack(N, t, Ps): # Ps is a direct argument
    """
    Calculates probability of successful attack based on Binomial distribution B(N, Ps).
    Attack succeeds if k (number of compromised nodes) >= t.
    """
    if not isinstance(N, int) or N < 1 or not isinstance(t, int) or t < 1 or t > N:
        return 0.0 # Invalid parameters for binomial scenario
    if not (0 <= Ps <= 1):
        warnings.warn(f"Ps should be between 0 and 1. Got Ps={Ps}. Clamping.")
        Ps = max(0.0, min(1.0, Ps))

    try:
        # P(k >= t) is equivalent to P(k > t-1) for discrete distribution
        # binom.sf(k, n, p) calculates P(X > k)
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
for n_iter in range(2, max_N + 1): # N values to iterate
    for t_iter in range(2, n_iter + 1): # t values (2 <= t <= N)
         scenarios_Nt.append((n_iter, t_iter))

# Ps: Probability of compromising a single shareholder node
ps_values = [0.05, 0.20, 0.35, 0.50, 0.65, 0.8, 0.95]

print(f"Generated {len(scenarios_Nt) * len(ps_values)} total scenarios (N, t, Ps).")

# --- 5. Run Simulation and Store Results (Attacker Focus) ---
results_list = []

# Iterate over all combinations of (N, t) and Ps
for (N_val, t_val), ps_val in itertools.product(scenarios_Nt, ps_values):

    dummy_S, payoffs_A = calculate_payoffs_attacker_focus(N_val, t_val, ps_val, BASE_PARAMS)

    scenario_result = {
        'N': N_val, 't': t_val, 'Ps': ps_val,
        'A_NotAttack_PO': np.nan, 'A_Attack_PO': np.nan, # Attacker's payoffs
        'Equilibria_Count': 0,
        'A_Strategy_NotAttack': np.nan, # Attacker's equilibrium strategy
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
                # Check if Attacker's strategy is mixed
                is_A_mixed = not np.allclose(eq1[1], eq1[1].round())
                if is_A_mixed:
                    scenario_result['Error'] = 'Mixed Strategy NE for Attacker'
                
                # Store Attacker's probability of playing "Not Attack"
                scenario_result['A_Strategy_NotAttack'] = eq1[1][0]
            else:
                scenario_result['Error'] = 'No NE found'
        except Exception as e:
            scenario_result['Error'] = f'Computation Error: {type(e).__name__}'

    results_list.append(scenario_result)

# Convert results to DataFrame for analysis and plotting
results_df = pd.DataFrame(results_list)
float_cols = results_df.select_dtypes(include=['float64']).columns
results_df[float_cols] = results_df[float_cols].round(4)


# --- 6. Analyze and Visualize Results (Attacker Focus) ---
print("\n--- Simulation Results Summary (Attacker Focus) ---")
print(results_df) # Display summarized results

# Plotting Attacker Strategy: Prob(Attack) vs. t for different N, faceted by Ps
unique_ps = sorted(results_df['Ps'].unique())
all_n_groups = sorted(results_df['N'].unique())
num_ps = len(unique_ps)

fig1, axes1 = plt.subplots(1, num_ps, figsize=(5 * num_ps, 5), sharey=True, squeeze=False)
# squeeze=False ensures axes1 is always 2D array, even if num_ps=1

for i, ps_val in enumerate(unique_ps):
    ax = axes1[0, i] # Access subplot from 2D array
    subset_ps = results_df[results_df['Ps'] == ps_val]
    
    for n_group in all_n_groups:
        subset_n = subset_ps[subset_ps['N'] == n_group].sort_values(by='t')
        if not subset_n.empty:
            prob_attack = 1 - subset_n['A_Strategy_NotAttack']
            ax.plot(subset_n['t'], prob_attack, marker='o', label=f'N={n_group}')

    ax.set_xlabel("Threshold (t)")
    if i == 0:
        ax.set_ylabel("Prob(Attacker Attacks)")
    ax.set_title(f"Attacker Strategy vs t (Ps={ps_val:.2f})")
    ax.legend(title='N values')
    ax.grid(True)
    ax.set_ylim(-0.1, 1.1)
    if not results_df.empty and 't' in results_df.columns:
        ax.set_xlim(1.5, results_df['t'].max() + 0.5)

fig1.suptitle('Attacker Equilibrium Strategy vs. Threshold t, Faceted by Ps', y=1.03)
plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust rect for suptitle
plt.show()