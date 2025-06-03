# Game-Theoretic Analysis of Participant Incentives in Clinical Trial Blockchains

This repository provides the Python simulation code for the research paper investigating strategic behaviors in permissioned blockchain networks designed for clinical trials. The code implements two main game-theoretic analyses:

* **Attacker Strategy Analysis:** Simulates an external attacker's decision-making process (`Attack` vs. `Not Attack`) based on Expected Utility and computes Nash Equilibria using parameters `N` (total nodes), `t` (threshold), and `Ps` (single-node compromise probability). Includes scripts for generating plots of the attacker's equilibrium strategy and the critical deterrence threshold `t^*`.
* **Shareholder Incentive Boundary Analysis:** Calculates the critical trade-off between the Benefit of Collusion (`B_{collude}`) and the Penalty for Collusion (`P_{collude}`) that determines a shareholder's rational choice between `Honest` participation and `Collude`. Includes scripts for plotting this decision boundary for various `N` and fixed `t` values.

The models use `NumPy`, `SciPy` (for binomial distributions), `Nashpy`, `Pandas`, and `Matplotlib`. See this README for further details on running the simulations and interpreting the outputs.
