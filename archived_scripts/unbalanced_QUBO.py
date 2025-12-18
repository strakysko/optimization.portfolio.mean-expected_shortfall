import numpy as np
import dimod

def binary_var(label, n_bits):
    """Create binary variable labels for encoding."""
    return [f"{label}_{i}" for i in range(n_bits)]

def build_qubo_unbalanced_penalty_dimod(rt_samples, expected_returns, π, B, r, C,
                                         J=4, K=4, L=4,
                                         λ=1.0, α=0.95, P=10.0,
                                         λ1=1.0, λ2=10.0):
    """
    Build the unbalanced-penalization QUBO model using dimod.
    """
    T, d = rt_samples.shape
    bqm = dimod.BinaryQuadraticModel('BINARY')

    # Variable labels
    ξ_vars = [f"ξ_{i}" for i in range(d)]
    a_vars = binary_var("a", J)  # ξ0
    b_vars = binary_var("b", K)  # τ
    ct_vars = [binary_var(f"ct_{t}", L) for t in range(T)]  # y_t

    # Binary encoding steps
    ξ0_step = B / (2**J - 1)
    τ_step = (B - C) / (2**K - 1)
    y_step = (B - C) / (2**L - 1)

    # ---- Portfolio Return Term ----
    # ξ0 term (non-risky asset)
    for j, var in enumerate(a_vars):
        coeff = r * ξ0_step * 2**j
        bqm.add_variable(var, -coeff)  # maximize → minimize negative

    # Risky assets expected return
    for i, var in enumerate(ξ_vars):
        coeff = -expected_returns[i]  # maximize
        bqm.add_variable(var, coeff)

    # ---- Expected Shortfall Term ----
    for k, var in enumerate(b_vars):  # τ
        coeff = λ * τ_step * 2**k
        bqm.add_variable(var, coeff)

    for t in range(T):
        for l, var in enumerate(ct_vars[t]):  # y_t
            coeff = λ / ((1 - α) * T) * y_step * 2**l
            bqm.add_variable(var, coeff)

    # ---- Budget Constraint Penalty (ξ0 + ξ·π - B)^2 ----
    budget_vars = {}
    # ξ0 terms
    for j, var in enumerate(a_vars):
        budget_vars[var] = ξ0_step * 2**j

    # ξ terms (risky assets)
    for i, var in enumerate(ξ_vars):
        budget_vars[var] = π[i]

    # Build quadratic budget penalty
    for u, coeff_u in budget_vars.items():
        bqm.add_variable(u, bqm.get_linear(u) + P * (coeff_u**2 - 2 * coeff_u * B))
        for v, coeff_v in budget_vars.items():
            if u < v:
                bqm.add_interaction(u, v, 2 * P * coeff_u * coeff_v)

    bqm.offset += P * B**2

    # ---- Unbalanced Penalization for yt + r*ξ0 + rt·ξ + τ ≥ 0 ----
    for t in range(T):
        linear_combo = {}

        # yt
        for l, var in enumerate(ct_vars[t]):
            linear_combo[var] = y_step * 2**l

        # ξ0 (a_vars)
        for j, var in enumerate(a_vars):
            linear_combo[var] = linear_combo.get(var, 0) + r * ξ0_step * 2**j

        # ξ·rt
        for i, var in enumerate(ξ_vars):
            linear_combo[var] = linear_combo.get(var, 0) + rt_samples[t, i]

        # τ (b_vars)
        for k, var in enumerate(b_vars):
            linear_combo[var] = linear_combo.get(var, 0) + τ_step * 2**k

        # Constant term C
        const = C

        # Expand the unbalanced penalty: -λ1 * h + λ2 * h²
        # Linear: -λ1 * coeffs
        for var, coeff in linear_combo.items():
            bqm.add_variable(var, bqm.get_linear(var) + -λ1 * coeff + λ2 * coeff**2)

        # Quadratic: λ2 * coeff_u * coeff_v
        vars_list = list(linear_combo.items())
        for i in range(len(vars_list)):
            for j in range(i + 1, len(vars_list)):
                (var_u, coeff_u), (var_v, coeff_v) = vars_list[i], vars_list[j]
                bqm.add_interaction(var_u, var_v, 2 * λ2 * coeff_u * coeff_v)

        # Linear with constant: λ2 * 2 * const * coeff
        for var, coeff in linear_combo.items():
            bqm.add_variable(var, bqm.get_linear(var) + 2 * λ2 * const * coeff)

        # Constant terms
        bqm.offset += -λ1 * const + λ2 * const**2

    return bqm