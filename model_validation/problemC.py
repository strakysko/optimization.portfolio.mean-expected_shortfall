import numpy as np
import neal
from models.unbalanced_QUBO import QUBOMeanESPortfolio

# --- Medium Test A (prices sum to 100) ---

B = 100.0
returns_data = np.array([
    [0.10, 0.12, -0.02],
    [0.10, 0.12, -0.02],  # T=2 identical rows
])
prices = np.array([55.0, 35.0, 10.0])   # <-- sums to 100

# assert abs(prices.sum() - 100.0) < 1e-9

model = QUBOMeanESPortfolio(
    returns_data=returns_data,
    prices=prices,
    risk_free_rate=0.0,    # r = 0
    budget=B,
    alpha=0.975,
    lambda_risk=0.0        # ES off
)

bqm = model.build_qubo(
    penalty_budget=1000.0,  # enforce budget tightly
    lambda1_penalty=0.0,
    lambda2_penalty=0.0
)

ss = neal.SimulatedAnnealingSampler().sample(-bqm, num_reads=500)
dec = model.decode_solution(ss.first.sample)

print("=== Medium A ===")
print("xi_risky:", dec["xi_risky"], "xi_0:", dec["xi_0"], "budget_used:", dec["total_budget_used"])

# Assertions
scale = model.B / (2**model.J - 1)    # ξ0 step ≈ 100/63 ≈ 1.587
assert np.allclose(dec["xi_risky"], [1, 1, 0]), "Should pick assets 1 & 2 only"
assert dec["expected_return"] > 0.21, "Mean should be near 0.22"
assert dec["budget_violation"] <= scale + 1e-6, "Budget should be met within encoding step"
print("Medium A passed ✅")