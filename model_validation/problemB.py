import numpy as np
import neal
from models.unbalanced_QUBO import QUBOMeanESPortfolio

# --- Trivial B: all in cash ---
returns_data = np.array([[-0.10],
                         [-0.10]])    # T=2, d=1
prices = np.array([100.0])
B = 100.0

model = QUBOMeanESPortfolio(
    returns_data=returns_data,
    prices=prices,
    risk_free_rate=0.0,
    budget=B,
    alpha=0.975,
    lambda_risk=0.0
)

bqm = model.build_qubo(
    penalty_budget=10_000.0,
    lambda1_penalty=0.0,
    lambda2_penalty=0.0
)

sampler = neal.SimulatedAnnealingSampler()
sampleset = sampler.sample(-bqm, num_reads=100)
decoded = model.decode_solution(sampleset.first.sample)

print("=== Trivial B (all in cash) ===")
print("xi_0:", decoded["xi_0"])
print("xi_risky:", decoded["xi_risky"])
print("total_budget_used:", decoded["total_budget_used"])

# Assertions for validation
assert decoded["xi_risky"][0] == 0.0, "Should NOT buy the asset"
# With J=6, ξ0 can exactly hit B by setting all a_j=1
assert abs(decoded["xi_0"] - B) < 1e-6, "Risk-free allocation should equal the budget"
assert abs(decoded["total_budget_used"] - B) < 1e-6, "Budget must be exactly met"
print("Trivial B passed ✅")
