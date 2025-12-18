import numpy as np
import neal
from models.unbalanced_QUBO import QUBOMeanESPortfolio

# --- Trivial A: buy the only asset ---
returns_data = np.array([[0.10],
                         [0.10]])    # T=2, d=1
prices = np.array([100.0])           # π1 = 100
B = 100.0

model = QUBOMeanESPortfolio(
    returns_data=returns_data,
    prices=prices,
    risk_free_rate=0.0,   # r = 0
    budget=B,
    alpha=0.975,
    lambda_risk=0.0       # turn off ES term in the objective
)

bqm = model.build_qubo(
    penalty_budget=10000.0,  # strong budget enforcement
    lambda1_penalty=0.0,      # turn off unbalanced penalties
    lambda2_penalty=0.0
)

sampler = neal.SimulatedAnnealingSampler()
sampleset = sampler.sample(-bqm, num_reads=100)  # we built a "max" QUBO, so minimize -bqm
decoded = model.decode_solution(sampleset.first.sample)

print("=== Trivial A (buy the asset) ===")
print("xi_0:", decoded["xi_0"])
print("xi_risky:", decoded["xi_risky"])
print("total_budget_used:", decoded["total_budget_used"])

# Assertions for validation
assert decoded["xi_risky"][0] == 1.0, "Should buy the only asset"
assert abs(decoded["xi_0"] - 0.0) < 1e-9, "Risk-free allocation should be zero"
assert abs(decoded["total_budget_used"] - B) < 1e-6, "Budget must be exactly met"
print("Trivial A passed ✅")
