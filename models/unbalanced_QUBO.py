import numpy as np
import dimod
from typing import List, Dict, Tuple
import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
import neal


class QUBOMeanESPortfolio:
    """
    Unbalanced-penalization-based QUBO Mean-ES portfolio optimization model
    following the formulation from equation (5.15) in the thesis.
    """

    def __init__(self, returns_data: np.ndarray, prices: np.ndarray,
                 risk_free_rate: float, budget: float = 100.0,
                 alpha: float = 0.975, lambda_risk: float = 1.0):
        """
        Initialize the QUBO Mean-ES portfolio model.

        Args:
            returns_data: Array of shape (T, d) containing T observations of d risky assets returns
            prices: Array of shape (d,) containing prices of d risky assets
            risk_free_rate: Risk-free interest rate r
            budget: Total budget B (default: 100.0)
            alpha: Confidence level for ES (default: 0.95)
            lambda_risk: Risk aversion parameter λ (default: 1.0)
        """
        self.returns_data = returns_data  # T x d matrix
        self.T, self.d = returns_data.shape
        self.prices = prices  # d-dimensional vector (π_i for risky assets)
        self.r = risk_free_rate / 251 * 10 # interest rate per 10 trading days with 251 trading days per year
        self.B = budget
        self.alpha = alpha
        self.lambda_risk = lambda_risk

        # Calculate expected returns
        self.expected_returns = np.mean(returns_data, axis=0)

        # Calculate covariance matrix and its largest eigenvalue
        # self.cov_matrix = np.cov(returns_data, rowvar=False)
        self.cov_matrix = self._calculate_cov_matrix(returns_data)
        self.lambda_max = np.max(np.linalg.eigvalsh(self.cov_matrix))

        # Calculate lower bound C for τ (VaR)
        self.C = self._calculate_C()

        # Binary encoding parameters (can be adjusted based on precision needs)

        self.J = 12  # bits for ξ₀
        self.K = 6  # bits for τ
        self.L = 6  # bits for y_t

        print(f"Model initialized:")
        print(f"  Assets: {self.d}, Observations: {self.T}")
        print(f"  Budget: {self.B}, Risk-free rate: {self.r}")
        print(f"  Lower bound C: {self.C:.4f}")
        print(f"  Binary encoding bits - ξ₀: {self.J}, τ: {self.K}, y_t: {self.L}")

    def _calculate_cov_matrix(self, returns_data: np.ndarray) -> np.ndarray:
        """
        Calculate the covariance matrix Cov(R) from Proposition 5.3.
        """
        cov_R = np.cov(returns_data, rowvar=False)

        # Create a (d+1)x(d+1) zero matrix
        cov_matrix = np.zeros((self.d + 1, self.d + 1))

        # Place Cov_R in the bottom-right block
        cov_matrix[1:, 1:] = cov_R
        return cov_matrix

    def _calculate_C(self) -> float:
        """
        Calculate the lower bound C for τ (VaR) from Proposition 5.3.
        C = -rB + Σ[1_{E[R_i] > rπ_i}](rπ_i - E[R_i]) - √((1-alpha) / alpha * λ_max(B² + d))
        """
        # First term: -rB
        term1 = -self.r * self.B

        # Second term: sum over assets where expected return exceeds risk-free return
        term2 = 0.0
        for i in range(self.d):
            if self.expected_returns[i] > self.r * self.prices[i]:
                term2 += self.r * self.prices[i] - self.expected_returns[i]

        # Third term: -√((1-alpha)/alpha)√(λ_max(B² + d))
        term3 = -np.sqrt((1 - self.alpha) / self.alpha) * np.sqrt(self.lambda_max * (self.B ** 2 + self.d))

        return term1 + term2 + term3

    def _get_variable_name(self, var_type: str, indices: Tuple) -> str:
        """Generate variable names for QUBO formulation."""
        if var_type == 'xi':
            return f"xi_{indices[0]}"
        elif var_type == 'a':
            return f"a_{indices[0]}"
        elif var_type == 'b':
            return f"b_{indices[0]}"
        elif var_type == 'c':
            return f"c_{indices[0]}_{indices[1]}"
        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    def build_qubo(self, penalty_budget: float = 1000.0,
                   lambda1_penalty: float = 10.0, lambda2_penalty: float = 100.0) -> dimod.BinaryQuadraticModel:
        """
        Build the QUBO model following equation (5.15) from the thesis.

        Args:
            penalty_budget: Penalty coefficient P for budget constraint
            lambda1_penalty: λ₁ parameter for unbalanced penalization
            lambda2_penalty: λ₂ parameter for unbalanced penalization

        Returns:
            dimod.BinaryQuadraticModel: The QUBO model
        """
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, 'BINARY')

        # Add variables
        variables = self._add_variables(bqm)

        # Add objective terms
        self._add_return_terms(bqm, variables)
        self._add_risk_terms(bqm, variables)
        self._add_budget_constraint(bqm, variables, penalty_budget)
        self._add_unbalanced_penalties(bqm, variables, lambda1_penalty, lambda2_penalty)

        print(f"QUBO model built with {len(bqm.variables)} variables")
        return bqm

    def _add_variables(self, bqm: dimod.BinaryQuadraticModel) -> Dict:
        """Add all binary variables to the model."""
        variables = {
            'xi': [],  # Binary portfolio weights for risky assets
            'a': [],  # Binary encoding for ξ₀
            'b': [],  # Binary encoding for τ
            'c': []  # Binary encoding for y_t
        }

        # Add binary portfolio variables ξᵢ for i = 1, ..., d
        for i in range(self.d):
            var_name = self._get_variable_name('xi', (i,))
            variables['xi'].append(var_name)
            bqm.add_variable(var_name, 0.0)

        # Add binary encoding variables for ξ₀
        for j in range(self.J):
            var_name = self._get_variable_name('a', (j,))
            variables['a'].append(var_name)
            bqm.add_variable(var_name, 0.0)

        # Add binary encoding variables for τ
        for k in range(self.K):
            var_name = self._get_variable_name('b', (k,))
            variables['b'].append(var_name)
            bqm.add_variable(var_name, 0.0)

        # Add binary encoding variables for y_t
        for t in range(self.T):
            variables['c'].append([])
            for l in range(self.L):
                var_name = self._get_variable_name('c', (t, l))
                variables['c'][t].append(var_name)
                bqm.add_variable(var_name, 0.0)

        return variables

    def _add_return_terms(self, bqm: dimod.BinaryQuadraticModel, variables: Dict):
        """Add return maximization terms to objective."""
        # Term: r * (B/(2^J - 1)) * Σ(2^j * a_j) + ξ · E[R]

        # Linear terms for ξ₀ encoding (risk-free asset)
        coeff_xi0 = self.r * self.B / (2 ** self.J - 1)
        for j in range(self.J):
            bqm.add_variable(variables['a'][j], coeff_xi0 * (2 ** j))

        # Linear terms for risky assets
        for i in range(self.d):
            bqm.add_variable(variables['xi'][i], self.expected_returns[i])

    def _add_risk_terms(self, bqm: dimod.BinaryQuadraticModel, variables: Dict):
        """Add ES risk terms to objective."""
        # Term: -λ[C + (B-C)/(2^K-1) * Σ(2^k * b_k) + (1/(1-α)) * (1/T) * Σ((B-C)/(2^L-1) * Σ(2^l * c_{t,l}))]

        # Constant term: -λ * C
        bqm.offset += -self.lambda_risk * self.C

        # Linear terms for τ encoding
        coeff_tau = -self.lambda_risk * (self.B - self.C) / (2 ** self.K - 1)
        for k in range(self.K):
            bqm.add_variable(variables['b'][k], coeff_tau * (2 ** k))

        # Linear terms for y_t encoding
        coeff_y = -self.lambda_risk * (1 / (1 - self.alpha)) * (1 / self.T) * (self.B - self.C) / (2 ** self.L - 1)
        for t in range(self.T):
            for l in range(self.L):
                bqm.add_variable(variables['c'][t][l], coeff_y * (2 ** l))

    def _add_budget_constraint(self, bqm: dimod.BinaryQuadraticModel, variables: Dict, penalty: float):
        """Add budget constraint penalty term."""
        # Penalty: -P * (B/(2^J-1) * Σ(2^j * a_j) + ξ · π - B)²

        scale_xi0 = self.B / (2 ** self.J - 1)

        # Quadratic terms: (scale_xi0 * Σ(2^j * a_j))²
        for j1 in range(self.J):
            for j2 in range(self.J):
                coeff = -penalty * scale_xi0 ** 2 * (2 ** j1) * (2 ** j2)
                if j1 == j2:
                    bqm.add_variable(variables['a'][j1], coeff)
                else:
                    bqm.add_interaction(variables['a'][j1], variables['a'][j2], coeff)

        # Quadratic terms: (ξ · π)²
        for i1 in range(self.d):
            for i2 in range(self.d):
                coeff = -penalty * self.prices[i1] * self.prices[i2]
                if i1 == i2:
                    bqm.add_variable(variables['xi'][i1], coeff)
                else:
                    bqm.add_interaction(variables['xi'][i1], variables['xi'][i2], coeff)

        # Cross terms: 2 * (scale_xi0 * Σ(2^j * a_j)) * (ξ · π)
        for j in range(self.J):
            for i in range(self.d):
                coeff = -2 * penalty * scale_xi0 * (2 ** j) * self.prices[i]
                bqm.add_interaction(variables['a'][j], variables['xi'][i], coeff)

        # Linear terms: -2B * (scale_xi0 * Σ(2^j * a_j) + ξ · π)
        for j in range(self.J):
            coeff = 2 * penalty * self.B * scale_xi0 * (2 ** j)
            bqm.add_variable(variables['a'][j], coeff)

        for i in range(self.d):
            coeff = 2 * penalty * self.B * self.prices[i]
            bqm.add_variable(variables['xi'][i], coeff)

        # Constant term: -P * B²
        bqm.offset += -penalty * self.B ** 2

    def _add_unbalanced_penalties(self, bqm: dimod.BinaryQuadraticModel, variables: Dict,
                                  lambda1: float, lambda2: float):
        """Add unbalanced penalization terms for inequality constraints."""
        # For each constraint t: y_t ≥ -ξ̄ · r̄_t - τ
        # Penalty: - Σ_t [-λ₁ * h*_t + λ₂ * (h*_t)²]
        # where h*_t = y*_t + ξ̄* · r̄_t + τ*

        scale_xi0 = self.B / (2 ** self.J - 1)
        scale_tau = (self.B - self.C) / (2 ** self.K - 1)
        scale_y = (self.B - self.C) / (2 ** self.L - 1)

        for t in range(self.T):

            # Linear terms from λ₁ * h*_t
            # y_t terms
            for l in range(self.L):
                coeff = lambda1 * scale_y * (2 ** l)
                bqm.add_variable(variables['c'][t][l], coeff)

            # ξ₀ terms (a_j variables)
            for j in range(self.J):
                coeff = lambda1 * scale_xi0 * (2 ** j) * self.r
                bqm.add_variable(variables['a'][j], coeff)

            # ξᵢ terms (xi variables)
            for i in range(self.d):
                coeff = lambda1 * self.returns_data[t, i]
                bqm.add_variable(variables['xi'][i], coeff)

            # τ terms (b_k variables)
            for k in range(self.K):
                coeff = lambda1 * scale_tau * (2 ** k)
                bqm.add_variable(variables['b'][k], coeff)

            # Constant term: λ₁ * C
            bqm.offset += lambda1 * self.C

            # Quadratic terms from - λ₂ * (h*_t)²
            # This involves all pairwise products of y_t, ξ₀, ξᵢ, and τ terms

            # y_t × y_t terms
            for l1 in range(self.L):
                for l2 in range(self.L):
                    coeff = - lambda2 * (scale_y ** 2) * (2 ** l1) * (2 ** l2)
                    if l1 == l2:
                        bqm.add_variable(variables['c'][t][l1], coeff)
                    else:
                        bqm.add_interaction(variables['c'][t][l1], variables['c'][t][l2], coeff)

            # ξ₀ × ξ₀ terms
            for j1 in range(self.J):
                for j2 in range(self.J):
                    coeff = - lambda2 * (scale_xi0 ** 2) * (2 ** j1) * (2 ** j2) * (self.r ** 2)
                    if j1 == j2:
                        bqm.add_variable(variables['a'][j1], coeff)
                    else:
                        bqm.add_interaction(variables['a'][j1], variables['a'][j2], coeff)

            # ξᵢ × ξᵢ terms
            for i1 in range(self.d):
                for i2 in range(self.d):
                    coeff = - lambda2 * self.returns_data[t, i1] * self.returns_data[t, i2]
                    if i1 == i2:
                        bqm.add_variable(variables['xi'][i1], coeff)
                    else:
                        bqm.add_interaction(variables['xi'][i1], variables['xi'][i2], coeff)

            # τ × τ terms
            # Constant term: - λ₂ * C²
            bqm.offset += - lambda2 * (self.C ** 2)

            # - λ₂ * 2 * C * scale_τ * Σ_k 2^k b_k
            for k in range(self.K):
                coeff = - 2 * lambda2 * self.C * scale_tau * (2 ** k)
                bqm.add_variable(variables['b'][k], coeff)

            # - λ₂ * scale_τ² * Σ_k₁ Σ_k₂ 2^k₁ 2^k₂ b_k₁ b_k₂
            for k1 in range(self.K):
                for k2 in range(self.K):
                    coeff = - lambda2 * (scale_tau ** 2) * (2 ** k1) * (2 ** k2)
                    if k1 == k2:
                        bqm.add_variable(variables['b'][k1], coeff)
                    else:
                        bqm.add_interaction(variables['b'][k1], variables['b'][k2], coeff)

            # Cross terms: y_t × ξ₀, y_t × ξᵢ, y_t × τ, ξ₀ × ξᵢ, ξ₀ × τ, ξᵢ × τ

            # y_t × ξ₀ terms
            for l in range(self.L):
                for j in range(self.J):
                    coeff = - 2 * lambda2 * scale_y * scale_xi0 * (2 ** l) * (2 ** j) * self.r
                    bqm.add_interaction(variables['c'][t][l], variables['a'][j], coeff)

            # y_t × ξᵢ terms
            for l in range(self.L):
                for i in range(self.d):
                    coeff = - 2 * lambda2 * scale_y * (2 ** l) * self.returns_data[t, i]
                    bqm.add_interaction(variables['c'][t][l], variables['xi'][i], coeff)

            # y_t × τ terms
            for l in range(self.L):
                coeff = - 2 * lambda2 * self.C * scale_y * (2 ** l)
                bqm.add_variable(variables['c'][t][l], coeff)

            for l in range(self.L):
                for k in range(self.K):
                    coeff = - 2 * lambda2 * scale_y * scale_tau * (2 ** l) * (2 ** k)
                    bqm.add_interaction(variables['c'][t][l], variables['b'][k], coeff)

            # ξ₀ × ξᵢ terms
            for j in range(self.J):
                for i in range(self.d):
                    coeff = - 2 * lambda2 * scale_xi0 * (2 ** j) * self.r * self.returns_data[t, i]
                    bqm.add_interaction(variables['a'][j], variables['xi'][i], coeff)

            # ξ₀ × τ terms
            for j in range(self.J):
                coeff = - 2 * lambda2 * scale_xi0 * self.C * self.r * (2 ** j)
                bqm.add_variable(variables['a'][j], coeff)

            for j in range(self.J):
                for k in range(self.K):
                    coeff = - 2 * lambda2 * scale_xi0 * (2 ** j) * (2 ** k) * self.r * scale_tau
                    bqm.add_interaction(variables['a'][j], variables['b'][k], coeff)

            # ξᵢ × τ terms
            for i in range(self.d):
                coeff = - 2 * lambda2 * self.returns_data[t, i] * self.C
                bqm.add_variable(variables['xi'][i], coeff)

            for i in range(self.d):
                for k in range(self.K):
                    coeff = - 2 * lambda2 * self.returns_data[t, i] * scale_tau * (2 ** k)
                    bqm.add_interaction(variables['xi'][i], variables['b'][k], coeff)

    def _check_budget_constraint(self, xi_0: float, xi_risky: np.ndarray) -> float:
        """Check budget constraint violation."""
        total_budget = xi_0 + np.sum(xi_risky * self.prices)
        return abs(total_budget - self.B)

    def _check_es_constraints(self, xi_0: float, xi_risky: np.ndarray,
                              tau: float, y_values: np.ndarray) -> List[float]:
        """Check ES constraint violations."""
        violations = []
        for t in range(self.T):
            portfolio_return = self.r * xi_0 + np.dot(xi_risky, self.returns_data[t])
            violation = y_values[t] + portfolio_return + tau
            violations.append(min(0, violation))  # Only negative values are violations
        return violations

    def _calculate_portfolio_metrics(self, xi_0: float, xi_risky: np.ndarray) -> Dict:
        """Calculate portfolio returns and ES metrics."""
        portfolio_returns = self.r * xi_0 + np.dot(self.returns_data, xi_risky)
        sorted_returns = np.sort(portfolio_returns)
        var_index = int(self.alpha * self.T)
        es = -np.mean(sorted_returns[:var_index])

        return {
            'expected_return': self.r * xi_0 + np.sum(xi_risky * self.expected_returns),
            'empirical_var': -sorted_returns[var_index],
            'empirical_es': es
        }

    def decode_solution(self, sample: Dict) -> Dict:
        """
        Decode the binary solution back to portfolio weights and other variables.

        Args:
            sample: Dictionary containing binary variable assignments

        Returns:
            Dictionary with decoded portfolio weights and other variables
        """
        # Decode ξ₀ (non-risky asset weight)
        xi_0 = 0.0
        for j in range(self.J):
            var_name = self._get_variable_name('a', (j,))
            if sample.get(var_name, 0):
                xi_0 += (2 ** j) * self.B / (2 ** self.J - 1)

        # Decode ξᵢ (risky asset weights)
        xi_risky = np.zeros(self.d)
        for i in range(self.d):
            var_name = self._get_variable_name('xi', (i,))
            xi_risky[i] = sample.get(var_name, 0)

        # Decode τ (VaR)
        tau = self.C
        for k in range(self.K):
            var_name = self._get_variable_name('b', (k,))
            if sample.get(var_name, 0):
                tau += (2 ** k) * (self.B - self.C) / (2 ** self.K - 1)

        # Decode y_t values
        y_values = np.zeros(self.T)
        for t in range(self.T):
            for l in range(self.L):
                var_name = self._get_variable_name('c', (t, l))
                if sample.get(var_name, 0):
                    y_values[t] += (2 ** l) * (self.B - self.C) / (2 ** self.L - 1)

        # Check constraint violations and calculate metrics
        budget_violation = self._check_budget_constraint(xi_0, xi_risky)
        es_violations = self._check_es_constraints(xi_0, xi_risky, tau, y_values)
        portfolio_metrics = self._calculate_portfolio_metrics(xi_0, xi_risky)

        return {
            'xi_0': xi_0,
            'xi_risky': xi_risky,
            'tau': tau,
            'y_values': y_values,
            'total_budget_used': xi_0 + np.sum(xi_risky * self.prices),
            'budget_violation': budget_violation,
            'max_es_violation': min(0, min(es_violations)) if es_violations else 0,
            'avg_es_violation': np.mean([v for v in es_violations if v < 0]) if es_violations else 0,
            'expected_return': portfolio_metrics['expected_return'],
            'empirical_var': portfolio_metrics['empirical_var'],
            'empirical_es': portfolio_metrics['empirical_es']
        }


# Example usage
def create_sample_data():
    """Create sample returns data for testing."""
    np.random.seed(42)
    T, d = 5, 5  # 50 observations, 5 assets

    # Generate some realistic return data
    returns = np.random.multivariate_normal(
        mean=[0.08, 0.10, 0.12, 0.07, 0.09],  # Expected returns
        cov=np.array([
            [0.04, 0.01, 0.02, 0.01, 0.01],
            [0.01, 0.06, 0.01, 0.02, 0.01],
            [0.02, 0.01, 0.08, 0.01, 0.02],
            [0.01, 0.02, 0.01, 0.03, 0.01],
            [0.01, 0.01, 0.02, 0.01, 0.05]
        ]) * 0.01,  # Scale down covariance
        size=T
    )

    # Asset prices
    prices = np.array([10.0, 15.0, 20.0, 12.0, 18.0])

    return returns, prices


if __name__ == "__main__":
    # # Create sample data
    # returns_data, prices = create_sample_data()

    # returns_data = pd.read_parquet("../data/processed/sp500_last255_10day_returns.parquet")
    # symbols = list(returns_data.columns)
    # returns_data = returns_data.to_numpy()

    # weightings = pd.read_csv("../data/raw/sp500_weightings.csv")
    # weightings = weightings.sort_values("Ticker", key=lambda s: s.astype(str).str.casefold()).reset_index(drop=True)
    # prices = list(weightings.Weight)

    df = pd.read_parquet("../data/processed/sp500_last255_10day_returns.parquet")
    symbols, returns_data = df.columns.tolist(), df.to_numpy()

    prices = pd.read_csv("../data/raw/sp500_weightings.csv") \
        .sort_values("Ticker", key=lambda s: s.astype(str).str.casefold(), ignore_index=True)["Weight"] \
        .tolist()

    # Initialize the model
    model = QUBOMeanESPortfolio(
        returns_data=returns_data,
        prices=prices,
        risk_free_rate=0.02,
        budget=100.0,
        alpha=0.975,
        lambda_risk=0
    )

    # Build the QUBO model
    bqm = model.build_qubo(
        penalty_budget=1000.0,
        lambda1_penalty=10.0,
        lambda2_penalty=100.0
    )

    print(f"\nQUBO model statistics:")
    print(f"  Variables: {len(bqm.variables)}")
    print(f"  Quadratic interactions: {len(bqm.quadratic)}")
    print(f"  Offset: {-bqm.offset:.4f}")

    # Create the sampler
    sampler = neal.SimulatedAnnealingSampler()
    print("Done with the initialization of the sampler. Running the solver now.")

    # Run the sampler
    sampleset = sampler.sample(-bqm, num_reads=100)
    print("The problem is run on the simulated annealing solver")

    # # Example: solve with simulated annealing (for demonstration)
    # sampler = dimod.SimulatedAnnealingSampler()
    # print("Done with the initialization of the sampler. Running the solver now.")
    # sampleset = sampler.sample(bqm, num_reads=10)
    # print("The problem is run on the simulated annealing solver")

    # # Initialize D-Wave sampler with automatic embedding
    # sampler = EmbeddingComposite(DWaveSampler())
    #
    # # Run on quantum annealer
    # sampleset = sampler.sample(bqm,
    #                            num_reads=1,
    #                            label='Portfolio Optimization')
    # print("The problem is run on the D-Wave system")

    # # Initialize hybrid solver
    # sampler = LeapHybridSampler()
    #
    # # Run using hybrid solver
    # sampleset = sampler.sample(bqm,
    #                            label='Portfolio Optimization - Hybrid',
    #                            time_limit=10)
    # print("The problem is run on the D-Wave hybrid solver")

    # Get best solution
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy

    print(f"\nBest solution energy: {best_energy:.4f}")

    # Decode solution
    decoded = model.decode_solution(best_sample)
    print(f"\nDecoded solution:")
    print(f"  Risk-free asset weight: {decoded['xi_0']:.4f}")
    print(f"  Risky asset weights: {decoded['xi_risky']}")
    print(f"  VaR (τ): {decoded['tau']:.4f}")
    print(f"  Total budget used: {decoded['total_budget_used']:.4f}")
    print(f"\nConstraint violations:")
    print(f"  Budget violation: {decoded['budget_violation']:.4f}")
    print(f"  Max ES violation: {decoded['max_es_violation']:.4f}")
    print(f"  Average ES violation: {decoded['avg_es_violation']:.4f}")
    print(f"\nPortfolio metrics:")
    print(f"  Expected return: {decoded['expected_return']:.4f}")
    print(f"  Empirical VaR: {decoded['empirical_var']:.4f}")
    print(f"  Empirical ES: {decoded['empirical_es']:.4f}")
