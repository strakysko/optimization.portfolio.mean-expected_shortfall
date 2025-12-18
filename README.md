# Mean–Expected Shortfall Portfolio Optimization on a Quantum Annealer (Case Study)

This repository contains a **Python reference implementation** of the thesis case study: translating a **Mean–Expected Shortfall (Mean–ES)** portfolio optimization problem into a **QUBO** and executing it on **D-Wave quantum annealing hardware** via the **Ocean SDK**.

The case study is positioned as a **feasibility demonstration** (i.e., the QUBO runs on real hardware) rather than a calibrated, investment-ready workflow.

---

## What this implements

### Optimization model (high level)
- Objective: balance **expected return** against **Expected Shortfall (ES)** risk using a Mean–ES formulation, where ES is made tractable through the Rockafellar–Uryasev reformulation and auxiliary variables.
- QUBO construction: the case study implements the **unbalanced-penalization** QUBO formulation (Equation (4.16) in the thesis). 
- Constraint handling:  
  - Budget constraint enforced via a quadratic penalty term.
  - Inequality constraints handled using **unbalanced penalization** to reduce qubit usage (at the cost of slight penalization even when constraints hold).

### Portfolio representation used to fit current hardware limits
To keep qubit usage tractable, the case study uses an **inclusion/exclusion** portfolio:
- Decision vector: $\xi \in {0,1}^d$ (select or exclude each asset). 
- Budget: $B = 100$.
- Initial “prices” $\pi_i$ are set to the **S&P 500 index weights** (a simplifying assumption made for the case study).

---

## Dataset (case study configuration)

- Universe: **S&P 500 constituents**; as of **August 1, 2025**, the index had **d = 503** constituents.
- Source: **Stooq** daily close prices.
- Window: last **265 trading days** from **July 11, 2024** to **July 31, 2025**. 
- Adjustments: prices are adjusted for **stock splits** and **cash dividends**. 

---

## Basel-style ES settings used in the case study

The case study follows Basel-style parameter choices:
- ES computed from a **12-month** period. 
- Confidence level: $\alpha = 0.975$. 
- Horizon: **10 trading days**.
- Returns: uses **overlapping 10-day returns**, with **T = 255** overlapping observations per constituent.

---

## Results status (important)

The thesis reports that:
- The implementation produces “seemingly meaningful” portfolios, but
- It **does not validate** outputs and **does not calibrate** key hyperparameters due to time constraints, so the results **should not be interpreted as financially meaningful**.
- The main takeaway is **feasibility**: running the Mean–ES QUBO on a real D-Wave annealer.

---
