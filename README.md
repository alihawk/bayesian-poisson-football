# Bayesian Poisson Regression on Football Match Data

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A two-part Bayesian analysis of total goals in football matches:

1. **MCMC** Poisson GLM with PyMC/NUTS
2. **Laplace Approximation**: fast Gaussian substitute + rapid predictions

---

## 📂 Repository Structure

```
.
├── code/                   # Python scripts
│   ├── hw.py              # Main analysis pipeline
│   └── test.py            # Scratch / experiments
├── data/                   # (Optional) CSV data
│   └── football.csv
├── figures/                # Diagnostic plots (SVG/PNG)
├── report/                 # Final PDF report
├── README.md               # Project overview & usage
└── .gitignore             # Exclusions
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip install -r requirements.txt (create one from your environment)

### Running the Analysis

```bash
cd code
python hw.py
```

This will:

1. **Load & clean** the data
2. **Build** design matrices
3. **Fit** Poisson GLM via MCMC (4 chains, NUTS)
4. **Derive** Laplace Approximation (MAP + Hessian)
5. **Validate** Laplace vs. MCMC
6. **Predict** on 100 held-out matches (MSE, MAE, exact-count)

## 📈 Results

* **Posterior summaries** for each β (means, sds, 94% HDIs, ESS, R̂)
* **Laplace MAP** & covariance match MCMC within ±0.004
* **Predictions** hover around 2–3 goals per match under 3 loss rules

See the full write-up in `report/Homework6_DS.pdf`.

## 💡 Contributing

Contributions welcome! Please open an issue or pull request.

## 📄 License

This project is licensed under the MIT License. See LICENSE.
