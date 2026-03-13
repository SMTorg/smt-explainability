# ⚙️ SMT Explainability

**Enhancing the Surrogate Modeling Toolbox (SMT) with Explainable AI**

*A lightweight extension bringing transparency, visualization, and robust uncertainty quantification to surrogate models in engineering design.*

---

## About

`SMT Explainability` is an extension for the
[Surrogate Modeling Toolbox (SMT)](https://smt.readthedocs.io/)
that provides a collection of **model-agnostic Explainable AI (XAI) methods**
specifically designed for surrogate models such as Kriging, RBF, polynomial models,
and neural networks.

The library focuses on **engineering design and simulation-based workflows**, with
native support for **mixed continuous and categorical variables**, which are common
in real-world optimization and surrogate modeling problems.

It enables users to analyze, interpret, and validate surrogate models using
state-of-the-art explainability techniques.

---

## Key Features

- **Model-Agnostic Explainability**  
  Works with any SMT surrogate model to understand predictions and model behavior.

- **SHAP (Shapley Additive Explanations)**  
  Compute local and global feature contributions for surrogate predictions.

- **PDP & ICE Plots**  
  Partial Dependence and Individual Conditional Expectation plots for marginal effects.

- **Mixed-Variable Support**  
  Handles continuous, integer, categorical, and mixed design spaces.

- **Uncertainty Quantification**  
  Split conformal prediction for distribution-free prediction intervals.

- **Global Sensitivity Analysis**  
  Sobol indices and variance-based importance measures.

- **Reproducible Sampling**
  Deterministic random generators using `seed` for reproducible experiments.

---

## Why It Matters

Surrogate models replace expensive simulations in engineering workflows,
but their black-box nature makes interpretation difficult.

`SMT Explainability` helps to:

- **Build trust** in surrogate predictions
- **Understand feature influence**
- **Detect nonlinear effects and interactions**
- **Support engineering decisions**
- **Validate surrogate accuracy**

---

## Installation

Install the latest release from PyPI:

```bash
pip install smt-explainability

Or install from source the development version:

```bash
git clone https://github.com/SMTorg/smt-explainability.git
cd smt-explainability
pip install -e .
```

## Requirements

See `requirements.txt` for full details. Minimal requirements:

smt >= 2.13.0

numpy >= 2.4.0

scipy >= 1.17.1

scikit-learn >= 1.8.0

matplotlib >= 3.10.0

Python >= 3.11
---


## Reference / Citation

If you use this package in research, please cite:

```
@inproceedings{robani2025smtex,
  title={SMT-EX: An Explainable Surrogate Modeling Toolbox for Mixed-Variables Design Exploration},
  author={Robani, Mohammad Daffa and Saves, Paul and Palar, Pramudita Satria and Zuhal, Lavi Rizki and Morlier, Joseph},
  booktitle={AIAA SCITECH 2025 Forum},
  year={2025},
  doi={10.2514/6.2025-0777}
}
```
A short description of the validation cases (wing weight prediction, beam bending) and methodology is contained in the paper.
---


## License
This project is distributed under the BSD-3 CLause License.
