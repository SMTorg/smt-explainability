# SMT Explainability 🧠

**Enhancing SMT with Explainable AI for Surrogate Models**

*A compact extension to the Surrogate Modeling Toolbox (SMT) that brings Explainable AI (XAI) techniques to surrogate models used in engineering: interpretation, visualization, and statistically-valid uncertainty.*

---


## About

`SMT Explainability` is an extension to the SMT (Surrogate Modeling Toolbox). It provides a collection of model-agnostic XAI methods tailored to surrogate models (Kriging, neural networks, etc.), with special care for mixed continuous/categorical inputs common in engineering design.

## Key Features

* **Model-agnostic interpretation** — Works with a wide range of surrogate models to reveal how predictions are produced.
* **SHAP (Shapley Additive Explanations)** — Local and global explanations using Shapley values.
* **PDP & ICE plots** — Partial Dependence and Individual Conditional Expectation visualizations for marginal effects and heterogeneity.
* **Mixed-variable support** — Handles continuous and categorical (including mixed-integer) variables.
* **Uncertainty quantification** — Split conformal prediction for robust, distribution-free prediction intervals.
* **Global sensitivity analysis** — Sobol indices for variance-based sensitivity and corresponding visual reports.

## Why It Matters

Surrogate models replace expensive simulations in engineering workflows, but they can act as opaque black boxes. `SMT Explainability` helps engineers:

* **Build trust** by making model decisions transparent.
* **Discover insight** such as non-linearities, interactions, and variable importance.
* **Support decisions** with quantified contributions (why the model predicted X for a given input).

---

## Installation

```bash
pip install smt-explainability
```

Or install from source:

```bash
git clone https://github.com/your-org/smt-explainability.git
cd smt-explainability
pip install -e .
```

## Requirements

See `requirements.txt` for full details. Minimal requirements:

* `smt` (Surrogate Modeling Toolbox)
* `numpy`
* `matplotlib`
* `scikit-learn`

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

