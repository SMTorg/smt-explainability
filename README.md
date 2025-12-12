SMT Explainability 🧠
Enhancing SMT with Explainable AI for Surrogate Models

This package is an extension to the SMT (Surrogate Modeling Toolbox), offering a comprehensive suite of Explainable AI (XAI) methods to interpret, visualize, and trust "black-box" surrogate models.
🔍 What It Does

    Model Agnostic Interpretation: Works with various surrogate models to reveal how predictions are made.

Key XAI Methods: Implements Shapley Additive Explanations (SHAP), Partial Dependence Plots (PDP), and Individual Conditional Expectations (ICE).

Mixed-Variable Support: Handles both continuous and categorical (mixed-integer) variables seamlessly in explanations.

Uncertainty Quantification: Features Split Conformal Prediction to provide robust, statistically valid confidence intervals.

Global Sensitivity Analysis: Computes Sobol indices for variance-based sensitivity analysis alongside visual explanations.

📖 Why It Matters

Surrogate models (like Kriging or Neural Networks) are essential for replacing expensive simulations, but they often act as opaque "black boxes". In critical engineering fields (aerospace, mechanics), high accuracy is not enough; engineers need to understand the underlying physical relationships learned by the model. This extension provides:

    Transparency: Dissects the input-output relationships to build trust in the model.

Insight Discovery: Detects non-linearities, interactions, and variable importance that might be missed by simple metrics.

Decision Support: Helps justify design choices by quantifying the contribution of each variable to a specific prediction.

🔗 Reference

This implementation is based on:

SMT-EX: An Explainable Surrogate Modeling Toolbox for Mixed-Variables Design Exploration

Mohammad D. Robani, Pramudita S. Palar, Lavi R. Zuhal, Paul Saves, and Joseph Morlier (Jan 2025). AIAA SciTech 2025 Forum. DOI: 10.2514/6.2025-0777

The paper details the methodology and validates the toolbox on engineering test cases like wing weight prediction and beam bending.

📦 Installation
Bash

pip install smt-explainability

Requirements: smt, numpy, matplotlib. See requirements.txt for details.

✅ Citation

If you use this package in research, please cite:
Code snippet

@inproceedings{robani2025smtex,
  title={SMT-EX: An Explainable Surrogate Modeling Toolbox for Mixed-Variables Design Exploration},
  author={Robani, Mohammad Daffa and Palar, Pramudita Satria and Zuhal, Lavi Rizki and Saves, Paul and Morlier, Joseph},
  booktitle={AIAA SCITECH 2025 Forum},
  year={2025},
  doi={10.2514/6.2025-0777}
}

📜 License

Distributed under the BSD-3-Clause license. See LICENSE for details.
