import os

replacements = {
    "smt.explainability_tools._partial_dependence": "smt_explainability.pdp",
    "smt.explainability_tools._pd_feature_importance": "smt_explainability.pdp",
    "smt.explainability_tools._plot.partial_dependence": "smt_explainability.pdp",
    "smt.explainability_tools._shap_values": "smt_explainability.shap",
    "smt.explainability_tools": "smt_explainability",
}

files = [
    "tutorial/development/dev_pdp_2d_mixed.ipynb",
    "tutorial/development/pdp_based_feature_importance.ipynb",
    "tutorial/development/pd_feature_importance_plot.ipynb",
    "tutorial/development/mixed_variables_smt_2.ipynb",
    "tutorial/development/test_partial_dependence.ipynb",
    "tutorial/development/pd_interaction_development_temp.ipynb",
    "tutorial/development/shap_development_categorical.ipynb",
    "tutorial/development/pd_interaction_development.ipynb",
    "tutorial/development/partial_dependence_plot.ipynb",
    "tutorial/development/partial_dependence_plot_2.ipynb",
    "tutorial/development/demo_shap_categorical.ipynb",
    "tutorial/development/demo_shap.ipynb",
    "tutorial/development/example_pd_interaction.ipynb",
    "tutorial/development/demo_pdp_mixed_cantilever.ipynb",
    "tutorial/development/demo_pdp_interaction.ipynb",
    "tutorial/development/demo_pdp_mixed_variables.ipynb",
    "tutorial/development/2d_pdp_plot_mixed.ipynb",
    "tutorial/development/demo_pdp.ipynb",
    "tutorial/development/demo/demo_shap_mixed_cantilever.ipynb",
    "tutorial/development/demo/demo_shap_wing_weight.ipynb",
    "tutorial/development/demo/demo_true_mixed_cantilever.ipynb",
    "tutorial/development/demo/demo_pdp_wing_weight.ipynb",
    "tutorial/development/demo/demo_pdp_mixed_cantilever.ipynb",
    "tutorial/development/_partial_dependence.ipynb",
    "tutorial/development/test_partial_dependence.py",
    "tutorial/development/2d_plot_mixed.py",
    "tutorial/development/demo/test_shap_mixed.py",
]

for file_path in files:
    if not os.path.exists(file_path):
        print(f"File {file_path} not found, skipping.")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # We replace the longest strings first to avoid partial replacements
    # BUT wait, since I already replaced some with 'smt_explainability',
    # I need to handle that.

    # Actually, it's safer to just replace any remaining 'smt.explainability_tools'
    # and also 'smt_explainability._partial_dependence' if they were partially replaced.

    # Let's just do a clean sweep:
    # Replace all old patterns with their new ones.
    # Also handle the ones I already changed to 'smt_explainability' but should be '.pdp' or '.shap'

    content = content.replace(
        "smt.explainability_tools._partial_dependence", "smt_explainability.pdp"
    )
    content = content.replace(
        "smt.explainability_tools._pd_feature_importance", "smt_explainability.pdp"
    )
    content = content.replace(
        "smt.explainability_tools._plot.partial_dependence", "smt_explainability.pdp"
    )
    content = content.replace(
        "smt.explainability_tools._shap_values", "smt_explainability.shap"
    )
    content = content.replace("smt.explainability_tools", "smt_explainability")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {file_path}")
