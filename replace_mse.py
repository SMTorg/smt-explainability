import json
import os
import re

files = [
    r"tutorial\development\demo_pdp_mixed_cantilever.ipynb",
    r"tutorial\development\new_kernel_shap_wing_weight.ipynb",
    r"tutorial\development\pd_feature_importance_plot.ipynb",
    r"tutorial\development\pd_interaction_development_temp.ipynb",
    r"tutorial\development\shap_development_categorical.ipynb",
    r"tutorial\development\shap_package_mixed_cantilever.ipynb",
    r"tutorial\development\shap_development.ipynb",
    r"tutorial\development\pdp_based_feature_importance.ipynb",
    r"tutorial\development\pd_interaction_development.ipynb",
    r"tutorial\development\partial_dependence_plot_2.ipynb",
    r"tutorial\development\new_version\cantilever.ipynb",
    r"tutorial\development\new_kernel_shap_cantilever.ipynb",
    r"tutorial\development\partial_dependence_plot.ipynb",
    r"tutorial\development\mixed_variables_smt.ipynb",
    r"tutorial\development\mixed_variables_smt_2.ipynb",
    r"tutorial\development\kernel_shap_development.ipynb",
    r"tutorial\development\dev_pdp_2d_mixed.ipynb",
    r"tutorial\development\demo_shap_categorical.ipynb",
    r"tutorial\development\example_pd_interaction.ipynb",
    r"tutorial\development\demo_shap.ipynb",
    r"tutorial\development\demo_pdp_interaction.ipynb",
    r"tutorial\development\demo_pdp_mixed_variables.ipynb",
    r"tutorial\development\new_version\wing_weight.ipynb",
    r"tutorial\development\demo_pdp.ipynb",
    r"tutorial\development\demo\demo_shap_wing_weight.ipynb",
    r"tutorial\development\demo\demo_shap_mixed_cantilever.ipynb",
    r"tutorial\development\demo\demo_true_mixed_cantilever.ipynb",
    r"tutorial\development\demo\demo_wing_weight.ipynb",
    r"tutorial\development\demo\demo_pdp_mixed_cantilever.ipynb",
    r"tutorial\development\demo\demo_pdp_wing_weight.ipynb",
    r"tutorial\development\demo\demo_feature_interaction.ipynb",
    r"tutorial\development\demo\aiaa_abstract_wing_weight.ipynb",
    r"tutorial\development\demo\demo_mixed_cantilever.ipynb",
    r"tutorial\development\2d_plot_mixed.py",
    r"tutorial\development\demo\aiaa_abstract_cantilever.ipynb",
    r"tutorial\development\2d_pdp_plot_mixed.ipynb",
]

# Pattern to match mean_squared_error(..., squared=False)
# We want to capture the arguments before squared=False
pattern = re.compile(r"mean_squared_error\(([^,)]+,\s*[^,)]+),\s*squared=False\)")


def replace_in_content(content):
    return pattern.sub(r"np.sqrt(mean_squared_error(\1))", content)


for file_path in files:
    # Convert backslashes to forward slashes for cross-platform compatibility if needed,
    # but since we are on Windows (based on paths), we keep them or normalize.
    normalized_path = file_path.replace("\\", "/")
    if not os.path.exists(normalized_path):
        print(f"File not found: {normalized_path}")
        continue

    print(f"Processing {normalized_path}...")

    if normalized_path.endswith(".ipynb"):
        with open(normalized_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        changed = False
        for cell in data.get("cells", []):
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                new_source = []
                for line in source:
                    new_line = replace_in_content(line)
                    if new_line != line:
                        changed = True
                    new_source.append(new_line)
                cell["source"] = new_source

        if changed:
            with open(normalized_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1)
            print(f"Updated {normalized_path}")
        else:
            print(f"No changes in {normalized_path}")

    elif normalized_path.endswith(".py"):
        with open(normalized_path, "r", encoding="utf-8") as f:
            content = f.read()

        new_content = replace_in_content(content)

        if new_content != content:
            with open(normalized_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated {normalized_path}")
        else:
            print(f"No changes in {normalized_path}")
