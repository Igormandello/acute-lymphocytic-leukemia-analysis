import pandas as pd

from model.optimization import EvalResults

def bold_max(s):
    is_max = s == s.max()
    return ["\\textbf{" + str(v) + "}" if m else str(v) for v, m in zip(s, is_max)]

def generate_models_table(model_header: str, model_results: dict[str, EvalResults]):
    results = []
    for name, model_result in model_results.items():
        metrics = {}
        metrics[model_header] = name
        metrics["F1-W"] = model_result.f1_weighted
        metrics["F1-M"] = model_result.f1_macro
        metrics["Recall"] = model_result.recall_micro
        metrics["BAcc."] = model_result.balanced_accuracy
        metrics["Accuracy"] = model_result.accuracy
        results.append(metrics)

    df = pd.DataFrame(results)
    df.iloc[:, 1:] = df.iloc[:, 1:] * 100
    df.iloc[:, 1:] = df.iloc[:, 1:].round(2)

    styled_df = df.copy()
    for col in df.columns[1:]:
        styled_df[col] = bold_max(df[col])

    styled_df.columns = [f"\\textbf{{{col}}}" for col in styled_df.columns]
    latex_table = styled_df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * (len(styled_df.columns) - 1)
    )

    latex_table = latex_table.replace("\\toprule", "\\hline") \
                            .replace("\\midrule", "\\hline") \
                            .replace("\\bottomrule", "\\hline")

    return latex_table

def generate_one_vs_all_feature_table(importances, feature_count_by_subtype, acc_by_subtype, feature_name="gene", importance_name="importance", importance_scale=1000):
    def format_features(features, k):
        return ", ".join(
            f"{g[feature_name]} ({(g[importance_name] * importance_scale):.2f})"
            for g in features[:k]
        )

    lines = []

    # Header
    lines.append("\\begin{tabular}{l c l p{9.5cm}}")
    lines.append("\\hline")
    lines.append(f"\\textbf{{Subtype}} & \\textbf{{Max BAcc. (\\%)}} & \\textbf{{Sex}} & \\textbf{{Features ({{{importance_name}}} x {{{importance_scale}}})}} \\\\")
    lines.append("\\hline")

    for subtype in importances:
        bacc = acc_by_subtype[subtype]

        sexes = list(importances[subtype].keys())
        first = True

        for sex in sexes:
            featutes_str = format_features(
                importances[subtype][sex],
                feature_count_by_subtype[subtype]
            )

            if first:
                line = (
                    f"\\multirow{{{len(sexes)}}}{{*}}{{{subtype}}} "
                    f"& \\multirow{{{len(sexes)}}}{{*}}{{{bacc:.2f}}} "
                    f"& {sex} & {featutes_str} \\\\"
                )
                first = False
            else:
                line = f"& & {sex} & {featutes_str} \\\\"

            lines.append(line)

        lines.append("\\hline")

    lines.append("\\end{tabular}")

    return "\n".join(lines)