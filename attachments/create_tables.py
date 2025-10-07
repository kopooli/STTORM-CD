import os
import json

# ---------------- CONFIG ----------------
LOG_DIR = "test_metrics_logs"
OUTPUT_DIR = "latex_tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Datasets
disasters_ravaen = ["landslides", "fires", "hurricanes", "floods_ravaen"]
disasters_sttorm = ["floods"]

dataset_display_names = {
    "landslides": "RaVAEn-Landslides",
    "fires": "RaVAEn-Wildfires",
    "hurricanes": "RaVAEn-Hurricanes",
    "floods_ravaen": "RaVAEn-Floods",
    "floods": "STTORM-Floods"
}
# Models (internal names)
models = [
    "small_original", "medium_original", "large_original",
    "small_my_variable", "medium_my_variable", "large_my_variable",
    "small_my_fixed", "medium_my_fixed", "large_my_fixed",
    "Index", "Cos_baseline"
]

# Display names (same order as models)
display_names = [
    "RaVAEn (small)", "RaVAEn (medium)", "RaVAEn (large)",
    "STTORM-CD (small, variable)", "STTORM-CD (medium, variable)", "STTORM-CD (large, variable)",
    "STTORM-CD (small, fixed)", "STTORM-CD (medium, fixed)", "STTORM-CD (large, fixed)",
    "Index", "Cosine baseline"
]

# Metrics configuration
metrics_config = {
    "AURC": ["area_under_the_curve_one_overall", "area_under_the_curve_min_overall", "area_under_the_curve_avg_overall"],
    "RDP": ["downlink_amount_one_overall", "downlink_amount_min_overall", "downlink_amount_avg_overall"],
    "AUPRC": ["Tile-level AUPRC (one memory)_overall", "Tile-level AUPRC (min memory)_overall", "Tile-level AUPRC (avg memory)_overall"],
    #"AP": ["Custom MAP (one memory)_overall", "Custom MAP (min memory)_overall", "Custom MAP (avg memory)_overall"],
    "F1": ["Tile-level F1-Score (one memory)_overall", "Tile-level F1-Score (min memory)_overall", "Tile-level F1-Score (avg memory)_overall"],
    "Precision": ["Tile-level Precision (one memory)_overall", "Tile-level Precision (min memory)_overall", "Tile-level Precision (avg memory)_overall"],
    "Recall": ["Tile-level Recall (one memory)_overall", "Tile-level Recall (min memory)_overall", "Tile-level Recall (avg memory)_overall"]
}

PRIMARY_METRIC = "AURC"
SECONDARY_METRIC = "RDP"
MEMORY_STRATEGY = "min"

# ---------------- HELPERS ----------------
def latex_escape(s):
    """Escape underscores and other special LaTeX characters."""
    return s.replace("_", r"\_")

def load_metrics(json_path):
    if not os.path.exists(json_path):
        print(f"Warning: Missing JSON file {json_path}")
        return None
    with open(json_path, "r") as f:
        metrics = json.load(f)
    if isinstance(metrics, list):
        metrics = metrics[0]
    return metrics

def extract_metrics(metrics, metric_name):
    keys = metrics_config[metric_name]
    return [metrics.get(k, 0) for k in keys]

# ---------------- TABLE GENERATORS ----------------
def make_ravaen_table(all_metrics_per_dataset, memory_strategy=MEMORY_STRATEGY):
    """
    Creates a LaTeX table comparing Geo-index, baselines, RaVAEn variable models, 
    and STTORM-CD variable models across multiple datasets.
    Numbers are displayed as percentages with two decimals.
    """
    datasets = list(all_metrics_per_dataset.keys())
    
    col_format = "|l|" + "cc|" * len(datasets)
    header = (
        "\\begin{table}\n"
        "  \\centering\n"
        f"  \\begin{{tabular}}{{{col_format}}}\n"
        "    \\toprule\n"
        "    \\multicolumn{1}{|r|}{Dataset} "
        + "".join([f"& \\multicolumn{{2}}{{|r|}}{{{latex_escape(dataset_display_names[d])}}}" for d in datasets])
        + " \\\\\n"
        "    \\multicolumn{1}{|r|}{Tiles count} "
        + "".join([f"& \\multicolumn{{2}}{{|r|}}{{{int(all_metrics_per_dataset[d].get('total_tiles',0))}}}" for d in datasets])
        + " \\\\\n"
        "    \\multicolumn{1}{|r|}{Changed tiles [\\%]} "
        + "".join([f"& \\multicolumn{{2}}{{|r|}}{{{all_metrics_per_dataset[d].get('changed_percent',0):.2f}}}" for d in datasets])
        + " \\\\\n"
        "    \\multicolumn{1}{|r|}{Metric} "
        + "".join([f"& {PRIMARY_METRIC}~\%~$\\uparrow$ & {SECONDARY_METRIC}~\%~$\\downarrow$" for _ in datasets])
        + " \\\\\n"
        "    \\midrule\n"
    )

    # All models including baselines
    baseline_models = ["Index", "Cos_baseline"]
    ravaen_models = ["small_original", "medium_original", "large_original"]
    sttorm_models = ["small_my_variable", "medium_my_variable", "large_my_variable"]
    order = baseline_models + ravaen_models + sttorm_models
    display_order = ["Geo-index", "Cosine baseline",
                     "RaVAEn -- small", "RaVAEn -- medium", "RaVAEn -- large",
                     "STTORM-CD -- small", "STTORM-CD -- medium", "STTORM-CD -- large"]

    # Compute best per group per dataset for bolding
    mem_idx = {"one": 0, "min": 1, "avg": 2}[memory_strategy]
    best_primary = {d: {} for d in datasets}
    best_secondary = {d: {} for d in datasets}
    for d in datasets:
        best_primary[d]["ravaen"] = max(all_metrics_per_dataset[d][k].get(PRIMARY_METRIC,[0,0,0])[mem_idx]*100 for k in ravaen_models)
        best_secondary[d]["ravaen"] = max(all_metrics_per_dataset[d][k].get(SECONDARY_METRIC,[0,0,0])[mem_idx]*100 for k in ravaen_models)
        best_primary[d]["sttorm"] = max(all_metrics_per_dataset[d][k].get(PRIMARY_METRIC,[0,0,0])[mem_idx]*100 for k in sttorm_models)
        best_secondary[d]["sttorm"] = max(all_metrics_per_dataset[d][k].get(SECONDARY_METRIC,[0,0,0])[mem_idx]*100 for k in sttorm_models)

    rows = ""
    for i, model_key in enumerate(order):
        disp_name = display_order[i]
        group = None
        if model_key in ravaen_models:
            group = "ravaen"
        elif model_key in sttorm_models:
            group = "sttorm"

        row = disp_name
        for d in datasets:
            metrics = all_metrics_per_dataset[d].get(model_key, {})
            p_val = metrics.get(PRIMARY_METRIC, [0,0,0])[mem_idx]*100
            s_val = metrics.get(SECONDARY_METRIC, [0,0,0])[mem_idx]*100

            # Bold best only for RaVAEn and STTORM groups
            if group == "ravaen":
                p_str = f"\\textbf{{{p_val:.2f}}}" if p_val == best_primary[d][group] else f"{p_val:.2f}"
                s_str = f"\\textbf{{{s_val:.2f}}}" if s_val == best_secondary[d][group] else f"{s_val:.2f}"
            elif group == "sttorm":
                p_str = f"\\textbf{{{p_val:.2f}}}" if p_val == best_primary[d][group] else f"{p_val:.2f}"
                s_str = f"\\textbf{{{s_val:.2f}}}" if s_val == best_secondary[d][group] else f"{s_val:.2f}"
            else:
                # baselines: no bold
                p_str = f"{p_val:.2f}"
                s_str = f"{s_val:.2f}"

            row += f" & {p_str} & {s_str}"
        rows += f"    {row} \\\\\n"

        # Midrules after baselines and after RaVAEn
        if model_key == baseline_models[-1] or model_key == ravaen_models[-1]:
            rows += "    \\midrule\n"

    footer = (
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "  \\caption{Reevaluating RaVAEn. The final change predictions were derived by using $\min()$ on the change predictions from the memory. In RaVAEn-Floods, the geo-index used was NDWI; in RaVAEn-Wildfires, the NBR; and in RaVAEn-Landslides and RaVAEn-Hurricanes, NDVI. Highlighted are the best scores for STTORM-CD and RaVAEn models for each metric. *Used as a validation dataset.}\n"
        "  \\label{tab:ravaen-revisited-combined}\n"
        "\\end{table}\n"
    )

    return header + rows + footer

def make_sttorm_table(all_metrics):
    col_format = "|l|ccc|ccc|"
    header = (
        f"\\begin{{table}}\n"
        f"  \\centering\n"
        f"  \\begin{{tabular}}{{{col_format}}}\n"
        f"    \\toprule\n"
        f"    \\multicolumn{{1}}{{|c|}}{{Method}} & "
        f"\\multicolumn{{3}}{{c|}}{{{PRIMARY_METRIC}~[\%]~$\\uparrow$}} & "
        f"\\multicolumn{{3}}{{c|}}{{{SECONDARY_METRIC}~[\%]~$\\downarrow$}} \\\\\n"
        f"    \\cmidrule(r){{2-4}} \\cmidrule(l){{5-7}}\n"
        f"    & Most recent & $min$(memory) & $avg$(memory) & Most recent & $min$(memory) & $avg$(memory) \\\\\n"
        f"    \\midrule\n"
    )

    order = [
        ("Index", "Geo-Index"),   
        ("Cos_baseline", "Cosine baseline"),
        ("small_original", "RaVAEn -- small"),
        ("medium_original", "RaVAEn -- medium"),
        ("large_original", "RaVAEn -- large"),
        ("small_my_variable", "STTORM-CD -- small"),
        ("medium_my_variable", "STTORM-CD -- medium"),
        ("large_my_variable", "STTORM-CD -- large"),
    ]

    # Groups for bolding
    ravaen_keys = ["small_original", "medium_original", "large_original"]
    sttorm_keys = ["small_my_variable", "medium_my_variable", "large_my_variable"]

    # Determine best/worst depending on metric direction
    best_primary_ravaen = [max(all_metrics[k].get(PRIMARY_METRIC, [0,0,0])[i] for k in ravaen_keys) for i in range(3)]
    best_primary_sttorm = [max(all_metrics[k].get(PRIMARY_METRIC, [0,0,0])[i] for k in sttorm_keys) for i in range(3)]
    # For SECONDARY_METRIC (down-arrow), lower is better
    best_secondary_ravaen = [min(all_metrics[k].get(SECONDARY_METRIC, [0,0,0])[i] for k in ravaen_keys) for i in range(3)]
    best_secondary_sttorm = [min(all_metrics[k].get(SECONDARY_METRIC, [0,0,0])[i] for k in sttorm_keys) for i in range(3)]

    rows = ""
    for model_key, disp_name in order:
        metrics = all_metrics.get(model_key, {})
        p_vals = metrics.get(PRIMARY_METRIC, [0,0,0])
        s_vals = metrics.get(SECONDARY_METRIC, [0,0,0])

        # Bold best scores per group
        p_bold = [
            f"\\textbf{{{v*100:.2f}}}" if 
            (model_key in ravaen_keys and v == best_primary_ravaen[i]) or
            (model_key in sttorm_keys and v == best_primary_sttorm[i]) else f"{v*100:.2f}" 
            for i, v in enumerate(p_vals)
        ]
        s_bold = [
            f"\\textbf{{{v*100:.2f}}}" if 
            (model_key in ravaen_keys and v == best_secondary_ravaen[i]) or
            (model_key in sttorm_keys and v == best_secondary_sttorm[i]) else f"{v*100:.2f}" 
            for i, v in enumerate(s_vals)
        ]

        row = f"{disp_name} & {p_bold[0]} & {p_bold[1]} & {p_bold[2]} & {s_bold[0]} & {s_bold[1]} & {s_bold[2]}"
        rows += f"    {row} \\\\\n"

        # Insert midrules between groups
        if model_key == "Cos_baseline" or model_key == "large_original":
            rows += "    \\midrule\n"

    footer = (
        f"    \\bottomrule\n"
        f"  \\end{{tabular}}\n"
        f"  \\caption{{Comparison of approaches' performance on the test set from the STTORM-CD-Floods dataset. "
        f"For RDP interpretation, it is important to mention that the total number of tiles in this dataset is 6,678, "
        f"and 22.34\\,\\% of them contain the defined change ($>5\\,\\%$ of pixels changed). "
        f"The ``Most recent'' column represents results from comparing the changed tile with only the most recent non-cloudy tile. "
        f"In other columns, all non-cloudy tiles from memory were used to derive the final change predictions using the $min()$ or $average()$ function. "
        f"Highlighted are the best scores for STTORM-CD models and RaVAEn models for easy comparison.}}\n"
        f"\\label{{tab:my-results-combined}}\n"
        f"\\end{{table}}\n"
    )

    return header + rows + footer

# ---------------- MAIN ----------------
def process_ravaen_tables():
    all_metrics_per_dataset = {}
    for disaster in disasters_ravaen:
        folder = f"{disaster}"
        dataset_metrics = {"total_tiles": 0, "changed_percent": 0}
        models_metrics = {}
        for model in models:
            filename = f"{model}_test_metrics.json"
            path = os.path.join(LOG_DIR, folder, filename)
            metrics = load_metrics(path)
            if metrics is None:
                continue
            if dataset_metrics["total_tiles"] == 0:
                dataset_metrics["total_tiles"] = metrics.get("Total N of tiles_overall", 0)
                dataset_metrics["changed_percent"] = metrics.get("percentage_of_changed_overall", 0)*100
            models_metrics[model] = {m: extract_metrics(metrics, m) for m in [PRIMARY_METRIC, SECONDARY_METRIC]}
        dataset_metrics.update(models_metrics)
        all_metrics_per_dataset[disaster] = dataset_metrics

    latex_str = make_ravaen_table(all_metrics_per_dataset)
    out_path = os.path.join(OUTPUT_DIR, f"ravaen_comparison_{MEMORY_STRATEGY}.tex")
    with open(out_path, "w") as f:
        f.write(latex_str)
    print(f"Saved RaVAEn table: {out_path}")

def process_sttorm_table():
    all_metrics = {}
    for disaster in disasters_sttorm:
        folder = f"{disaster}_sttorm"
        for model in models:
            filename = f"{model}_test_metrics.json"
            path = os.path.join(LOG_DIR, folder, filename)
            metrics = load_metrics(path)
            if metrics is None:
                continue
            all_metrics[model] = {m: extract_metrics(metrics, m) for m in [PRIMARY_METRIC, SECONDARY_METRIC]}

    latex_str = make_sttorm_table(all_metrics)
    out_path = os.path.join(OUTPUT_DIR, "sttorm_comparison.tex")
    with open(out_path, "w") as f:
        f.write(latex_str)
    print(f"Saved STTORM-CD table: {out_path}")

def make_combined_table(all_ravaen_metrics, all_sttorm_metrics, metric=PRIMARY_METRIC):
    # Corrected keys to match dataset_display_names
    columns = ["landslides", "fires", "hurricanes", "floods_ravaen", "floods"]

    header = "\\begin{table}\n  \\centering\n"
    header += "  \\begin{tabular}{" + "|l|" + "c|" * len(columns) + "}\n"
    header += "    \\toprule\n"
    header += "    Method " + "".join([f"& {latex_escape(dataset_display_names.get(c, c))}" for c in columns]) + " \\\\\n"
    header += "    \\midrule\n"

    rows = ""
    for i, model in enumerate(models):
        row = latex_escape(display_names[i])

        # RaVAEn datasets → min memory
        for d in ["landslides", "fires", "hurricanes", "floods_ravaen"]:
            metrics = all_ravaen_metrics.get(d, {}).get(model, {})
            val = metrics.get(metric, [0, 0, 0])[1]  # min memory
            row += f" & {val*100:.2f}~\%"

        # STTORM-CD Floods → avg memory
        metrics = all_sttorm_metrics.get(model, {})
        val = metrics.get(metric, [0, 0, 0])[2]  # avg memory
        row += f" & {val*100:.2f}~\%"

        rows += f"    {row} \\\\\n"

    footer = "    \\bottomrule\n  \\end{tabular}\n"
    footer += f"  \\caption{{Combined table ({metric}): RaVAEn min memory, STTORM-CD Floods avg memory}}\n\\end{{table}}\n"

    return header + rows + footer

def process_combined_table():
    # Load all RaVAEn metrics
    all_ravaen_metrics = {}
    for disaster in disasters_ravaen:
        folder = f"{disaster}"
        dataset_metrics = {}
        for model in models:
            filename = f"{model}_test_metrics.json"
            path = os.path.join(LOG_DIR, folder, filename)
            metrics = load_metrics(path)
            if metrics is None:
                continue
            dataset_metrics[model] = {m: extract_metrics(metrics, m) for m in [PRIMARY_METRIC]}
        all_ravaen_metrics[disaster] = dataset_metrics

    # Load STTORM metrics
    all_sttorm_metrics = {}
    for disaster in disasters_sttorm:
        folder = f"{disaster}_sttorm"
        for model in models:
            filename = f"{model}_test_metrics.json"
            path = os.path.join(LOG_DIR, folder, filename)
            metrics = load_metrics(path)
            if metrics is None:
                continue
            all_sttorm_metrics[model] = {m: extract_metrics(metrics, m) for m in [PRIMARY_METRIC]}

    # Make LaTeX table
    latex_str = make_combined_table(all_ravaen_metrics, all_sttorm_metrics, metric=PRIMARY_METRIC)
    out_path = os.path.join(OUTPUT_DIR, f"combined_table_{PRIMARY_METRIC}.tex")
    with open(out_path, "w") as f:
        f.write(latex_str)
    print(f"Saved combined table: {out_path}")

# ---------------- MARKDOWN TABLES PER MEMORY ----------------
def generate_markdown_tables_per_memory():
    """
    Generate 3 Markdown files (one, min, avg memory),
    each containing tables for all metrics and confusion matrices.
    """
    import os
    os.makedirs("markdown_tables", exist_ok=True)

    mem_idx_map = {"one": 0, "min": 1, "avg": 2}
    
    # Map for constructing JSON keys for confusion matrix values
    json_cm_key_map = {
        "one": "One memory",
        "min": "Min memory",
        "avg": "Avg memory"
    }

    # Load all metrics once
    all_metrics = {}
    for dataset in disasters_ravaen + disasters_sttorm:
        folder = dataset if dataset in disasters_ravaen else f"{dataset}_sttorm"
        dataset_metrics = {}
        for model in models:
            path = os.path.join(LOG_DIR, folder, f"{model}_test_metrics.json")
            metrics = load_metrics(path)
            if metrics is None:
                continue
            # Store all raw metrics, including confusion matrix data
            dataset_metrics[model] = metrics 
        all_metrics[dataset] = dataset_metrics

    # --- Generate one file per memory strategy ---
    for mem_str, mem_idx in mem_idx_map.items():
        lines = [f"# Metrics & Confusion Matrices ({mem_str} memory)\n"]

        # --- Part 1: Metric Tables (existing logic) ---
        for metric in metrics_config.keys():
            lines.append(f"## {metric}\n")

            # Header
            header = ["Model"] + [dataset_display_names.get(d, d) for d in disasters_ravaen + disasters_sttorm]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")

            # Rows
            for i, model in enumerate(models):
                row = [display_names[i]]
                for dataset in disasters_ravaen + disasters_sttorm:
                    # Extract metric values using the helper function as before
                    raw_metrics = all_metrics.get(dataset, {}).get(model, {})
                    if not raw_metrics:
                         vals = [0,0,0]
                    else:
                         vals = extract_metrics(raw_metrics, metric)
                    
                    row.append(f"{vals[mem_idx]*100:.2f}")
                lines.append("| " + " | ".join(row) + " |")
            
            lines.append("\n") # space between tables

        # --- Part 2: Confusion Matrix Tables (new logic) ---
        lines.append("\n---\n")
        lines.append("## Confusion Matrices\n")
        
        json_mem_key = json_cm_key_map[mem_str] # e.g., "Min memory"

        for dataset in disasters_ravaen + disasters_sttorm:
            lines.append(f"### {dataset_display_names.get(dataset, dataset)}\n")
            
            # CM Table Header
            cm_header = ["Model", "True Positive", "False Positive", "False Negative", "True Negative"]
            lines.append("| " + " | ".join(cm_header) + " |")
            lines.append("|" + "|".join(["---"] * len(cm_header)) + "|")

            # CM Table Rows
            for i, model in enumerate(models):
                metrics = all_metrics.get(dataset, {}).get(model, {})
                
                # Construct keys and get values, defaulting to 0 if not found
                tp = int(metrics.get(f"{json_mem_key} - True Positives_overall", 0))
                fp = int(metrics.get(f"{json_mem_key} - False Positives_overall", 0))
                fn = int(metrics.get(f"{json_mem_key} - False Negatives_overall", 0))
                tn = int(metrics.get(f"{json_mem_key} - True Negatives_overall", 0))
                
                row = [display_names[i], str(tp), str(fp), str(fn), str(tn)]
                lines.append("| " + " | ".join(row) + " |")
            
            lines.append("\n") # space between dataset tables

        # --- Save the combined Markdown file ---
        out_path = os.path.join("markdown_tables", f"metrics_and_cm_{mem_str}.md")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved Markdown tables and CM ({mem_str} memory): {out_path}")


def make_ablation_table_multirow_model(all_metrics_per_dataset):
    from itertools import product

    sizes = ["Small", "Medium", "Large"]
    margins = ["Variable", "Fixed"]

    if "metrics_config" in globals():
        metrics = list(metrics_config.keys())
    else:
        metrics_set = set()
        for ds in all_metrics_per_dataset.values():
            for model_dict in ds.values():
                metrics_set.update(model_dict.keys())
        metrics = sorted(metrics_set) if metrics_set else ["F1", "Precision", "Recall", "IoU", "RDP"]

    metric_arrow = {m: "↑" for m in metrics}  
    if "RDP" in metrics:
        metric_arrow["RDP"] = "↓"

    datasets = list(all_metrics_per_dataset.keys())
    if "floods" not in datasets:
        datasets.append("floods")

    mem_idx_ravaen = {"one":0,"min":1,"avg":2}
    mem_idx_sttorm = {"one":0,"min":1,"avg":2}

    def _latex_escape(s):
        return str(s).replace("_", "\\_")

    dd = dataset_display_names if "dataset_display_names" in globals() else {d: d for d in datasets}

    # --- HARDCODED VALUES ---
    tiles_count = [626, 27865, 11773, 11253, 6678]
    changed_pct = [23.64, 57.89, 27.57, 21.70, 22.34]  # Already in %

    col_spec = "|l|c|" + "c|" * len(datasets)
    header = "\\begin{table}[h!]\n  \\centering\n"
    header += "  \\begin{tabular}{" + col_spec + "}\n"
    header += "    \\toprule\n"

    # Hardcoded top rows
    header += "    Tiles N: & & " + " & ".join(str(n) for n in tiles_count) + " \\\\\n"
    header += "    Changed [\%]: & & " + " & ".join(f"{n:.2f}" for n in changed_pct) + " \\\\\n"

    # Model & Metric row with two-line dataset names
    header += "    Model & Metric & "
    header += " & ".join([
        r"\shortstack{RaVAEn\\Landslides}",
        r"\shortstack{RaVAEn\\Wildfires}",
        r"\shortstack{RaVAEn\\Hurricanes}",
        r"\shortstack{RaVAEn\\Floods}",
        r"\shortstack{STTORM-CD\\Floods}"
    ])
    header += " \\\\\n"
    header += "    \\midrule\n"

    rows = ""
    for size_idx, size in enumerate(sizes):
        vals_by_metric = {}
        winners_by_metric = {}

        # Compute values
        for metric in metrics:
            vals_by_metric[metric] = {}
            for margin in margins:
                model_name = f"{size.lower()}_my_{margin.lower()}"
                vals = []
                for dataset in datasets:
                    metrics_dict = all_metrics_per_dataset.get(dataset, {}).get(model_name, {})
                    if dataset == "floods" or dataset.lower().endswith("floods"):
                        val = metrics_dict.get(metric, [0,0,0])[mem_idx_ravaen["avg"]] if metrics_dict else 0.0
                    elif dataset.lower().startswith("ravaen"):
                        val = metrics_dict.get(metric, [0,0,0])[mem_idx_ravaen["min"]] if metrics_dict else 0.0
                    else:
                        val = metrics_dict.get(metric, [0,0,0])[mem_idx_sttorm["min"]] if metrics_dict else 0.0
                    vals.append(float(val))
                vals_by_metric[metric][margin] = vals

            # Determine winners
            winners = []
            var_vals = vals_by_metric[metric]["Variable"]
            fix_vals = vals_by_metric[metric]["Fixed"]
            metric_key = metric.lower()
            for v_var, v_fix in zip(var_vals, fix_vals):
                if metric_key == "rdp":
                    if v_var < v_fix: winners.append("Variable")
                    elif v_fix < v_var: winners.append("Fixed")
                    else: winners.append(None)
                else:
                    if v_var > v_fix: winners.append("Variable")
                    elif v_fix > v_var: winners.append("Fixed")
                    else: winners.append(None)
            winners_by_metric[metric] = winners

        # Emit rows grouped by margin
        n_metrics = len(metrics)
        for margin_idx, margin in enumerate(margins):
            other_margin = "Fixed" if margin == "Variable" else "Variable"
            for i, metric in enumerate(metrics):
                cells = []
                if i == 0:
                    model_cell = "\\multirow{" + str(n_metrics) + "}{*}{\\shortstack{" + _latex_escape(size) + " \\\\ (" + margin + ")}}"
                    cells.append(model_cell)
                else:
                    cells.append("")

                cells.append(f"{_latex_escape(metric)}~[\%]~{metric_arrow.get(metric, '')}")

                vals = vals_by_metric[metric][margin]
                other_vals = vals_by_metric[metric][other_margin]
                winners = winners_by_metric[metric]

                for j, v in enumerate(vals):
                    gain_loss = v - other_vals[j]  # fraction difference
                    gain_loss_str = f" ({gain_loss*100:+.2f})" if abs(gain_loss) > 1e-6 else ""
                    cell_text = f"{v*100:.2f}{gain_loss_str}"
                    if winners[j] == margin:
                        cell_text = "\\textbf{" + cell_text + "}"
                    cells.append(cell_text)

                rows += "    " + " & ".join(cells) + " \\\\\n"

            if margin == "Variable":
                rows += "    \\midrule\n"

        if size_idx < len(sizes) - 1:
            rows += "    \\specialrule{1.5pt}{0pt}{0pt}\n"

    footer = "    \\bottomrule\n  \\end{tabular}\n"
    footer += (
        "  \\caption{Ablation study for margin types. Numbers in parentheses indicate the difference relative to the same model with the opposite margin strategy. Bold numbers highlight better results across margins for the same model size. For flood datasets (STTORM-CD Floods and RaVAEn Floods), the final change predictions were derived using $avg()$ on the memory. For the other RaVAEn datasets, we used $min()$ to filter out noise that the model was not trained for.}\n"
    )
    footer += "  \\label{tab:ablation-study}\n"
    footer += "\\end{table}\n"

    return header + rows + footer


def process_ablation_table_multirow_model():
    all_metrics_per_dataset = {}
    # Load RAVAEn metrics
    for disaster in disasters_ravaen:
        dataset_metrics = {}
        for model in models:
            if not "_my_" in model:
                continue
            path = os.path.join(LOG_DIR, disaster, f"{model}_test_metrics.json")
            metrics = load_metrics(path)
            if metrics is None:
                continue
            dataset_metrics[model] = {m: extract_metrics(metrics, m) for m in metrics_config.keys()}
        all_metrics_per_dataset[disaster] = dataset_metrics

    # Load STTORM-CD Floods metrics
    sttorm_metrics = {}
    for model in models:
        if not "_my_" in model:
            continue
        path = os.path.join(LOG_DIR, "floods_sttorm", f"{model}_test_metrics.json")
        metrics = load_metrics(path)
        if metrics is None:
            continue
        sttorm_metrics[model] = {m: extract_metrics(metrics, m) for m in metrics_config.keys()}
    all_metrics_per_dataset["floods"] = sttorm_metrics
    latex_str = make_ablation_table_multirow_model(all_metrics_per_dataset)
    out_path = os.path.join(OUTPUT_DIR, f"ablation_table_multirow_model.tex")
    with open(out_path, "w") as f:
        f.write(latex_str)
    print(f"Saved multirow model ablation table: {out_path}")
        
if __name__ == "__main__":
    process_ravaen_tables()
    process_sttorm_table()
    process_combined_table()
    generate_markdown_tables_per_memory()  # <-- new function call
    process_ablation_table_multirow_model()
    print("All LaTeX and Markdown tables generated.")