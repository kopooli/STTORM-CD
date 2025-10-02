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
    "fires": "RaVAEn-Fires",
    "hurricanes": "RaVAEn-Hurricanes",
    "floods_ravaen": "RaVAEn-Floods",
    "floods": "STTORM-Floods"
}
# Models (internal names)
models = [
    "original_small", "original_medium", "original_large",
    "my_small_variable", "my_medium_variable", "my_large_variable",
    "my_small_fixed", "my_medium_fixed", "my_large_fixed",
    "Index", "Cos baseline"
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
    "AP": ["Custom MAP (one memory)_overall", "Custom MAP (min memory)_overall", "Custom MAP (avg memory)_overall"],
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
def make_ravaen_table(all_metrics_per_dataset):
    datasets = list(all_metrics_per_dataset.keys())
    header = "\\begin{table}\n  \\centering\n  \\begin{tabular}{|l|" + "ll|" * len(datasets) + "}\n"
    header += "    \\toprule\n"
    header += "    \\multicolumn{1}{|r|}{Dataset} " + \
          "".join([f"& \\multicolumn{{2}}{{|r|}}{{{latex_escape(dataset_display_names[d])}}}" for d in datasets]) + " \\\\\n"
    header += "    \\multicolumn{1}{|r|}{Tiles count} " + "".join([f"& \\multicolumn{{2}}{{|r|}}{{{all_metrics_per_dataset[d].get('total_tiles',0)}}}" for d in datasets]) + " \\\\\n"
    header += "    \\multicolumn{1}{|r|}{Changed tiles [\\%]} " + "".join([f"& \\multicolumn{{2}}{{|r|}}{{{all_metrics_per_dataset[d].get('changed_percent',0):.2f}}}" for d in datasets]) + " \\\\\n"
    header += f"    \\multicolumn{{1}}{{|r|}}{{Metric}} " + "".join([f"& {PRIMARY_METRIC} $\\uparrow$ & {SECONDARY_METRIC} $\\downarrow$" for _ in datasets]) + " \\\\\n    \\midrule\n"

    rows = ""
    for i, model in enumerate(models):
        row = latex_escape(display_names[i])
        for dataset in datasets:
            metrics = all_metrics_per_dataset[dataset].get(model, {})
            mem_idx = {"one": 0, "min": 1, "avg": 2}[MEMORY_STRATEGY]
            p_vals = metrics.get(PRIMARY_METRIC, [0,0,0])
            s_vals = metrics.get(SECONDARY_METRIC, [0,0,0])
            row += f" & {p_vals[mem_idx]:.3f} & {s_vals[mem_idx]:.2f}"
        rows += f"    {row} \\\\\n"

    footer = f"    \\bottomrule\n  \\end{{tabular}}\n  \\caption{{RaVAEn comparison ({MEMORY_STRATEGY} memory) for {latex_escape(PRIMARY_METRIC)} and {latex_escape(SECONDARY_METRIC)}}}\n\\end{{table}}\n"
    return header + rows + footer

def make_sttorm_table(all_metrics):
    col_format = "|l|ccc|ccc|"
    header = f"\\begin{{table}}\n  \\begin{{tabular}}{{{col_format}}}\n    \\toprule\n"
    header += f"    \\multicolumn{{1}}{{|c|}}{{Method}} & \\multicolumn{{3}}{{c|}}{{{PRIMARY_METRIC} $\\uparrow$}} & \\multicolumn{{3}}{{c|}}{{{SECONDARY_METRIC} $\\downarrow$}} \\\\\n"
    header += "    \\cmidrule(r){2-4} \\cmidrule(l){5-7}\n"
    header += "    & Most recent & min(memory) & avg(memory) & Most recent & min(memory) & avg(memory) \\\\\n    \\midrule\n"

    rows = ""
    for i, model in enumerate(models):
        metrics = all_metrics.get(model, {})
        p_vals = metrics.get(PRIMARY_METRIC, [0,0,0])
        s_vals = metrics.get(SECONDARY_METRIC, [0,0,0])
        row = f"{latex_escape(display_names[i])} & {p_vals[0]:.3f} & {p_vals[1]:.3f} & {p_vals[2]:.3f} & {s_vals[0]:.2f} & {s_vals[1]:.2f} & {s_vals[2]:.2f}"
        rows += f"    {row} \\\\\n"

    footer = f"    \\bottomrule\n  \\end{{tabular}}\n  \\caption{{STTORM-CD Floods dataset metrics: {latex_escape(PRIMARY_METRIC)} and {latex_escape(SECONDARY_METRIC)}}}\n\\end{{table}}\n"
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
            row += f" & {val:.3f}"

        # STTORM-CD Floods → avg memory
        metrics = all_sttorm_metrics.get(model, {})
        val = metrics.get(metric, [0, 0, 0])[2]  # avg memory
        row += f" & {val:.3f}"

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
    each containing tables for all metrics.
    """
    import os
    os.makedirs("markdown_tables", exist_ok=True)

    mem_idx_map = {"one": 0, "min": 1, "avg": 2}

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
            model_metrics = {metric: extract_metrics(metrics, metric) for metric in metrics_config.keys()}
            dataset_metrics[model] = model_metrics
        all_metrics[dataset] = dataset_metrics

    # Generate one file per memory strategy
    for mem_str, mem_idx in mem_idx_map.items():
        lines = [f"# Metrics Table ({mem_str} memory)\n"]

        for metric in metrics_config.keys():
            lines.append(f"## {metric}\n")

            # Header
            header = ["Model"] + [dataset_display_names.get(dataset, dataset) for dataset in disasters_ravaen + disasters_sttorm]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")

            # Rows
            for i, model in enumerate(models):
                row = [display_names[i]]
                for dataset in disasters_ravaen + disasters_sttorm:
                    metrics = all_metrics.get(dataset, {}).get(model, {})
                    vals = metrics.get(metric, [0, 0, 0]) if metrics else [0,0,0]
                    row.append(f"{vals[mem_idx]:.3f}")
                lines.append("| " + " | ".join(row) + " |")

            lines.append("\n")  # space between tables

        # Save Markdown
        out_path = os.path.join("markdown_tables", f"metrics_{mem_str}.md")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved Markdown tables ({mem_str} memory): {out_path}")

def make_ablation_table_multirow_model(all_metrics_per_dataset):
    """
    Generate LaTeX table for ablation study:
    - Model column includes size + margin (e.g., Small / newline (Fixed))
    - Metrics as separate rows
    - RAVAEn min memory, STTORM-CD avg memory
    """
    from itertools import product
    sizes = ["Small", "Medium", "Large"]
    margins = ["Fixed", "Variable"]
    metrics = list(metrics_config.keys())

    datasets = list(all_metrics_per_dataset.keys())
    if "floods" not in datasets:
        datasets.append("floods")  # STTORM-CD last column

    mem_idx_ravaen = {"one":0,"min":1,"avg":2}["min"]
    mem_idx_sttorm = {"one":0,"min":1,"avg":2}["avg"]
    
    # Correctly define the number of data columns
    num_data_cols = len(datasets) 
    
    # Define the table with the correct number of columns: 1 for Model, 1 for Metric, and num_data_cols for values.
    # The provided LaTeX actually has 7 columns (Model, Metric, 5x Data), so your tabular definition was also off.
    # It should have 1 'l' column (Model), 1 'c' column (Metric), and num_data_cols 'c' columns.
    # Based on your provided table, the column headers are: Model, Metric, and 5 datasets.
    # Therefore, your table should have 1 (Model/Metric) + 5 (Datasets) = 6 columns. The error was in the row generation.
    
    header = "\\begin{table}[h!]\n  \\centering\n"
    # This definition for 6 columns is correct.
    header += "  \\begin{tabular}{|l|c|" + "c|"*(len(datasets)-1) + "}\n" # Corrected: 1 'l' (Model), 1 'c' (Metric), 4 'c' (Data) is wrong.
    # The first column is the model, the second is the metric. The remaining are datasets.
    # Your provided table has 7 columns in the body, but 6 in the header. Let's fix this based on the body.
    # Column 1: Model, Col 2: Metric, Col 3-7: Data. This is 7 columns.
    
    # Let's rebuild based on the provided correct LaTeX output
    header = "\\begin{table}[h!]\n  \\centering\n"
    header += "  \\begin{tabular}{|l|c|c|c|c|c|c|}\n" # Correct definition for 7 columns
    header += "    \\toprule\n"
    # The header row in the code had one fewer column than the body. Let's fix that.
    # The first column header is for the Model, the second is empty but corresponds to the Metric column.
    header += "    Model & " + " & ".join([f"{latex_escape(dataset_display_names[d])}" for d in datasets]) + " \\\\\n"
    header += "    \\midrule\n"
    
    rows = ""
    for size, margin in product(sizes, margins):
        model_name = f"my_{size.lower()}_{margin.lower()}"
        n_metrics = len(metrics)
        model_cell = f"\\shortstack{{{size} \\\\ ({margin})}}"

        for i, metric in enumerate(metrics):
            # Create a list to hold all cells for the current row
            row_cells = []
            
            # --- Column 1 & 2: Model and Metric ---
            if i == 0:
                # First row of the group gets the multirow model name
                row_cells.append(f"\\multirow{{{n_metrics}}}{{*}}{{{model_cell}}}")
                row_cells.append(f"{latex_escape(metric)}")
            else:
                # Subsequent rows are blank in the first column
                row_cells.append("")
                row_cells.append(f"{latex_escape(metric)}")

            # --- Columns 3+: Data values ---
            for dataset in datasets:
                metrics_dict = all_metrics_per_dataset.get(dataset, {}).get(model_name, {})
                if dataset == "floods":  # STTORM-CD → avg memory
                    val = metrics_dict.get(metric, [0,0,0])[mem_idx_sttorm]
                else:  # RAVAEn → min memory
                    val = metrics_dict.get(metric, [0,0,0])[mem_idx_ravaen]
                row_cells.append(f"{val:.3f}")
            
            # Join all cells with " & " and add the row ending
            rows += "    " + " & ".join(row_cells) + " \\\\\n"

        rows += "    \\midrule\n"
    
    # Remove the final midrule to match LaTeX best practices before bottomrule
    if rows.endswith("    \\midrule\n"):
        rows = rows[:-len("    \\midrule\n")]

    footer = "    \\bottomrule\n  \\end{tabular}\n"
    footer += f"  \\caption{{Ablation study: model & margin effect combined (RAVAEn min memory, STTORM-CD avg memory, all metrics)}}\n\\end{{table}}\n"
    
    return header + rows + footer


def process_ablation_table_multirow_model():
    all_metrics_per_dataset = {}
    # Load RAVAEn metrics
    for disaster in disasters_ravaen:
        dataset_metrics = {}
        for model in models:
            if not model.startswith("my_"):
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
        if not model.startswith("my_"):
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