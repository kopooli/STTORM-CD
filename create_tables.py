import os
import json

# ---------------- CONFIG ----------------
LOG_DIR = "test_metrics_logs"       # Where JSON logs are stored
OUTPUT_DIR = "latex_tables"         # Where LaTeX tables will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Datasets configuration
disasters_ravaen = ["landslides", "fires", "hurricanes", "floods"]
disasters_sttorm = ["floods"]  # final floods dataset with raven=False

# Model names (must match JSON filenames)
models = [
    "original_small",
    "original_medium",
    "original_large",
    "my_small_variable",
    "my_medium_variable",
    "my_large_variable",
    "my_small_fixed",
    "my_medium_fixed",
    "my_large_fixed"
]

# ---------------- FUNCTIONS ----------------
def load_metrics(json_path):
    """Load JSON metrics from a file."""
    with open(json_path, "r") as f:
        metrics = json.load(f)
    # If stored as list with one dict, extract it
    if isinstance(metrics, list):
        metrics = metrics[0]
    return metrics

def extract_table_metrics(metrics):
    """Extract AURC, RDP, AUPRC, MAP for one run."""
    table = {}
    table["AURC"] = [
        metrics.get("area_under_the_curve_one_overall", 0),
        metrics.get("area_under_the_curve_min_overall", 0),
        metrics.get("area_under_the_curve_avg_overall", 0),
    ]
    table["RDP"] = [
        metrics.get("downlink_amount_one_overall", 0),
        metrics.get("downlink_amount_min_overall", 0),
        metrics.get("downlink_amount_avg_overall", 0),
    ]
    table["AUPRC"] = [
        metrics.get("Tile-level AUPRC (one memory)_overall", 0),
        metrics.get("Tile-level AUPRC (min memory)_overall", 0),
        metrics.get("Tile-level AUPRC (avg memory)_overall", 0),
    ]
    table["MAP"] = [
        metrics.get("Tile-level MAP (one memory)_overall", 0),
        metrics.get("Tile-level MAP (min memory)_overall", 0),
        metrics.get("Tile-level MAP (avg memory)_overall", 0),
    ]
    return table

def make_latex_table(dataset_name, metric1_name, metric2_name, all_metrics):
    """
    Generate LaTeX table string for two metrics (e.g., AURC + RDP)
    all_metrics: dict of model_name -> metric dict
    """
    header = (
        "\\begin{table}\n"
        "  \\begin{tabular}{|l|ccc|ccc|}\n"
        "    \\toprule\n"
        f"    \\multicolumn{{1}}{{|c|}}{{Method}} & "
        f"\\multicolumn{{3}}{{c|}}{{{metric1_name} $\\uparrow$}} & "
        f"\\multicolumn{{3}}{{c|}}{{{metric2_name} $\\downarrow$}} \\\\\n"
        "    \\cmidrule(r){2-4} \\cmidrule(l){5-7}\n"
        "    & Most recent & min(memory) & avg(memory) & Most recent & min(memory) & avg(memory) \\\\\n"
        "    \\midrule\n"
    )

    rows = ""
    for model_name, metrics in all_metrics.items():
        val1 = metrics[metric1_name]
        val2 = metrics[metric2_name]
        rows += f"    {model_name.replace('_',' ')} & {val1[0]:.3f} & {val1[1]:.3f} & {val1[2]:.3f} & {val2[0]:.2f} & {val2[1]:.2f} & {val2[2]:.2f} \\\\\n"

    footer = (
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        f"  \\caption{{{metric1_name} and {metric2_name} results for {dataset_name}}}\n"
        f"  \\label{{tab:{dataset_name}-{metric1_name.lower()}-{metric2_name.lower()}}}\n"
        "\\end{table}\n"
    )
    return header + rows + footer

# ---------------- MAIN ----------------
def process_dataset(disaster_name, is_ravaen=True):
    """Process a single dataset and generate LaTeX tables."""
    folder_name = f"{disaster_name}_ravaen" if is_ravaen else f"{disaster_name}_sttorm"
    output_folder = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    all_metrics = {}
    for model_name in models:
        json_path = os.path.join(LOG_DIR, folder_name, f"{model_name}_test_metrics.json")
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skipping")
            continue
        metrics = load_metrics(json_path)
        all_metrics[model_name] = extract_table_metrics(metrics)

    # Generate LaTeX tables
    table_pairs = [("AURC", "RDP"), ("AUPRC", "MAP")]
    for metric1, metric2 in table_pairs:
        latex_str = make_latex_table(folder_name, metric1, metric2, all_metrics)
        out_path = os.path.join(output_folder, f"{metric1}_{metric2}.tex")
        with open(out_path, "w") as f:
            f.write(latex_str)
        print(f"Saved LaTeX table: {out_path}")

if __name__ == "__main__":
    # Loop over RaVAEn disasters
    for disaster in disasters_ravaen:
        process_dataset(disaster, is_ravaen=True)

    # Loop over STTORM flood dataset
    for disaster in disasters_sttorm:
        process_dataset(disaster, is_ravaen=False)

    print("All LaTeX tables generated.")