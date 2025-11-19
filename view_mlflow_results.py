"""
View MLflow Experiment Results
Simple viewer for MLflow experiments without the UI
"""
import mlflow
from mlflow import MlflowClient
import pandas as pd
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Set tracking URI
mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

print("=" * 80)
print("🔬 MLflow Experiment Results Viewer")
print("=" * 80)

# Get all experiments
experiments = client.search_experiments()

for exp in experiments:
    if exp.name == "Default":
        continue

    print(f"\n📊 Experiment: {exp.name}")
    print(f"   ID: {exp.experiment_id}")
    print("-" * 80)

    # Get all runs for this experiment
    runs = client.search_runs(experiment_ids=[exp.experiment_id])

    if not runs:
        print("   No runs found")
        continue

    print(f"   Found {len(runs)} runs:\n")

    # Create a table of results
    results = []
    for run in runs:
        run_data = {
            "Run Name": run.info.run_name,
            "Status": run.info.status,
        }

        # Add all metrics
        for metric_name, metric_value in run.data.metrics.items():
            run_data[metric_name] = f"{metric_value:.4f}"

        results.append(run_data)

    # Convert to DataFrame for nice display
    df = pd.DataFrame(results)

    # Sort by r2_score if it exists, otherwise by first metric
    if 'r2_score' in df.columns:
        df = df.sort_values('r2_score', ascending=False)
    elif 'accuracy' in df.columns:
        df = df.sort_values('accuracy', ascending=False)

    print(df.to_string(index=False))
    print("\n")

    # Show best model
    if results:
        best_run = runs[0]  # Already sorted by metric
        print(f"   🏆 Best Model: {best_run.info.run_name}")
        print(f"      Metrics:")
        for metric_name, metric_value in best_run.data.metrics.items():
            print(f"         - {metric_name}: {metric_value:.4f}")
        print(f"\n      Parameters:")
        for param_name, param_value in best_run.data.params.items():
            print(f"         - {param_name}: {param_value}")

print("\n" + "=" * 80)
print("✅ Done! To view in MLflow UI, run: python run_mlflow.py")
print("=" * 80)
