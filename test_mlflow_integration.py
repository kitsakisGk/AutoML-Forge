"""
Test MLflow integration
"""
import pandas as pd
from backend.core.automl_engine import AutoMLEngine
import sys

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Create sample data
data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60, 25, 30, 35, 40, 45],
    'income': [30000, 45000, 60000, 75000, 90000, 105000, 120000, 135000, 32000, 47000, 62000, 77000, 92000],
    'city': ['Zurich', 'Geneva', 'Basel', 'Zurich', 'Geneva', 'Basel', 'Zurich', 'Geneva', 'Basel', 'Zurich', 'Geneva', 'Basel', 'Zurich'],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor'],
    'purchased': [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

print("🤖 Testing MLflow Integration...")
print(f"📊 Dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"🎯 Target: income")
print("-" * 50)

# Initialize AutoML with MLflow enabled
engine = AutoMLEngine(
    df=df,
    target_column='income',
    test_size=0.2,
    use_mlflow=True,
    experiment_name="Test_AutoML"
)

# Detect problem type
problem_type = engine.detect_problem_type()
print(f"✅ Problem type: {problem_type}")

# Prepare data
prep_info = engine.prepare_data()
print(f"✅ Data prepared: {prep_info['n_train']} train, {prep_info['n_test']} test samples")

# Train models
print("🚀 Training models with MLflow tracking...")
engine.train_models()

# Get summary
summary = engine.get_summary()
print(f"\n🏆 Best Model: {summary['best_model']}")
print(f"📈 Score: {summary['best_model_score']}")

print("\n✅ MLflow integration test complete!")
print("📊 View results: python run_mlflow.py")
print("   Then open: http://localhost:5000")
