"""
Launch MLflow UI to view experiment tracking
"""
import subprocess
import sys
import os
import webbrowser
import time
from pathlib import Path

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

if __name__ == "__main__":
    print("🔬 Starting MLflow UI...")
    print("📊 MLflow will open at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop")
    print("-" * 50)

    # Ensure mlruns directory exists
    mlruns_dir = Path("mlruns")
    if not mlruns_dir.exists():
        print("❌ No mlruns directory found. Please train some models first!")
        print("   Run: python test_mlflow_integration.py")
        sys.exit(1)

    try:
        # Start MLflow UI
        process = subprocess.Popen([
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", "file:./mlruns",
            "--host", "127.0.0.1",
            "--port", "5000"
        ])

        # Wait a bit for server to start
        print("⏳ Waiting for MLflow UI to start...")
        time.sleep(3)

        # Open browser
        print("🌐 Opening browser...")
        webbrowser.open("http://127.0.0.1:5000")

        print("\n✅ MLflow UI is running!")
        print("   If browser doesn't open automatically, visit: http://127.0.0.1:5000")
        print("   Press Ctrl+C to stop\n")

        # Wait for process
        process.wait()

    except KeyboardInterrupt:
        print("\n👋 MLflow UI stopped")
        if process:
            process.terminate()
