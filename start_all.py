"""
Start all services: Backend API + Streamlit Frontend
Run MLflow UI separately with: python run_mlflow.py
"""
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def start_service(name, command, cwd=None):
    """Start a service in the background"""
    print(f"🚀 Starting {name}...")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )
    return process

if __name__ == "__main__":
    print("=" * 70)
    print("🤖 AutoML Pipeline Builder - Starting Services")
    print("=" * 70)

    processes = []

    try:
        # Start FastAPI backend
        backend_cmd = f"{sys.executable} run_backend.py"
        backend = start_service("FastAPI Backend", backend_cmd)
        processes.append(("Backend", backend))
        time.sleep(3)

        # Start Streamlit frontend (it will auto-open browser)
        streamlit_cmd = f"{sys.executable} -m streamlit run frontend/app.py"
        frontend = start_service("Streamlit Frontend", streamlit_cmd)
        processes.append(("Frontend", frontend))
        time.sleep(5)  # Wait for Streamlit to start

        print("\n✅ All services started!")
        print("=" * 70)
        print("📱 Application URLs:")
        print("   Frontend:  http://localhost:8501 (should open automatically)")
        print("   API Docs:  http://localhost:8000/api/docs")
        print("\n📊 To view MLflow experiments:")
        print("   Run in new terminal: python run_mlflow.py")
        print("   Or use terminal viewer: python view_mlflow_results.py")
        print("=" * 70)
        print("\nPress Ctrl+C to stop all services\n")

        # Don't open browser - Streamlit does it automatically

        # Keep running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping all services...")
        for name, process in processes:
            print(f"   Stopping {name}...")
            process.terminate()
        print("👋 All services stopped")
