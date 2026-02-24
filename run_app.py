import os
import subprocess
import time
import sys

def run_app():
    print("--- RestoraAI Demo Environment Startup ---")
    
    # Check if a model exists
    model_path = os.path.join("saved_models", "restora_gen.pth")
    if not os.path.exists(model_path):
        print("Note: Restoration weights not detected. Demonstration results may be uninitialized.")
    
    # Start Backend
    print("Starting RestoraAI Backend (FastAPI)...")
    backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"])
    
    print("\nApplication is ready!")
    print("Click 'Start Restoration' to demonstrate the supervised pipeline.")
    print("Access URL: http://127.0.0.1:8000")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend.terminate()

if __name__ == "__main__":
    run_app()
