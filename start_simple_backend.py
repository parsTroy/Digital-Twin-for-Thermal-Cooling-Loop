#!/usr/bin/env python3
"""
Simple startup script for the Digital Twin Thermal Cooling Loop backend server
This version uses minimal dependencies for better compatibility
"""

import os
import sys
import subprocess
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Start the backend server with minimal dependencies"""
    print("Starting Digital Twin Thermal Cooling Loop Backend Server (Simple Mode)...")
    print("=" * 70)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    os.chdir(backend_dir)
    
    # Install minimal dependencies first
    print("Installing minimal dependencies...")
    minimal_packages = [
        'fastapi',
        'uvicorn',
        'websockets'
    ]
    
    for package in minimal_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                '--upgrade', '--no-cache-dir', package
            ], check=True, capture_output=True)
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {package}")
    
    # Try to install scientific packages
    print("\nInstalling scientific packages...")
    scientific_packages = [
        'numpy',
        'pandas', 
        'scipy',
        'scikit-learn'
    ]
    
    for package in scientific_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                '--upgrade', '--no-cache-dir', package
            ], check=True, capture_output=True)
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {package}")
    
    # Start the server
    print("\n" + "=" * 70)
    print("Starting server on http://localhost:8000")
    print("Web client will be available at: http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    
    try:
        # Import and run the FastAPI app
        from main import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("\nTrying to install missing packages...")
        
        # Try to install the missing package
        missing_package = str(e).split("'")[1] if "'" in str(e) else "fastapi"
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                '--upgrade', '--no-cache-dir', missing_package
            ], check=True)
            print(f"✓ {missing_package} installed, please try again")
        except:
            print(f"Failed to install {missing_package}")
            print("\nPlease install manually: pip install fastapi uvicorn")
        
        sys.exit(1)
    except Exception as e:
        print(f"\nError starting server: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Python 3.8+ is installed")
        print("2. Try: pip install --upgrade pip setuptools wheel")
        print("3. Try: pip install fastapi uvicorn websockets")
        print("4. Check if port 8000 is available")
        sys.exit(1)

if __name__ == "__main__":
    main()
