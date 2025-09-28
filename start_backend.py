#!/usr/bin/env python3
"""
Startup script for the Digital Twin Thermal Cooling Loop backend server
"""

import os
import sys
import subprocess
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Start the backend server"""
    print("Starting Digital Twin Thermal Cooling Loop Backend Server...")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    os.chdir(backend_dir)
    
    # Check if requirements are installed
    try:
        import fastapi
        import uvicorn
        print("✓ Backend dependencies found")
    except ImportError:
        print("Installing backend dependencies...")
        print("This may take a few minutes...")
        
        # Try to install with --upgrade and --no-cache-dir for better compatibility
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                '--upgrade', '--no-cache-dir', 
                '-r', 'requirements.txt'
            ], check=True)
            print("✓ Backend dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            print("\nTrying alternative installation method...")
            
            # Try installing packages one by one
            packages = [
                'fastapi>=0.100.0',
                'uvicorn[standard]>=0.20.0',
                'websockets>=11.0',
                'numpy>=1.21.0',
                'pandas>=1.5.0',
                'scipy>=1.9.0',
                'scikit-learn>=1.1.0',
                'matplotlib>=3.5.0',
                'joblib>=1.2.0'
            ]
            
            for package in packages:
                try:
                    print(f"Installing {package}...")
                    subprocess.run([
                        sys.executable, '-m', 'pip', 'install', 
                        '--upgrade', '--no-cache-dir', package
                    ], check=True)
                except subprocess.CalledProcessError:
                    print(f"⚠️  Warning: Failed to install {package}")
                    continue
            
            print("✓ Alternative installation completed")
    
    # Start the server
    print("\nStarting server on http://localhost:8000")
    print("Web client will be available at: http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Import and run the FastAPI app
        from main import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\nError starting server: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Python 3.8+ is installed")
        print("2. Try: pip install --upgrade pip setuptools wheel")
        print("3. Try: pip install --upgrade fastapi uvicorn")
        sys.exit(1)

if __name__ == "__main__":
    main()
