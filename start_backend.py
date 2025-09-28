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
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Backend dependencies installed")
    
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
        sys.exit(1)

if __name__ == "__main__":
    main()
