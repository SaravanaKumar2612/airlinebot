#!/usr/bin/env python3
"""
Setup script for the airline chatbot with DistilBERT model
"""

import subprocess
import sys
import os

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Python requirements installed!")

def check_model():
    """Check if the DistilBERT model exists"""
    model_path = "../dilbert_airline"
    if os.path.exists(model_path):
        print(f"✅ DistilBERT model found at {model_path}")
        return True
    else:
        print(f"❌ DistilBERT model not found at {model_path}")
        print("Please ensure the dilbert_airline folder exists in the project root.")
        return False

def main():
    print("🚀 Setting up Airline Chatbot with DistilBERT Model")
    print("=" * 55)
    
    # Change to backend directory
    os.chdir("backend")
    
    try:
        # Check if model exists
        if not check_model():
            sys.exit(1)
        
        # Install requirements
        install_requirements()
        
        print("\n🎉 Setup complete!")
        print("\nTo run the application:")
        print("1. Start the backend: python main.py")
        print("2. In another terminal, start the frontend: cd ../frontend && npm start")
        print("\nThe chatbot will be available at http://localhost:3000")
        print("\nSupported intents include:")
        print("- Cancel Trip, Change Flight, Flight Status")
        print("- Baggage queries, Seat requests, Check-in help")
        print("- And 22+ more airline-related intents!")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
