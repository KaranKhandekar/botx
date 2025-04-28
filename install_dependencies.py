import subprocess
import sys
import os

def install_dependencies():
    """Install all required dependencies for marketplace_x.py"""
    # List of all required packages
    required_packages = [
        "requests",
        "PyQt6",
        "Pillow",
        "numpy",
        "imagehash",
        "opencv-python",
        "pandas",
        "openpyxl"
    ]
    
    print("=== Marketplace-X Dependency Installer ===")
    print("This will install all required packages.")
    
    # Check which packages are already installed
    missing_packages = []
    for package in required_packages:
        try:
            # Handle special cases
            if package == "Pillow":
                module_name = "PIL"
            elif package == "opencv-python":
                module_name = "cv2"
            else:
                module_name = package.lower().split('-')[0]
                
            __import__(module_name)
            print(f"✓ {package} is already installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    # If there are missing packages, install them
    if missing_packages:
        print("\nInstalling missing dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("\nAll dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\nError installing dependencies: {e}")
            print("\nPlease try installing them manually using:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    else:
        print("\nAll dependencies are already installed!")
    
    return True

if __name__ == "__main__":
    success = install_dependencies()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)