#!/usr/bin/env python3
"""
Brand Eye Setup Script
Automated setup for the YOLO Object Detection GUI application
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command with error handling"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Please upgrade to Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_pip():
    """Check if pip is available"""
    print("\n📦 Checking pip availability...")
    try:
        import pip
        print("✅ pip is available")
        return True
    except ImportError:
        print("❌ pip is not installed")
        return False

def create_virtual_environment():
    """Create virtual environment"""
    venv_path = Path(".venv")
    if venv_path.exists():
        print("\n🏠 Virtual environment already exists")
        return True
    
    return run_command(
        f"{sys.executable} -m venv .venv",
        "Creating virtual environment"
    )

def get_activation_command():
    """Get the correct activation command for the platform"""
    if platform.system() == "Windows":
        return ".venv\\Scripts\\activate"
    else:
        return "source .venv/bin/activate"

def get_python_executable():
    """Get the correct Python executable path"""
    if platform.system() == "Windows":
        return ".venv\\Scripts\\python.exe"
    else:
        return ".venv/bin/python"

def install_requirements():
    """Install required packages"""
    python_exe = get_python_executable()
    
    # Upgrade pip first
    if not run_command(
        f"{python_exe} -m pip install --upgrade pip",
        "Upgrading pip"
    ):
        return False
    
    # Install requirements
    return run_command(
        f"{python_exe} -m pip install -r requirements.txt",
        "Installing requirements"
    )

def check_model_file():
    """Check if YOLO model file exists"""
    model_path = Path("model/best.pt")
    print(f"\n🤖 Checking for YOLO model file...")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file found: {model_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"⚠️  Model file not found: {model_path}")
        print("   Please place your YOLO model file at 'model/best.pt'")
        
        # Create model directory if it doesn't exist
        model_path.parent.mkdir(exist_ok=True)
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print(f"\n🧪 Testing package imports...")
    python_exe = get_python_executable()
    
    test_packages = [
        "PyQt6",
        "torch",
        "torchvision", 
        "ultralytics",
        "cv2",
        "PIL",
        "numpy"
    ]
    
    for package in test_packages:
        if not run_command(
            f"{python_exe} -c \"import {package}; print(f'{package} imported successfully')\"",
            f"Testing {package}"
        ):
            return False
    
    return True

def check_cuda():
    """Check CUDA availability"""
    print(f"\n🚀 Checking CUDA availability...")
    python_exe = get_python_executable()
    
    result = run_command(
        f"{python_exe} -c \"import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)\"",
        "Checking CUDA"
    )
    
    return result

def create_launcher_scripts():
    """Create launcher scripts for different platforms"""
    print(f"\n🚀 Creating launcher scripts...")
    
    # Windows batch file
    if platform.system() == "Windows":
        bat_content = '''@echo off
echo Starting Brand Eye - Image Detection GUI
cd /d "%~dp0"
call .venv\\Scripts\\activate
python main_gui.py
pause
'''
        with open("launch_image_gui.bat", "w") as f:
            f.write(bat_content)
        
        bat_content_video = '''@echo off
echo Starting Brand Eye - Video Processing GUI
cd /d "%~dp0"
call .venv\\Scripts\\activate
python video_gui.py
pause
'''
        with open("launch_video_gui.bat", "w") as f:
            f.write(bat_content_video)
        
        print("✅ Created Windows launcher files:")
        print("   - launch_image_gui.bat")
        print("   - launch_video_gui.bat")
    
    # Unix shell script
    else:
        sh_content = '''#!/bin/bash
echo "Starting Brand Eye - Image Detection GUI"
cd "$(dirname "$0")"
source .venv/bin/activate
python main_gui.py
'''
        with open("launch_image_gui.sh", "w") as f:
            f.write(sh_content)
        os.chmod("launch_image_gui.sh", 0o755)
        
        sh_content_video = '''#!/bin/bash
echo "Starting Brand Eye - Video Processing GUI"
cd "$(dirname "$0")"
source .venv/bin/activate
python video_gui.py
'''
        with open("launch_video_gui.sh", "w") as f:
            f.write(sh_content_video)
        os.chmod("launch_video_gui.sh", 0o755)
        
        print("✅ Created Unix launcher files:")
        print("   - launch_image_gui.sh")
        print("   - launch_video_gui.sh")

def main():
    """Main setup function"""
    print("🎯 Brand Eye Setup Script")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Setup virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install packages
    if not install_requirements():
        sys.exit(1)
    
    # Test installation
    if not test_imports():
        print("⚠️  Some packages failed to import. Please check the error messages above.")
    
    # Check CUDA
    check_cuda()
    
    # Check model file
    model_exists = check_model_file()
    
    # Create launchers
    create_launcher_scripts()
    
    # Final summary
    print("\n🎉 Setup Summary")
    print("=" * 30)
    print("✅ Virtual environment created")
    print("✅ Requirements installed")
    print("✅ Launcher scripts created")
    
    if model_exists:
        print("✅ YOLO model file found")
    else:
        print("⚠️  YOLO model file missing")
    
    print(f"\n🚀 Ready to run!")
    if platform.system() == "Windows":
        print("   Double-click 'launch_image_gui.bat' for image detection")
        print("   Double-click 'launch_video_gui.bat' for video processing")
    else:
        print("   Run './launch_image_gui.sh' for image detection")
        print("   Run './launch_video_gui.sh' for video processing")
    
    print(f"\n📝 Manual activation command:")
    print(f"   {get_activation_command()}")

if __name__ == "__main__":
    main()