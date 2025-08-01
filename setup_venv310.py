#!/usr/bin/env python3
"""
###############################################################################
AUTOMATED PYTHON 3.10 VIRTUAL ENVIRONMENT SETUP WITH RETRY LOGIC
###############################################################################

Author: Christos Botos
Description: Automated script to create and configure a Python 3.10 virtual 
             environment with retry logic for robust package installation.
Dependencies: Python 3.10, pip, venv module
Usage: python setup_venv310.py
Arguments: None
Inputs: requirements.txt file in the same directory
Outputs: venv310/ directory with fully configured virtual environment
Key Features:
    - Automated retry loop for failed installations
    - Complete environment cleanup and recreation on failures
    - Comprehensive logging and status reporting
    - Package version verification
    - Cross-platform compatibility (Windows/Linux/macOS)
Notes: 
    - Requires Python 3.10 to be installed and accessible
    - Will completely remove and recreate venv310 on each retry
    - Continues until successful installation or maximum retries reached
###############################################################################
"""

import os
import sys
import subprocess
import shutil
import time
import traceback
from pathlib import Path

# Configuration constants.
VENV_NAME = "venv310"
MAX_RETRIES = 1
RETRY_DELAY = 2  # seconds between retries
REQUIREMENTS_FILE = "requirements.txt"

def print_header(title):
    """Print a formatted header for better output organization."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_status(message, status="INFO"):
    """Print a status message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{status}] {message}")

def check_python310():
    """Check if Python 3.10 is available and accessible."""
    print_header("CHECKING PYTHON 3.10 AVAILABILITY")
    
    # Try different Python 3.10 command variations.
    python_commands = ["python3.10", "python3", "python"]
    
    for cmd in python_commands:
        try:
            result = subprocess.run([cmd, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_info = result.stdout.strip()
                print_status(f"Found Python: {version_info}")
                
                # Check if it's Python 3.10.x
                if "Python 3.10" in version_info:
                    print_status(f"✓ Python 3.10 confirmed with command: {cmd}")
                    return cmd
                else:
                    print_status(f"✗ {cmd} is not Python 3.10: {version_info}")
                    
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            print_status(f"✗ {cmd} not found or not accessible")
            continue
    
    print_status("ERROR: Python 3.10 not found!", "ERROR")
    print_status("Please install Python 3.10 and ensure it's in your PATH", "ERROR")
    return None

def remove_venv():
    """Remove existing virtual environment directory."""
    venv_path = Path(VENV_NAME)
    if venv_path.exists():
        print_status(f"Removing existing {VENV_NAME} directory...")
        try:
            shutil.rmtree(venv_path)
            print_status(f"✓ Successfully removed {VENV_NAME}")
            return True
        except Exception as e:
            print_status(f"✗ Failed to remove {VENV_NAME}: {e}", "ERROR")
            return False
    else:
        print_status(f"No existing {VENV_NAME} directory found")
        return True

def create_venv(python_cmd):
    """Create a new Python 3.10 virtual environment."""
    print_status(f"Creating new {VENV_NAME} virtual environment...")
    
    try:
        result = subprocess.run([python_cmd, "-m", "venv", VENV_NAME], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print_status(f"✓ Successfully created {VENV_NAME}")
            return True
        else:
            print_status(f"✗ Failed to create {VENV_NAME}: {result.stderr}", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"✗ Exception during venv creation: {e}", "ERROR")
        return False

def get_activation_command():
    """Get the appropriate activation command for the current platform."""
    if sys.platform == "win32":
        return os.path.join(VENV_NAME, "Scripts", "activate.bat")
    else:
        return f"source {VENV_NAME}/bin/activate"

def get_pip_command():
    """Get the appropriate pip command for the current platform."""
    if sys.platform == "win32":
        return os.path.join(VENV_NAME, "Scripts", "pip.exe")
    else:
        return os.path.join(VENV_NAME, "bin", "pip")

def get_python_command():
    """Get the appropriate Python command for the virtual environment."""
    if sys.platform == "win32":
        return os.path.join(VENV_NAME, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_NAME, "bin", "python")

def upgrade_pip():
    """Upgrade pip to the latest version in the virtual environment."""
    print_status("Upgrading pip to latest version...")

    # Use the Python executable from the venv to upgrade pip (Windows requirement)
    python_cmd = get_python_command()

    try:
        result = subprocess.run([python_cmd, "-m", "pip", "install", "--upgrade", "pip"],
                              capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print_status("✓ Successfully upgraded pip")
            return True
        else:
            print_status(f"✗ Failed to upgrade pip: {result.stderr}", "ERROR")
            return False

    except Exception as e:
        print_status(f"✗ Exception during pip upgrade: {e}", "ERROR")
        return False

def install_requirements():
    """Install packages from requirements.txt file."""
    print_status(f"Installing packages from {REQUIREMENTS_FILE}...")
    
    # Check if requirements file exists.
    if not Path(REQUIREMENTS_FILE).exists():
        print_status(f"✗ {REQUIREMENTS_FILE} not found!", "ERROR")
        return False
    
    pip_cmd = get_pip_command()
    
    try:
        # Install with verbose output and timeout.
        result = subprocess.run([pip_cmd, "install", "-r", REQUIREMENTS_FILE, "-v"], 
                              capture_output=True, text=True, timeout=1800)  # 30 minutes timeout
        
        if result.returncode == 0:
            print_status("✓ Successfully installed all packages")
            return True
        else:
            print_status(f"✗ Package installation failed:", "ERROR")
            print_status(f"STDOUT: {result.stdout}", "ERROR")
            print_status(f"STDERR: {result.stderr}", "ERROR")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("✗ Package installation timed out (30 minutes)", "ERROR")
        return False
    except Exception as e:
        print_status(f"✗ Exception during package installation: {e}", "ERROR")
        return False

def verify_installation():
    """Verify that all packages are properly installed."""
    print_status("Verifying package installation...")
    
    pip_cmd = get_pip_command()
    
    try:
        result = subprocess.run([pip_cmd, "list"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print_status("✓ Package verification successful")
            print_status("Installed packages:")
            print(result.stdout)
            return True
        else:
            print_status(f"✗ Package verification failed: {result.stderr}", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"✗ Exception during package verification: {e}", "ERROR")
        return False

def setup_environment_with_retry():
    """Main function to set up the environment with retry logic."""
    print_header("PYTHON 3.10 VIRTUAL ENVIRONMENT SETUP")
    
    # Check Python 3.10 availability.
    python_cmd = check_python310()
    if not python_cmd:
        return False
    
    # Retry loop for environment setup.
    for attempt in range(1, MAX_RETRIES + 1):
        print_header(f"ATTEMPT {attempt} OF {MAX_RETRIES}")
        
        try:
            # Step 1: Remove existing environment.
            if not remove_venv():
                print_status(f"Attempt {attempt} failed at environment removal", "ERROR")
                if attempt < MAX_RETRIES:
                    print_status(f"Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
                continue
            
            # Step 2: Create new environment.
            if not create_venv(python_cmd):
                print_status(f"Attempt {attempt} failed at environment creation", "ERROR")
                if attempt < MAX_RETRIES:
                    print_status(f"Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
                continue
            
            # Step 3: Upgrade pip.
            if not upgrade_pip():
                print_status(f"Attempt {attempt} failed at pip upgrade", "ERROR")
                if attempt < MAX_RETRIES:
                    print_status(f"Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
                continue
            
            # Step 4: Install requirements.
            if not install_requirements():
                print_status(f"Attempt {attempt} failed at package installation", "ERROR")
                if attempt < MAX_RETRIES:
                    print_status(f"Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
                continue
            
            # Step 5: Verify installation.
            if not verify_installation():
                print_status(f"Attempt {attempt} failed at verification", "ERROR")
                if attempt < MAX_RETRIES:
                    print_status(f"Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
                continue
            
            # Success!
            print_header("SUCCESS!")
            print_status(f"✓ Virtual environment {VENV_NAME} created successfully!")
            print_status(f"✓ All packages installed and verified")
            
            # Print activation instructions.
            activation_cmd = get_activation_command()
            print_status("To activate the environment:")
            if sys.platform == "win32":
                print(f"    {activation_cmd}")
            else:
                print(f"    {activation_cmd}")
            
            return True
            
        except Exception as e:
            print_status(f"Unexpected error in attempt {attempt}: {e}", "ERROR")
            traceback.print_exc()
            if attempt < MAX_RETRIES:
                print_status(f"Waiting {RETRY_DELAY} seconds before retry...")
                time.sleep(RETRY_DELAY)
            continue
    
    # All attempts failed.
    print_header("FAILURE")
    print_status(f"✗ Failed to set up environment after {MAX_RETRIES} attempts", "ERROR")
    print_status("Please check the error messages above and resolve any issues", "ERROR")
    return False

if __name__ == "__main__":
    try:
        success = setup_environment_with_retry()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_status("\nSetup interrupted by user", "ERROR")
        sys.exit(1)
    except Exception as e:
        print_status(f"Unexpected error: {e}", "ERROR")
        traceback.print_exc()
        sys.exit(1)
