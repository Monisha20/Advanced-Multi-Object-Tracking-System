import os
import subprocess
import sys

def create_virtual_env(env_name="venv"):  # Changed default to 'venv'
    # Check if the environment already exists
    if not os.path.exists(env_name):
        # Use Python directly to create a virtual environment
        print(f"Creating virtual environment '{env_name}'...")
        subprocess.call([sys.executable, '-m', 'venv', env_name])
        print(f"Virtual environment '{env_name}' created.")
    else:
        print(f"Virtual environment '{env_name}' already exists.")

def install_requirements(env_name="venv"):  # Changed default to 'venv'
    # Define the path to the pip executable within the virtual environment
    if os.name == 'nt':  # For Windows
        pip_executable = os.path.join(env_name, 'Scripts', 'pip.exe')
    else:  # For macOS/Linux
        pip_executable = os.path.join(env_name, 'bin', 'pip')

    # Install the required libraries using pip
    print("Installing libraries from requirements.txt...")
    subprocess.call([pip_executable, 'install', '-r', 'requirements.txt'])
    print("Libraries installed from requirements.txt.")

if __name__ == "__main__":
    create_virtual_env()
    install_requirements()