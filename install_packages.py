import subprocess

def install_packages(requirements_file):
    with open(requirements_file, encoding='utf-8-sig') as f:
        for line in f:
            package = line.strip()
            try:
                subprocess.check_call(["pip", "install", package])
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")

if __name__ == "__main__":
    requirements_file = "requirements.txt"
    install_packages(requirements_file)