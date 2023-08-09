import subprocess
import re

def install_packages(requirements_file):
    with open(requirements_file) as f:
        for line in f:
            package = line.strip()
            package_name = re.match(r'^[a-zA-Z0-9_-]+', package).group()
            try:
                subprocess.check_call(["pip", "install", package])
                print(package_name)
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")
                try:
                    print("Attempting bare installation of the package")
                    subprocess.check_call(["pip", "install", package_name])
                except subprocess.CalledProcessError as e:
                    print(f"Error installing {package}: {e}")


if __name__ == "__main__":
    requirements_file = "requirements_linux.txt"
    install_packages(requirements_file)