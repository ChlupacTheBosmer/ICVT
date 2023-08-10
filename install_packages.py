import subprocess
import re
import os

def install_packages(requirements_file, bare_installation, installer):
    with open(requirements_file) as f:
        for line in f:
            package = line.strip()
            package_name = re.match(r'^[a-zA-Z0-9_-]+', package).group()
            try:
                subprocess.check_call(["pip", "install", package])
                print(package_name)
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")
                if bare_installation:
                    try:
                        print("Attempting bare installation of the package")
                        subprocess.check_call([installer, "install", package_name])
                    except subprocess.CalledProcessError as e:
                        print(f"Error installing {package}: {e}")


if __name__ == "__main__":

    # Get a list of .txt files in the current directory
    txt_files = [file for file in os.listdir() if file.endswith(".txt")]

    # Print the list of .txt files
    print("Available .txt files:")
    for i, txt_file in enumerate(txt_files, start=1):
        print(f"{i}. {txt_file}")

    # Ask the user for input to choose a .txt file
    while True:
        try:
            user_choice = int(input("Enter the number of the .txt file you want to use: "))
            if 1 <= user_choice <= len(txt_files):
                requirements_file = txt_files[user_choice - 1]
                break
            else:
                print("Invalid input. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Ask whether run bare installation in case of failure
    bare_install = input(f"Do you want to perform a bare installation of the packages if pip fails to locate the specified version? (y/n): ")
    if bare_install.lower() == 'y':
        install_bare = True
    else:
        install_bare = False

    # Ask whether to use conda or pip
    print(f"Available installers:")
    print(f"1. pip")
    print(f"2. conda")
    conda_or_pip = input(
        f"Enter the number of the installer you want to use: ")
    if conda_or_pip.lower() == "1":
        installer = "pip"
    elif conda_or_pip.lower() == "2":
        installer = "conda"

    # Run the installation
    install_packages(requirements_file, install_bare, installer)
