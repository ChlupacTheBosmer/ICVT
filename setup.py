# def install_packages(requirements_file, bare_installation, installer):
#     try:
#         subprocess.check_call(["conda", "config", "--add", "channels", "conda-forge"])
#     except subprocess.CalledProcessError as e:
#         print(f"Error updating conda channels: {e}")
#
#     error_list = ["Error installing the following packages, bare install executed:"]
#     fail_list = ["Error installing the following packages, package installation failed:"]
#     success_list = ["Successfully installed:"]
#
#     with open(requirements_file, encoding='utf-8') as f:
#         for line in f:
#             package = line.strip()
#             package_name = re.match(r'([^=]+)', package).group()
#             try:
#                 subprocess.check_call([installer, "install", package])
#                 success_list.append(f"{package_name}")
#             except subprocess.CalledProcessError as e:
#                 print(f"Error installing {package}: {e}")
#                 if bare_installation:
#                     try:
#                         print("Attempting bare installation of the package")
#                         subprocess.check_call([installer, "install", package_name])
#                         error_list.append(f"{package_name}")
#                     except subprocess.CalledProcessError as e:
#                         print(f"Error installing {package}: {e}")
#                         fail_list.append(f"{package_name}")
#
#     for lst in [success_list, error_list, fail_list]:
#         if len(lst) > 1:
#             for line in lst:
#                 print(line)
#
#
# if __name__ == "__main__":
#     txt_files = [file for file in os.listdir() if file.endswith(".txt")]
#
#     print("Available .txt files:")
#     for i, txt_file in enumerate(txt_files, start=1):
#         print(f"{i}. {txt_file}")
#
#     while True:
#         try:
#             user_choice = int(input("Enter the number of the .txt file you want to use: "))
#             if 1 <= user_choice <= len(txt_files):
#                 requirements_file = txt_files[user_choice - 1]
#                 break
#             else:
#                 print("Invalid input. Please enter a valid number.")
#         except ValueError:
#             print("Invalid input. Please enter a valid number.")
#
#     bare_install = input(
#         "Do you want to perform a bare installation of the packages if pip fails to locate the specified version? (y/n): ")
#     install_bare = bare_install.lower() == 'y'
#
#     while True:
#         print("Available installers:")
#         print("1. pip")
#         print("2. conda")
#         choice = input("Enter the number of the installer you want to use: ")
#         if choice == "1":
#             installer = "pip"
#             break
#         elif choice == "2":
#             installer = "conda"
#             break
#         else:
#             print("Invalid choice, please select 1 or 2.")
#
#     install_packages(requirements_file, install_bare, installer)

import subprocess
import json
import sys
import re
import os


def install_submodule(submodule_path):
    global bare_install
    global installer
    print(f"Initializing submodule at {submodule_path}")
    subprocess.run(["git", "submodule", "update", "--init", submodule_path])

    print(f"Installing dependencies for submodule at {submodule_path}")
    requirements_file = select_requirements_file(submodule_path)
    install_packages(requirements_file, bare_install, installer)

    # # Check if this submodule has its own submodule dependencies
    # modules_json_path = f"{submodule_path}/modules.json"
    # try:
    #     with open(modules_json_path, "r") as f:
    #         submodule_config = json.load(f)
    #
    #     for nested_submodule_path in submodule_config.get("submodules", []):
    #         install_submodule(nested_submodule_path)
    #
    # except FileNotFoundError:
    #     print(f"No modules.json found for submodule at {submodule_path}. Skipping nested submodules.")

def install_basic_requirements():
    print("Installing basic requirements...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

def install_packages(requirements_file, bare_installation, installer):

    if installer == "conda":
        try:
            subprocess.check_call(["conda", "config", "--add", "channels", "conda-forge"])
        except subprocess.CalledProcessError as e:
            print(f"Error updating conda channels: {e}")

    error_list = ["Error installing the following packages, bare install executed:"]
    fail_list = ["Error installing the following packages, package installation failed:"]
    success_list = ["Successfully installed:"]

    with open(requirements_file, encoding='utf-8') as f:
        for line in f:
            package = line.strip().split('#')[0]
            package_name = re.match(r'([^=]+)', package).group()
            try:
                subprocess.check_call([installer, "install", package])
                success_list.append(f"{package_name}")
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")
                if bare_installation:
                    try:
                        print("Attempting bare installation of the package")
                        subprocess.check_call([installer, "install", package_name])
                        error_list.append(f"{package_name}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error installing {package}: {e}")
                        fail_list.append(f"{package_name}")

    for lst in [success_list, error_list, fail_list]:
        if len(lst) > 1:
            for line in lst:
                print(line)

def select_requirements_file(directory_path: str = ""):

    # Get txt files in the relevant folder
    if directory_path == "":
        directory_path = os.path.dirname(os.path.abspath(__file__))
    txt_files = [file for file in os.listdir(directory_path) if file.endswith(".txt")]

    print("Available .txt files:")
    for i, txt_file in enumerate(txt_files, start=1):
        print(f"{i}. {txt_file}")

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

    return requirements_file

def select_bare_install():
    bare_install = input(
        "Do you want to perform a bare installation of the packages if pip fails to locate the specified version? (y/n): ")
    install_bare = bare_install.lower() == 'y'
    return install_bare

def select_installer():

    # Set default
    conda = False
    installer = "pip"

    # Check whether conda is available
    if 'CONDA_PREFIX' in os.environ:
        conda = True

    while True:
        print("Available installers:")
        print("1. pip")
        if conda:
            print("2. conda")
        choice = input("Enter the number of the installer you want to use: ")
        if choice == "1":
            break
        elif choice == "2" and conda:
            installer = "conda"
            break
        else:
            print("Invalid choice, please select 1 or 2.")

    return installer

def load_json(filepath: str = "modules.json"):

    # Load JSON config
    with open(filepath, "r") as f:
        json_content = json.load(f)

    return json_content

def select_app():

    # Get the config for the different apps
    module_config = load_json()

    # Get user input for the app
    available_apps = list(module_config.keys())
    print(f"Available apps are: {', '.join(available_apps)}")
    selected_app = input("Which app do you want to set up? ")

    # Check if selected app exists in the JSON config
    if selected_app in module_config:
        for submodule_path in module_config[selected_app]:
            install_submodule(submodule_path)
    else:
        print(f"Invalid app name. Available apps are: {', '.join(available_apps)}")

if __name__ == "__main__":
    print("Welcome to ICVT Setup")
    global bare_install
    global installer

    # Install basic requirements first
    should_install_basic = input("Do you want to install basic requirements? (y/n): ")
    if should_install_basic.lower() == "y":

        requirements_file = select_requirements_file()
        bare_install = select_bare_install()
        installer = select_installer()
        install_packages(requirements_file, bare_install, installer)

    else:
        sys.exit("Setup terminated")

    select_app()

