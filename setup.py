import subprocess
import re
import os


def install_packages(requirements_file, bare_installation, installer):
    try:
        subprocess.check_call(["conda", "config", "--add", "channels", "conda-forge"])
    except subprocess.CalledProcessError as e:
        print(f"Error updating conda channels: {e}")

    error_list = ["Error installing the following packages, bare install executed:"]
    fail_list = ["Error installing the following packages, package installation failed:"]
    success_list = ["Successfully installed:"]

    with open(requirements_file, encoding='utf-8') as f:
        for line in f:
            package = line.strip()
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


if __name__ == "__main__":
    txt_files = [file for file in os.listdir() if file.endswith(".txt")]

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

    bare_install = input(
        "Do you want to perform a bare installation of the packages if pip fails to locate the specified version? (y/n): ")
    install_bare = bare_install.lower() == 'y'

    while True:
        print("Available installers:")
        print("1. pip")
        print("2. conda")
        choice = input("Enter the number of the installer you want to use: ")
        if choice == "1":
            installer = "pip"
            break
        elif choice == "2":
            installer = "conda"
            break
        else:
            print("Invalid choice, please select 1 or 2.")

    install_packages(requirements_file, install_bare, installer)