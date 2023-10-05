# ICVT

## Prerequisites

Before running the script, ensure you have the following installed:

- Python 3.10+
- `pip` or `conda` package manager
- Git

## Optional: Create a Conda Environment

Before running the script, you may want to create a new Conda environment to isolate your project dependencies. To create a new environment:

1. Open your terminal and run:

    ```bash
    conda create --name my_icvt_env python=3.x
    ```

    Replace `my_icvt_env` with your preferred environment name and `3.x` with your desired Python version (ideally 3.11).

2. Activate the environment:

    ```bash
    conda activate my_icvt_env
    ```

## Clone the Repository

1. To clone the repository, run the following command:

    ```bash
    git clone https://github.com/ChlupacTheBosmer/ICVT
    ```

2. Navigate into the project directory:

    ```bash
    cd ICVT
    ```

### Optional: Switch to the Dev Branch

If you would like to work with the latest, potentially unstable version, you can switch to the `dev` branch:

```bash
git checkout dev
git switch dev
git pull origin dev
```

## Run the Setup Script

### Navigate to the Directory
Open a terminal and navigate to the directory containing the `setup.py` script.

```bash
cd /path/to/repo/ICVT/
```
### Run the Script
Execute the script using Python.

```bash
python setup.py
```

### Install Basic Requirements
The script will first ask if you want to install basic requirements. Enter y for Yes or n for No. This refers to the requirements that are shared by all the basic launcher scripts and are mandatory.

### Choose a Requirements File
If you opt to install basic requirements, you'll be presented with a list of .txt files. Choose the one appropriate for your setup.

### Bare Installation
You'll be asked whether you'd like to perform a "bare installation" in case of failure to install a specific version of the package set in the requirements file. This means installing the package without a specific version. Enter y for Yes or n for No.

### Select Installer
Choose between pip (recommended) and conda for package installation. Enter the corresponding number to select.

### Select Application
Finally, select the application for which you want to set up submodules and dependencies. For example, write "ICCS" to install all required components to run the ICCS app without cluttering your disc and environment with the other apps' requirements.

### Monitor the Output
The script will initialize submodules and install the necessary packages, providing output logs for your review.

## Error Handling
If the script encounters an issue during package installation, it will either attempt a "bare installation" if permitted or log the failure for you to review. Typically, when using conda for installation, some packages will not be retrieved as they might only be accessible via pip index. In that case, you might want to repeat the setup process using the alternative installer.

By following these steps, you should be able to successfully execute the ICVT Setup Script to initialize and set up your application and its submodules.

![iccs_arrange](https://github.com/ChlupacTheBosmer/ICVT/assets/29023670/9af2387c-70cd-49ac-b396-fbc55d4fa1c4)
![iclv_arrange](https://github.com/ChlupacTheBosmer/ICVT/assets/29023670/41b5e4f5-386b-4b97-a81f-eb2ead563a15)
![icid_arrange](https://github.com/ChlupacTheBosmer/ICVT/assets/29023670/13d8b13e-132e-4822-afef-f3055726660f)
