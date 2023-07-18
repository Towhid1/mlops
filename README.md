# 💁 Week #1

<!-- 💁👌🎍😍 -->

### Environment setup
1. Install pyenv 

2. Install python 3.8.6 using pyenv (Pyevn cheat sheet added below)
3. Install vscode
4. Install following extentions in vscode:
    ```sh
    chrmarti.regex
    donjayamanne.githistory
    dzhavat.bracket-pair-toggler
    eamodio.gitlens
    GrapeCity.gc-excelviewer
    humao.rest-client
    ionutvmi.path-autocomplete
    iterative.dvc
    mechatroner.rainbow-csv
    ms-azuretools.vscode-docker
    ms-python.autopep8
    ms-python.flake8
    ms-python.isort
    ms-python.pylint
    ms-python.python
    ms-python.vscode-pylance
    ms-toolsai.jupyter
    ms-toolsai.jupyter-keymap
    ms-toolsai.jupyter-renderers
    ms-toolsai.vscode-jupyter-cell-tags
    ms-toolsai.vscode-jupyter-slideshow
    ms-vscode-remote.remote-containers
    ms-vscode-remote.remote-ssh
    ms-vscode-remote.remote-ssh-edit
    ms-vscode.remote-explorer
    njpwerner.autodocstring
    PKief.material-icon-theme
    Shan.code-settings-sync
    shardulm94.trailing-spaces
    shd101wyy.markdown-preview-enhanced
    VisualStudioExptTeam.intellicode-api-usage-examples
    VisualStudioExptTeam.vscodeintellicode
    wayou.vscode-todo-highlight
    yzhang.markdown-all-in-one
    ```
5. Create virtual environment (kivabe korben niche dewa ase)


### 🐍 pyenv cheat-sheet
####Installation of pyenv
#####Windows 
Here, a github link is provided for installing pyenv easily with all provided instructions for windows users only.

https://github.com/pyenv-win/pyenv-win#power-shell
- go to pyenv-win commands of the given github link
- go to the powershell option for installation
- copy the 1st code link in the powershell
- open the windows powershell in the laptop and paste/run the code.
- if it shows UnauthorizedAccess error, copy and run the following code-
  ```sh
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
  ```
- then re-run the previous powershell link code.
- scroll below in the screen to find add system settings
- copy and run the 1st and 2nd link code of add system settings separately.
- And the installation is done!!
  Besides powershell, There are some other options available in this link for installing pyenv in windows.
#####Obuntu & Linux Mint
Here, Follow the given link to install pyenv in obuntu.
https://github.com/zaemiel/ubuntu-pyenv-installer
- Only for obuntu users, install curl using the following code. For Linux Mint users, this step doesn't require.
    ```sh
  sudo apt install curl
  ```
- Copy and run the code link below install pyenv headings and choose 3rd options.

####some common command 
Here's a cheat sheet of some commonly used commands with pyenv:

- To list all the available Python versions that can be installed with pyenv:

    ```sh
    pyenv install --list
    ```
- pyenv install: Install a specific Python version.


    ```sh
    pyenv install <version>
    ```
- pyenv versions: List all installed Python versions.

    ```sh
    pyenv versions
    ```
- pyenv global: Set the global Python version to be used.


    ```sh
    pyenv global <version>
    ```
- pyenv local: Set a Python version for the current directory.


    ```sh
    pyenv local <version>
    ```
- pyenv shell: Set a Python version for the current shell session.


    ```sh
    pyenv shell <version>
    ```

- pyenv uninstall: Uninstall a specific Python version.


    ```sh
    pyenv uninstall <version>
    ```
- pyenv rehash: Rehash the installed executables.


    ```sh
    pyenv rehash
    ```
- pyenv which: Display the full path to the executable of a Python version.

    ```sh
    pyenv which <version>
    ```
- pyenv exec: Run a command using a specified Python version.

    ```sh
    pyenv exec <version> <command>
    ```
### 🌱 Creation of virtual environments

1. Install and create virtual environment:
    ```sh
    pip install virtualenv
    python -m venv mlops_env
    ```


2. Activate it:
    Linux:
    ```sh
    source mlops_env/bin/activate
    ```
    Windows:
    ```
    C:\Users\path\mlops_env\Scripts\activate
    ```
###Requirements.txt 
- To list down the all the required libraries for running the code, the following code is required to run in the terminal
    ```sh 
    pip freeze > requriements.txt
    ```
- To install the all the libraries in the requirements.txt file in one go, the following code is needed to run in the terminal of vscode.
    ```sh 
    pip install -r requriements.txt
    ```