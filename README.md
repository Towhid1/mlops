# Week #1

### Environment setup
1. Install pyenv (youtube e tutorial paben)
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


### pyenv cheat-sheet
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
### Creation of virtual environments

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

