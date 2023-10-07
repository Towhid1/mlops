# 💁 Week #1

<!-- 💁👌🎍😍 -->

### Environment setup
1. Install pyenv (kivabe korben niche dewa ase)
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
6. Install all python required libs-
   ```sh 
    pip install -r requriements.txt
    ```


### 🐍 pyenv cheat-sheet
#### Installation of pyenv
##### Windows


- Open windows powershell and run this command:
    ```pwsh
    Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
    ```

- If you are getting any **UnauthorizedAccess** error as below then start Windows PowerShell with the **Run as administrator** option and run -
    ```pwsh
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
    ```
- Then re-run the previous powershell link code.
- For more details visit this [link](https://github.com/pyenv-win/pyenv-win/blob/master/docs/installation.md#powershell).
##### Linux
- If you wish to install a specific release of Pyenv rather than the latest head, set the PYENV_GIT_TAG environment variable (e.g. export `PYENV_GIT_TAG=v2.2.5`).
    ```sh
    curl https://pyenv.run | bash
    ```
- For more details visit this [link](https://github.com/pyenv/pyenv-installer).
#### some common command 
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
# 💁 Week 2-3

> All about EDA. Check notebook folder.

# 💁 Week 4

### Install & Run Apache Airflow with Docker 👌

1. Install [Docker](https://docs.docker.com/desktop/install/windows-install/).
2. Download apache airflow YAML file [Link](https://airflow.apache.org/docs/apache-airflow/2.6.3/docker-compose.yaml)
3. Update yaml file and add dockerfile (provided updated files in repo)
4. Init Airflow:
    ```sh
    docker-compose up airflow-init
    ```
5. Start Airflow services
    ```sh
    docker-compose up
    ```
6. Stop Airflow
    ```sh
    docker-compose down
    ```



# 💁 Week 5

## Important links 🔗
1. For Cornjob stuff : https://crontab.guru/
2. Scheduler Doc : https://airflow.apache.org/docs/apache-airflow/1.10.1/scheduler.html

### Python code format fixing using Black
1. install black
    ```sh
    pip install black
    ```
2. fix code format. black then code directory.
   ```sh
   black .\dags\airline_price.py
   ```

# Issues:
1. airflow docker-compose up issue. Showing this error message:

```sh
File "/usr/local/lib/python3.9/logging/config.py", line 571, in configure mlops-airflow-webserver-1 | raise ValueError('Unable to configure handler ' mlops-airflow-webserver-1 | ValueError: Unable to configure handler 'processor'
```

>Solution : create logs, dags, data, config, plugins folder manually.

# 💁 Week 6

## To get your access key ID and secret access key

1. Open the IAM console at https://console.aws.amazon.com/iam/.
2. On the navigation menu, choose Users.
3. Choose your IAM user name (not the check box).
4. Open the Security credentials tab, and then choose Create access key.
5. To see the new access key, choose Show. Your credentials resemble the following:
    ```
    Access key ID: AKIAIOSFODNN7EXAMPLE
    Secret access key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
    ```

6. To download the key pair, choose Download `.csv` file. Store the `.csv` file with keys in a secure location.

for more details : https://docs.aws.amazon.com/powershell/latest/userguide/pstools-appendix-sign-up.html

## MLFlow
We will use mlflow for experiment tracking and model deploying.
MLFlow official doc: https://mlflow.org/docs/latest/index.html

# 💁 Week 7

1. Updated dag
2. docker-compose YAML file updated
3. AWS credential added inside YAML file
4. RUN docker compose

### Runing MLFlow using docker (without Airflow)
1. To build the Docker image, navigate to the directory containing the Dockerfile and run the following command:
    ```sh
    docker build -t mlflow-server -f Docker-mlflow .
    ```
2. Once the image is built, you can run the MLflow server using the following command:
   ```sh
   docker run -p 5000:5000 -v mlflow:/mlflow mlflow-server

   ```

## Issue
> MLflow stopped working after docker-compose up without any error. But work properly without Airflow. 😟

**Solution:** Reason was there was no space in my C drive for running it. I reomved few big files. and restarted the docker. I worked with the issue. 


# 💁 Week 8

- We solved MLFlow docker issue
- Then we discuss about DVC
#### Commonly used DVC commands:
- DVC install
    ```sh
    pip install dvc
    ```
- Init DVC
  ```sh
  dvc init
  ```
- data versioning
  ```sh
  dvc add data/
  git add data.dvc
    ```
- model versioning
  ```sh
  dvc add models/
  ```

# Week #9

- Discussion about data drift
- Evidently code added into pipeline (https://www.evidentlyai.com/)
- requirement file updated

# Week #10

- Model deploy with Flask & Docker
- Prepare Docker-app to deploy flask app
- Added flask app code and requirement
- Test notebook added
### Runing Flask API using Docker
1. Build command:
    ```sh
    docker image build -t flaskapp -f .\Docker-app .
    ```
2. Run command:
    ```sh
    docker run -p 80:80 -t flaskapp
    ```
