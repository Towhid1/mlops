# Important commands and others

### Airflow running

1. Init Airflow:
    ```sh
     docker-compose up airflow-init
     ```
2. Start Airflow services
    ```sh
     docker-compose up
     ```
     if you don't want to display any output from docker compose execution/running then run this command:
    ```sh
     docker-compose up -d
     ```
3. Access Airflow UI : http://localhost:8080/
4. Default **password and username** : `airflow`
5. You can even enter the worker container so that you can run airflow commands using the following command. You can find `<container-id>` for the Airflow worker service by running `docker ps`.
   ```sh
   docker exec -it <container-id> bash
   ```
6. Once you are done with your experimentation, you can clean up the mess we’ve just created by simply running
    ```sh
    docker-compose down --volumes --rmi all
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

### Runing MLFlow using docker (without Airflow)
1. To build the Docker image, navigate to the directory containing the Dockerfile and run the following command:
    ```sh
    docker build -t mlflow-server -f Docker-mlflow .
    ```
2. Once the image is built, you can run the MLflow server using the following command:
   ```sh
   docker run -p 5000:5000 -v mlflow:/mlflow mlflow-server

   ```