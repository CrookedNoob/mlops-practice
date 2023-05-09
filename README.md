# MLOPS Process End to End

### Setup EC2 Server
-   select an Ubuntu server for 16GB RAM and atleast 30GB of storage
-   For Windows, create a config file:

    ```Host mlops-practice
        HostName ec2-xx-X-xx-xx.ap-region-1.compute.amazonaws.com
        User ubuntu
        IdentityFile C:/Users/username/.ssh/mlops-pemfile.pem
        StrictHostKeyChecking no
    ```    

### Setup Anaconda on EC2
-   login to the server using ```ssh mlops-practice``` 
-   update ```sudo apt update```
-   download anaconda ```wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh```
-   install anaconda ```bash Anaconda3-2023.03-Linux-x86_64.sh```
-   Create an Anaconda environment - ```conda create -n mlops python=3.10.9``` and activate the same ```conda activate mlops```

### Install Docker and Docker Compose on EC2
-   create a separate folder (```soft```) for other tools
-   install docker 
    ```
    cd soft
    sudo apt install docker.io
    cd ..
    ```
-   install docker compose from github page ```wget https://github.com/docker/compose/releases/download/v2.17.2/docker-compose-linux-x86_64```
-   make the docker-compoise zip executable ```chmod +x docker-compose-linux-x86_64```
-   set environment path ```nano .bashrc``` and add at the end ```export PATH="${HOME}/soft:${PATH}"```
-   renameing docker compose for ease ```mv docker-compose-linux-x86_64 docker-compose```
-   check if rename and executable file conversion is done proberly ```which docker-compose```
-   run the codes to avoid using sudo for docker ```sudo groupadd docker``` and ```sudo usermod -aG docker $USER```
-   test docker ```docker run hello-world```

### Setup Github on EC2
-   setup github ```git config --global user.name "user_name"``` and ```git config --global user.email "mail_id@domain.com"```

### Setup VSCode on local to access the EC2 server
-   in VSCode, setup remote server using *Connect to Host* and add the ```mlops-practice``` config file 

### MLFlow setup on EC2
-   install mlflow ```pip install mlflow```
-   initialte mlflow from the folder ```mlops``` so that it can be used for other projects as well
-   can install other requirements using ```pip install -r requirements.txt```
-   run mlflow using ```mlflow ui --backend-store-uri sqlite:///mlflow.db``` to store all the artifacts and meta data on the sqlite db *mlflow.db*
-   for test, run ```python 2_experiment_tracking/main.py```
-   check the mlflow url

<img width="946" alt="image" src="https://user-images.githubusercontent.com/13174586/231063714-d8934fc1-f83b-412b-8d4d-8de47f3b1626.png">

### Using MLFlow
-   use the below codes to initialize the database to store metadata and setup the name of experiment to track
    ```
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment-1")
    ```

### create EC2 server for MLflow model registry and tracking
-   Use linux free tier server
-   update the server ```sudo yum update```
-   in case **pip3** is not available ```sudo yum update python3-pip```
-   install dependencies and packages - ```pip3 install boto3 mlflow psycopg2-binary```
-   setup the instance with credentials ```aws configure```
    ```
    AWS Access Key ID [None]: xxxxxxxxxxxxxxxxxxx
    AWS Secret Access Key [None]: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    Default region name [None]: us-east-2
    Default output format [None]:
    ```

-   check the list of **s3 buckets** available ```aws s3 ls```

-   setup the database
    -   go to *AWS RDS* and click on `Create Database`
    -   select **PostgresSQL** and **Free Tier** from Template
    -   update the below items
    ```
    DB instance identifier : database-1
    Master username : postgres
    Master password : *************
    uncheck Enable storage autoscaling
    VPC security group (firewall): Choose existing
    Existing VPC security groups: lkaunch-wizard-xy
    ```
-   upload `config` and `credentials` files from local system to ec2(used for model building)
    `scp -i mlops-ec2-practice.pem C:\Users\myname\.aws\config  ubuntu@ec2-xx-xxx-xxx-xxx.ap-south-1.compute.amazonaws.com:~/.`
    `scp -i mlops-ec2-practice.pem C:\Users\myname\.aws\credentials  ubuntu@ec2-xx-xxx-xxx-xxx.ap-south-1.compute.amazonaws.com:~/.`
-   copy these files to .aws folder `mv config ~/.aws/` and `mv credentials ~/.aws/`
-   run the below code to setup [aws reference](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)
    ```
    aws ec2 describe-instances --profile "soumyadip"
    export AWS_PROFILE=soumyadip
    ```
-   change the inbound rules of the mlflow tracking ec2 server  using **Edit Inbound Rules** and add the Public IP4 adress of the Model training server  
    <img width="395" alt="image" src="https://user-images.githubusercontent.com/13174586/236794859-3653696d-f63e-42e5-ab38-662923a2c13d.png">
    add both SSH and TCP for both local and ML model EC2 servers

-   in the *MLflow tracking server* run `mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/DB_NAME --default-artifact-root s3://S3_BUCKET_NAME`

```
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://mlflow:namep@mlflow-database.xzyasdf.us-east-3.rds.amazonaws.com:5432/mlflow_db --default-artifact-root s3://artifacts-remote
```
-   test using `3_model_registry/tracking_server.ipynb` and experiment details will be updated on the mlflow experiment UI
-   refer `3_model_registry/models_tracking.py` to train multiple models

