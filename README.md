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
