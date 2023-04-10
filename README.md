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
-   login to the server using ```ssh mlops-practice``` 