# MarLo Handson

## Requirement

For the content of this repository, you need
- Python 3.5+ environment with
    - Chainer v5.0.0
    - CuPy v5.0.0
    - ChainerRL v0.4.0
    - marlo v0.0.1.dev23

To follow the instruction below, you need
- Azure subscription

## Setup

### 1. Install the Azure CLI tool

```
$ pip install azure-cli
```

### 2. Login to Azure using the Azure CLI

```
$ az login
```

### 3. Select a subscription

List up all the subscriptions you have by
```
$ az account list --all
```

Then, specify one of them with
```
$ az account set --subscription [A SUBSCRIPTION ID]
```
Of cource you need to replace `[A SUBSCRIPTION ID]` with a specific ID you want to use.

### 4. Launch a GPU VM

First, you have to create a resource group:
```
$ az group create -g marLo-handson -l eastus
```

Next, let's cerate a data science VM:
```
$ az vm create \
--resource-group marLo-handson \
--name vm \
--admin-username ${USER} \
--public-ip-address-dns-name ${USER} \
--image microsoft-ads:linux-data-science-vm-ubuntu:linuxdsvmubuntu:latest \
--size Standard_NC6 \
--generate-ssh-keys
```

Then, you will see the message like following:
```
{
  "fqdns": "[YOUR USERNAME].eastus.cloudapp.azure.com",
  "id": "/subscriptions/[YOUR SUBSCRIPTION ID]/resourceGroups/marLo-handson/providers/Microsoft.Compute/virtualMachines/vm",
  "location": "eastus",
  "macAddress": "AA-BB-CC-DD-EE-FF",
  "powerState": "VM running",
  "privateIpAddress": "10.0.0.4",
  "publicIpAddress": "123.456.78.910",
  "resourceGroup": "marLo-handson",
  "zones": ""
}
```
Please do not care some slight differences. All you need is the `publicIpAddress` of the created VM.

### 5. SSH to the VM

```
$ ssh [IP OF THE VM]
```

Please replace `[IP OF THE VM]` with your IP address you can find in the result of the previous step.

### 6. Create a Conda environment for MarLo

On the VM,
```
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
&& distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list \
&& sudo apt-get update \
&& sudo apt-get install -y libopenal-dev
```

```
$ conda config --set always_yes yes \
&& conda create python=3.6 --name marlo \
&& conda config --add channels conda-forge \
&& conda activate marlo \
&& conda install -c crowdai malmo matplotlib ipython numpy scipy opencv \
&& pip install git+https://github.com/crowdAI/marLo.git \
&& pip install chainer==5.1.0 cupy-cuda92==5.1.0 chainerrl==0.5.0
```

```
$ mkdir /anaconda/envs/marlo/Minecraft/run/config \
&& echo 'enabled=false' > /anaconda/envs/marlo/Minecraft/run/config/splash.properties
```

### 7. Start a Minecraft client

```
$ sudo docker run -it \
-p 5901:5901 \
-p 6901:6901 \
-p 8888:8888 \
-p 10000:10000 \
-e VNC_PW=vncpassword andkram/malmo 
```

Then re-SSH to the server with port forwarding of 6901 like this:

```
$ ssh [IP OF THE VM] -L 6901:localhost:6901
```

Then please open this URL with your browser: http://localhost:6901/?password=vncpassword

You'll see the virtual desktop and the Minecraft window in it.

![](images/minecraft.png)

## Start hands-on
### 0. Activate conda

```
$ conda info -e
# conda environments:
#
base                     /anaconda
marlo                 *  /anaconda/envs/marlo
py35                     /anaconda/envs/py35
py36                     /anaconda/envs/py36
```

If you do not acitivate `marlo` in your environment, please run following command.

```
$ conda activate marlo
```

### 1. ClonClone this repo

```
$ git clone https://github.com/keisuke-umezawa/marlo-handson.git
$ cd marlo-handson
```

### 2. Run simple malro example

```
$ python test_malmo.py
```

### 3. Run example with chainerrl

```
$ wget https://github.com/mitmul/marlo-handson/releases/download/v0.1/88626_except.tar.gz
$ tar xvzf 88626_except.tar.gz
$ python train_DQN.py --demo --load 88626_except --monitor
```

```
$ python make_video.py
$ ls video.mp4 
video.mp4
```

### 4. Train your models

```
$ python train_DQN.py --load 88626_except
```
