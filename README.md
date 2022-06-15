# pytorch_oom_example
minimum example of OOM Error with PyTorch Net

# Clone repo from Github 
```
git clone https://github.com/david-engelmann/pytorch_oom_example.git
```

# Build Docker Image
From the folder that you cloned the repo into run the following command to build the docker image

```
docker build -t pytorch_oom_example:1 .
```

# Start the Container
From Docker Desktop / cli create a container with the image
* From the Docker Desktop, navigate to the images section and press run on the image created in the previous step
* From the command line, use the `docker run` command to start a container ie.

```
docker run pytorch_oom_example:1
```

# Execute the train.py file
Once the container is up and running, get a cli from the container.
* From Docker Desktop, go to the containers section, select the container you started in the previous setup and click on the ">_" / cli button to get a command line from within the container
* From the command line, use `docker ps` to get the ContainerID, then run the `docker exec -it` command to get a command line from within the container

```
docker exec -it ContainerID /bin/sh
```

Once you are in the command line of the container, run the following command to trigger the OOM memory error.

```
python train.py
```
