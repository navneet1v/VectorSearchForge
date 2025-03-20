### Benchmarking

This folder contains the code needed to benchmark GPU vs CPU build, using a docker container
on a GPU machine. This code does not use `opensearch-benchmark`, it uses a custom `main.py`
to run the experiments. 

### How to run the code

1. Make sure you `cd` into the `benchmarking` directory. 
2. Build the docker image on local computer: `docker build -t <your-dockerhub-repo>:<tag> . `
3. Push the docker image to your dockerhub repo: `docker push <your-dockerhub-repo>:<tag>`
4. Provision an EC2 instance with GPUs, with the `Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)`
    - Any instance from the `g5` family works
5. Connect to the EC2 instance in a terminal, then do `sudo bash`
6. Pull the docker image: `docker pull <your-dockerhub-repo>:<tag>`
   - One such docker image is here: `rchitale7/remote-index-build-service:vector-benchmark`
7. Create a directory in your root directory for docker mountpoint: `mkdir /docker-mountpoint`
8. Change the permissions on the directory: `chmod 777 /docker-mountpoint`
9. Run the docker container: `docker run -v /docker-mountpoint:/benchmarking/files --gpus all <your-dockerhub-repo>:<tag>`
    - One such docker image is here: `rchitale7/remote-index-build-service:vector-benchmark`
    - You can run the docker container in the background with `-d` option
10. All files will be stored in `/docker_mountpoint` directory