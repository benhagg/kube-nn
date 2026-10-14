# Distributed nanoGPT training

Simulate a distributed training environment with a separate tokenizer Job and CPU training workers.

## Run with Minikube

Install Docker Desktop, Minikube, `kubectl`, and Helm, then start a local cluster:

This starts a local Minikube cluster. Run Helm and `kubectl` commands from your terminal; you do not need to enter the Minikube node.

```powershell
minikube start --driver=docker --cpus=4 --memory=8192
```
Once the container starts you can view all pods

```bash
kubectl get pods -A
```
![output of running get pods](images/image-1.png)

Build both images and load them into Minikube:

```powershell
docker compose build
$tokenizerImageHash = docker image inspect train-gpt-tokenizer:latest --format '{{.Id}}'
minikube image load train-gpt:latest
minikube image load train-gpt-tokenizer:latest
```
![shows all images loaded into minikube](images/image.png)



### Run distributed locally

Minikube can simulate a distributed run with multiple worker Pods on its single Kubernetes node. Set `nnodes: 2` (or more) in `k8s/nanogpt/values.yaml`, then rerun the Helm command below; Helm sets the same count for the StatefulSet and `torchrun`. The Pods share Minikube's CPU and memory, so keep the count within the resources allocated at startup.

Install or update the training and tokenizer resources with Helm:

```powershell
helm upgrade --install nanogpt .\k8s\nanogpt --set-string tokenizer.imageHash=$tokenizerImageHash
```

Helm creates a tokenizer Job for each image and tokenizer configuration. After rebuilding the tokenizer image, inspect its ID and pass the new value on the Helm command so Helm creates a new Job automatically. Changing tokenizer settings also creates a new Job. The trainer waits only until it finds a `*_train_*.npy` file in `/data`; it does not wait for a particular tokenizer run or shard layout. To scale the training workers, change `nnodes` in `k8s/nanogpt/values.yaml` and run the Helm command again with the same image hash.

Watch the Pods and logs:

```powershell
kubectl get pods -w
kubectl logs -f -l app=nanogpt-tokenizer
kubectl logs -f nanogpt-worker-0 -c trainer
```

The tokenizer writes train and validation `.npy` shards to the PVC. The training Pod waits for those files, then starts `torchrun`.
