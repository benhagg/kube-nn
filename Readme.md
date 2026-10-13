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
minikube image load train-gpt:latest
minikube image load train-gpt-tokenizer:latest
```
![shows all images loaded into minikube](images/image.png)



### Run distributed locally

Minikube can simulate a distributed run with multiple worker Pods on its single Kubernetes node. Set `nnodes: 2` (or more) in `k8s/nanogpt/values.yaml`, then rerun the Helm command below; Helm sets the same count for the StatefulSet and `torchrun`. The Pods share Minikube's CPU and memory, so keep the count within the resources allocated at startup.

Install or update the training and tokenizer resources with Helm:

```powershell
helm upgrade --install nanogpt .\k8s\nanogpt
```

Helm creates one tokenizer Job for each partition configuration and reuses it on upgrades with unchanged settings. Changing tokenizer settings or incrementing `tokenizer.inputRevision` creates a new Job and partition; the tokenizer skips work only when the source, settings, and existing shards match that partition. To scale the training workers, change `nnodes` in `k8s/nanogpt/values.yaml` and run the Helm command again.

Watch the Pods and logs:

```powershell
kubectl get pods -w
kubectl logs -f -l app=nanogpt-tokenizer
kubectl logs -f nanogpt-worker-0 -c trainer
```

The tokenizer writes train and validation `.npy` shards to the PVC. The training Pod waits for those files, then starts `torchrun`.
