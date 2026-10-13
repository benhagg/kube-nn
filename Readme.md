# Distributed nanoGPT training

Simulate a distributed training environment with a separate tokenizer Job and CPU training workers.

## Run with Minikube

Install Docker Desktop, Minikube, `kubectl`, and Helm, then start a local cluster:

This starts a local kubernetes container and loads the train and tokenize images into it. after this you can run kubernetes commands from your terminal and it will execute within the minikube pod. I have aliased the kubectl command with "k" so you will see that in the images.

```powershell
minikube start --driver=docker --cpus=4 --memory=8192
```
Once the container starts you can view all pods

```bash
kubectl get pods -A
```

Build both images and load them into Minikube:

```powershell
docker compose build
minikube image load train-gpt:latest
minikube image load train-gpt-tokenizer:latest
```

![output of running get pods](image-1.png)

The chart's `nnodes` value controls both the worker replica count and `torchrun`'s `NNODES`. For a first run on Minikube's single node, leave `nnodes: 1` in `k8s/nanogpt/values.yaml`. The chart uses `ReadWriteOnce` for Minikube's default local storage.

### Run distributed locally

Minikube can simulate a distributed run with multiple worker Pods on its single Kubernetes node. Set `nnodes: 2` (or more) in `k8s/nanogpt/values.yaml`, then rerun the Helm command below; Helm sets the same count for the StatefulSet and `torchrun`. The Pods share Minikube's CPU and memory, so keep the count within the resources allocated at startup.

Install or update the training resources with Helm, then run the tokenizer Job:

```powershell
helm upgrade --install nanogpt .\k8s\nanogpt
kubectl apply -f k8s/tokenizer-job.yaml
```

To scale the training workers, change `nnodes` once in `k8s/nanogpt/values.yaml` and run the Helm command again.

Watch the Pods and logs:

```powershell
kubectl get pods -w
kubectl logs -f job/nanogpt-tokenize
kubectl logs -f nanogpt-worker-0 -c trainer
```

The tokenizer writes train and validation `.npy` shards to the PVC. The training Pod waits for those files, then starts `torchrun`.
