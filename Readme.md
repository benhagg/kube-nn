# Distributed nanoGPT training

Simulate a distributed training environment with a separate tokenizer Job and CPU training workers.

## Run with Minikube

Install Docker Desktop, Minikube, and `kubectl`, then start a local cluster:

```powershell
minikube start --driver=docker --cpus=4 --memory=8192
```

Build both images and load them into Minikube:

```powershell
docker compose build
minikube image load train-gpt:latest
minikube image load train-gpt-tokenizer:latest
```

For a first run on Minikube's single node, set `replicas: 1` in `k8s/training.yaml` so it matches `NNODES=1` in `build-nanogpt/.env`. Also change the PVC access mode from `ReadWriteMany` to `ReadWriteOnce`; Minikube's default local storage supports the latter.

Apply the training resources, then run the tokenizer Job:

```powershell
kubectl apply -f k8s/training.yaml
kubectl apply -f k8s/tokenizer-job.yaml
```

Watch the Pods and logs:

```powershell
kubectl get pods -w
kubectl logs -f job/nanogpt-tokenize
kubectl logs -f nanogpt-worker-0 -c trainer
```

The tokenizer writes train and validation `.npy` shards to the PVC. The training Pod waits for those files, then starts `torchrun`.
