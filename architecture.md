# Distributed nanoGPT training architecture

One training run has four worker Pods, each running one nanoGPT container with four processes.

```mermaid
flowchart TB
    Submit[Submit training run] --> Controller[Kubernetes job controller]
    Controller --> Workers

    subgraph Workers[Training worker Pods]
      direction TB
      subgraph TopRow[ ]
        direction LR
        P0[Pod 0<br/>nanoGPT<br/>4 processes] ~~~ P1[Pod 1<br/>nanoGPT<br/>4 processes]
      end
      subgraph BottomRow[ ]
        direction LR
        P2[Pod 2<br/>nanoGPT<br/>4 processes] ~~~ P3[Pod 3<br/>nanoGPT<br/>4 processes]
      end
      P0 ~~~ P2
      P1 ~~~ P3
    end

    Data[(Training data)] --> Workers
    Workers --> Checkpoints[(Checkpoint storage)]
```

Workers read the shared dataset and save checkpoints to durable storage. Gloo runs inside the worker processes for CPU training; GPU training typically uses NCCL.
