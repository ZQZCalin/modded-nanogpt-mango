# Some statistics of the runs

import wandb
import numpy as np

api = wandb.Api()

id_groups = {
    "seed_42": [
        "41b11b83-94ca-4248-8e83-ceb06ef6c384",
        "f56df529-9f5b-4315-8420-0b0da482274d",
        "273998b8-df48-44d3-9f8b-66aaf9f9492f",
        "c7527718-757c-4b58-beeb-2cce1575fe7b",
        "631fa0bc-48c7-4c55-a73e-52cea194be53",
    ],
    "diff_seed": [
        "41b11b83-94ca-4248-8e83-ceb06ef6c384",
        "ce1e3815-415f-40e7-bb7e-868b47036db5",
        "7574c444-3485-4d0e-b9cc-9d0192d6fd3c",
        "3ee7ed1b-3cbf-45f0-8997-c94a929d6b3b",
        "f0e0b321-d664-4f74-adcf-dd33881a8999",
    ],
    "record": [
        "85a1dd8a-f91b-40ae-b03f-7cb94a09e3cd",
        "e0cbc7c6-105b-4821-9737-3d030164fab4",
        "f226414f-3a55-4e4d-adab-160e68a10c17",
        "41ba3a84-cd9d-4292-87e3-abe5f026aff7",
        "80698b9f-4e4a-4047-8f84-58a2d19664e0",
    ],
}

def analyze(ids):
    num_logs = len(ids)
    val_loss = np.zeros(num_logs)
    for i, id in enumerate(ids):
        print(f"fetching {i+1}/{num_logs} wandb log...")
        run = api.run(f"optimizedlearning/nanogpt_speedrun/{id}")
        last_val_loss = [row["val_loss"] for row in run.scan_history(keys=["val_loss"])][-1]
        val_loss[i] = last_val_loss
    print(val_loss)
    print(f"mean = {np.mean(val_loss):.4f}, std = {np.std(val_loss):.4f}", 
        f"max-min = {np.max(val_loss)-np.min(val_loss):.4f}")

# -----------------------------------------------------------------------------
# Noise across different nodes with fixed random seed

print(f"\n{'='*100}")
print("Analyzing noise across nodes...")

ids = id_groups["seed_42"]
analyze(ids)

# -----------------------------------------------------------------------------
# Noise across different nodes with fixed random seed

print(f"\n{'='*100}")
print("Analyzing different random seed...")

ids = id_groups["diff_seed"]
analyze(ids)

# -----------------------------------------------------------------------------
# Reproducing record with different seeds

print(f"\n{'='*100}")
print("Analyzing 02/01 records...")

ids = id_groups["record"]
analyze(ids)
