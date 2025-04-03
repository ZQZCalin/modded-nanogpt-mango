# Load checkpoints, and for each checkpoint, examine the gradient noise.
import torch
from torch import nn
import wandb
import os
from pathlib import Path
import glob
from dataclasses import dataclass
from functools import lru_cache
from load_model import GPT
from tqdm import tqdm

DIR = "checkpoints/muon_baseline"
PROJ = "QZ_test_grad_noise"
NAME = "muon_noise"
KEYS = []
SAVE_STATS = True

def get_all_checkpoints(directory, file_type=".pt"):
    res = []
    for file in os.listdir(directory):
        if file.endswith(file_type):
            res.append(file)
    return res

def log_wandb(metrics):
    wandb.init(project=PROJ, name=NAME)
    raise NotImplementedError

#------------------------------------------------------------------------------
# Imported from train script.
@dataclass
class Hyperparameters:
    # data
    data_dir: str = "/projectnb/aclab/datasets"
    train_files = "fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 48*1024 # FlexAttention sequence length
    # val_seq_len = 4*64*1024 # FlexAttention sequence length for validation
    val_seq_len = 1024
    # optimization
    num_iterations = 1770 # number of iterations to run
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    # checkpoint
    save_checkpoint = True
    checkpoint_every = 125
    checkpoint_path = "checkpoints/muon_baseline"

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets

@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)
def get_window_size_blocks(step: int, args):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

#------------------------------------------------------------------------------
# Gradient noise validation function.
def val(model, step, args):
    rank = 0
    world_size = 1

    # For now, we keep eval mode; not sure if modded GPT behaves differently for train vs eval.
    model.eval()
    
    val_batch_size = world_size * args.val_seq_len
    assert args.val_tokens % val_batch_size == 0
    val_steps = args.val_tokens // val_batch_size
    val_loader = distributed_data_generator(
        os.path.join(args.data_dir, args.val_files), val_batch_size, rank, world_size)
    val_loss = 0

    print("Intializing mean and covariance")
    mean = {}
    covariance = {}
    for name, param in model.named_parameters():
        mean[name] = torch.zeros_like(param)
        if param.ndim <= 1:
            covariance[name] = torch.zeros_like(param)
        elif param.ndim == 2:
            r, c = param.size()
            covariance[f"{name}-L"] = torch.eye(r)
            covariance[f"{name}-R"] = torch.eye(c)
        elif param.ndim == 3:
            r, c = param.size(-2), param.size(-1)
            for i in range(param.size(0)):
                covariance[f"{name}_{i}-L"] = torch.eye(r)
                covariance[f"{name}_{i}-R"] = torch.eye(c)
        else:
            raise RuntimeError(f"{name} has dimension {param.ndim}, hence raising error.")

    print("First pass: computing mean")
    pbar = tqdm(range(val_steps), total=val_steps)
    for i in pbar:
        pbar.set_description(f"parsing step={i+1}/{val_steps}")
        model.zero_grad(set_to_none=True)
        inputs, targets = next(val_loader)
        loss = model(inputs, targets, get_window_size_blocks(step, args)) / val_steps
        val_loss += loss.detach()
        # Update mean and covariance
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None
            mean[name].add_(param.grad)

    print("Second pass: estimating covariance (left-right covariance for matrix gradients)")
    pbar = tqdm(range(val_steps), total=val_steps)
    for i in pbar:
        pbar.set_description(f"parsing step={i+1}/{val_steps}")
        model.zero_grad(set_to_none=True)
        inputs, targets = next(val_loader)
        loss = model(inputs, targets, get_window_size_blocks(step, args)) / val_steps
        val_loss += loss.detach()
        # Update mean and covariance
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None
            mean[name].add_(param.grad)
    del val_loader
    del val_loader
    return mean, covariance

if __name__ == "__main__":
    print("Examining gradient noise distributions...")
    
    checkpoints = get_all_checkpoints(DIR)
    metrics = {k: torch.zeros(len(checkpoints)) for k in KEYS}
    args = Hyperparameters()

    for file in checkpoints:
        ckpt = torch.load(os.path.join(DIR, file))
        step = ckpt["step"]
        print(f"Iteration = {step}...")

        # Loading model weights from checkpoint.
        model = GPT(vocab_size=args.vocab_size, num_layers=12, num_heads=6, model_dim=768, 
                    max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
        model = torch.compile(model, dynamic=False)
        model.load_state_dict(ckpt["model"])
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.bfloat16()

        mean, covariance = val(model, step, args)
        break
        if SAVE_STATS:
            torch.save(dict(step=step, mean=mean, covariance=covariance))
