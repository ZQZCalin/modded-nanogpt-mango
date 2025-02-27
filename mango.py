"""
PyTorch implementation of Mango.

Adapted from the JAX implementation here:

https://github.com/ZQZCalin/trainit/blob/master/optimizers/muon/mango.py
"""

import math
import torch
from torch.optim.optimizer import Optimizer


# --- Helper functions (analogous to the various scale_by_* transforms) --- #

def mango_label_gpt(name, param):
    """
    Assign a Mango label based on a parameter’s name and shape.
    This is similar to your JAX tree-map function.
    """
    parts = name.split(".")
    if "token_embedding" in parts or "position_embedding" in parts:
        return "embedding"
    if "head" in parts:
        return "head"
    if "attn_fc" in parts:
        if param.dim() == 2:
            return "attn_w"
        elif param.dim() == 1:
            return "attn_b"
    if param.dim() == 2:
        return "mat"
    if param.dim() == 1:
        if "weight" in parts:
            return "vec_w"
        if "bias" in parts:
            return "vec_b"
    raise ValueError(f"cannot categorize parameter: {name}")


def group_parameters_by_mango_label(model, normalizations=None):
    """
    Helper to split model parameters into groups based on mango_label_gpt.
    Each group is given a key 'mango_label' that will later determine which
    normalization transform to use.
    """
    groups = {}
    for name, param in model.named_parameters():
        # Skip frozen parameters.
        if not param.requires_grad:
            continue
        label = mango_label_gpt(name, param)
        if label not in groups:
            groups[label] = {'params': [], 'mango_label': label}
        groups[label]['params'].append(param)
    return list(groups.values())


def split_vmap(g, num_heads, f):
    """
    Mimics the JAX split_vmap: reshape a 1D or 2D tensor g whose first dimension is divisible
    by (3 * num_heads), apply f on each “slice” (the last axis) and reshape back.
    (This implementation uses Python loops for clarity; for performance you might vectorize.)
    """
    if g.dim() not in (1, 2):
        raise ValueError("split_vmap supports only 1D or 2D tensors")
    total = g.size(0)
    d = total // (3 * num_heads)
    if g.dim() == 1:
        g_reshaped = g.view(3, num_heads, d)
        for i in range(3):
            for j in range(num_heads):
                # f is expected to map a 1D tensor (of length d) to one of the same shape
                g_reshaped[i, j] = f(g_reshaped[i, j])
        return g_reshaped.view(-1)
    else:
        rest = g.size(1)
        g_reshaped = g.view(3, num_heads, d, rest)
        for i in range(3):
            for j in range(num_heads):
                # f expects a tensor of shape (d, rest)
                g_reshaped[i, j] = f(g_reshaped[i, j])
        return g_reshaped.view(3 * num_heads * d, rest)


def newton_schulz(A, num_iters=6, eps=1e-8):
    """
    A simple Newton–Schulz iteration for (approximately) computing the inverse square root of A.
    (Assumes A is a square matrix.)
    """
    normA = A.norm()
    A_norm = A / (normA + eps)
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    X = A_norm
    for i in range(num_iters):
        X = 0.5 * X @ (3 * I - X @ X)
    return X / (torch.sqrt(normA + eps))


def scale_by_grad_squared(grad, state, beta, eps):
    """
    Implements a grad-squared preconditioning:
        state['grad_squared'] = beta * state['grad_squared'] + (1-beta) * grad^2
        and returns grad divided by (sqrt(state) + eps).
    """
    if "grad_squared" not in state:
        state["grad_squared"] = grad.pow(2)
    else:
        state["grad_squared"].mul_(beta).add_(grad.pow(2) * (1 - beta))
    return grad / (state["grad_squared"].sqrt() + eps)


def scale_by_offset(grad, state, beta):
    """
    Implements an offset update:
        state['offset'] = beta * state['offset'] + (1-beta) * grad
        and returns grad + offset.
    """
    if "offset" not in state:
        state["offset"] = grad.clone()
    else:
        state["offset"].mul_(beta).add_(grad * (1 - beta))
    return grad + state["offset"]


# --- The Mango optimizer --- #

class Mango(Optimizer):
    def __init__(self, params, *,
                 # lrs may be a float (global lr) or (if you use parameter groups) set per group
                 lrs=0.05,
                 schedule=None,
                 momentum=0.95,
                 nesterov=True,
                 ns_steps=6,
                 eps=1e-8,
                 beta2=None,
                 offset_beta=None,
                 normalizations=None,
                 num_heads=12,
                 schedule_wrapper=None):
        """
        Args:
            params: iterable of parameters or parameter groups.
            lrs: float if global learning rate; if using parameter groups, each group should include 'lr'
            schedule: a function f(step) returning a multiplicative factor.
            momentum: momentum coefficient.
            nesterov: whether to use Nesterov momentum.
            ns_steps: number of iterations for Newton–Schulz (if using ns normalization).
            eps: small constant for numerical stability.
            beta2: if provided, used for grad-squared preconditioning.
            offset_beta: if provided, used for offset update.
            normalizations: dict mapping Mango labels (e.g. "mat", "embedding", etc.) to
                           a normalization type string. Defaults to your provided mapping.
            num_heads: used for “split” normalizations.
            schedule_wrapper: optional wrapper for the schedule.
        """
        defaults = dict(lr=lrs,
                        momentum=momentum,
                        nesterov=nesterov,
                        ns_steps=ns_steps,
                        eps=eps,
                        beta2=beta2,
                        offset_beta=offset_beta)
        super(Mango, self).__init__(params, defaults)
        self.schedule = schedule          # e.g. lambda step: <multiplier>
        self.schedule_wrapper = schedule_wrapper
        self.normalizations = normalizations if normalizations is not None else {
            "mat": "ns",
            "embedding": "l2_col",
            "head": "ns",
            "attn_w": "ns_split",
            "attn_b": "l2_split",
            "vec_w": "inf_",
            "vec_b": "l2",
        }
        self.num_heads = num_heads
        self.global_step = 0

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        # Determine learning rate multiplier from schedule (if provided)
        if self.schedule is not None:
            lr_multiplier = self.schedule(self.global_step)
            if self.schedule_wrapper is not None:
                lr_multiplier = self.schedule_wrapper(lr_multiplier)
        else:
            lr_multiplier = 1.0

        for group in self.param_groups:
            # Each group may specify its own base lr; if not, use default
            base_lr = group.get("lr", 0.05)
            # Determine the group’s label (this is used to pick the normalization)
            # We expect the group to have been pre-assigned a 'mango_label' (e.g. via group_parameters_by_mango_label)
            group_label = group.get("mango_label", "mat")
            norm_type = self.normalizations.get(group_label, None)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # --- 1. Grad-squared preconditioning ---
                beta2 = group.get("beta2", None)
                if beta2 is not None:
                    grad = scale_by_grad_squared(grad, state, beta2, group["eps"])

                # --- 2. Momentum update ---
                momentum = group["momentum"]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = grad.clone()
                else:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                if group["nesterov"]:
                    grad = grad + momentum * buf
                else:
                    grad = buf

                # --- 3. Normalization transform ---
                if norm_type is not None:
                    if norm_type == "l2":
                        norm = grad.norm() + group["eps"]
                        grad = grad / norm
                    elif norm_type == "l2_col":
                        if grad.dim() == 2:
                            norm = grad.norm(dim=1, keepdim=True) + group["eps"]
                            grad = grad / norm
                    elif norm_type == "l2_split":
                        grad = split_vmap(grad, self.num_heads,
                                          lambda x: x / (x.norm() + group["eps"]))
                    elif norm_type == "inf_":
                        norm = grad.abs().max() + group["eps"]
                        grad = grad / norm
                    elif norm_type == "inf_col":
                        if grad.dim() == 2:
                            norm = grad.abs().max(dim=1, keepdim=True)[0] + group["eps"]
                            grad = grad / norm
                    elif norm_type == "inf_split":
                        grad = split_vmap(grad, self.num_heads,
                                          lambda x: x / (x.abs().max() + group["eps"]))
                    elif norm_type == "ns":
                        # Only apply if grad is a square matrix.
                        if grad.dim() == 2 and grad.size(0) == grad.size(1):
                            grad = newton_schulz(grad, num_iters=group["ns_steps"], eps=group["eps"])
                    elif norm_type == "ns_split":
                        def ns_func(x):
                            if x.dim() == 2 and x.size(0) == x.size(1):
                                x = newton_schulz(x, num_iters=group["ns_steps"], eps=group["eps"])
                                scale_factor = math.sqrt(max(1, x.size(0) / x.size(1)))
                                return x * scale_factor
                            else:
                                return x
                        grad = split_vmap(grad, self.num_heads, ns_func)
                    else:
                        raise ValueError(f"invalid normalization type = '{norm_type}'")

                # --- 4. Learning-rate scaling ---
                effective_lr = base_lr * lr_multiplier
                grad.mul_(effective_lr)

                # --- 5. Offset update ---
                offset_beta = group.get("offset_beta", None)
                if offset_beta is not None:
                    grad = scale_by_offset(grad, state, offset_beta)

                # --- Parameter update (gradient descent) ---
                p.data.add_(-grad)

        self.global_step += 1
        return loss
