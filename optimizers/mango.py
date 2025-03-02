# -----------------------------------------------------------------------------
# Mango optimizer

import torch
from torch import Tensor
import torch.distributed as dist
from functools import partial

def tensor_pow(t, power):
    """A custom wrapper for `torch.pow`."""
    if power == 0:
        return t
    elif power == 0.5:
        return t.sqrt()
    elif power == 0.25:
        return t.sqrt().sqrt()
    else:
        return t.pow(power)

def rms(t):
    if t.numel() == 0:
        return torch.tensor(0.0, device=t.device, dtype=t.dtype)
    return torch.sqrt(torch.mean(t**2))

########################################
#         Normalization Methods        #
########################################

def zeropower_via_svd(G):
    """Simple SVD implementation."""
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int, scale_dim: bool = True) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    
    # We move dimension scale here
    if scale_dim:
        X = X * max(1, G.size(-2)/G.size(-1))**0.5
    return X

@torch.compile
def normalize_via_polynomial(G: Tensor, ):
    """Under construction..."""
    raise NotImplementedError

def normalize_via_colmax(G, transpose=False):
    """Normalize by column-max l2 norm."""
    assert G.dim() == 2
    if transpose:
        axis = 1    # broadcast along rows
    else:
        axis = 0    # broadcast along cols
    max_norm = G.norm(dim=axis).max()
    return G / max_norm

def normalize_via_sign(G):
    """Normalize by sign function."""
    return G.sign_()

normalize_backends = dict(
    svd=zeropower_via_svd,
    newtonschulz5=zeropower_via_newtonschulz5,
    polynomial=normalize_via_polynomial,
    colmax=normalize_via_colmax,
    sign=normalize_via_sign,
)

########################################
#         Main Mango Optimizer         #
########################################

class Mango(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, beta1=0.95, beta2=0.95, nesterov=True,
                 backend="newtonschulz5", scale_rms=True, eps=1e-8, laprop=False,
                 precond_power=0.5, postcond_power=0.0, **backend_args):
        """
        Mango optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            beta1: Coefficient for momentum.
            beta2: Coefficient for squared-gradient accumulator.
            nesterov: Whether to use Nesterov momentum.
            normalize_fn: Optional callable to normalize updates.
            scale_rms: Whether to scale updates by their RMS.
            eps: Small constant for numerical stability.
            laprop: If True, use LaProp-style preconditioning.
            precond_power: Exponent for preconditioning (e.g. 0.5 for square-root).
            postcond_power: Exponent for post-conditioning.

        Note that Mango reduces to Muon with 
        __init__(self, params, lr=0.02, beta1=0.95, beta2=0.0, nesterov=True,
                 backend="newtonschulz5", steps=5, scale_dim=True,
                 scale_rms=False, eps=1e-8, laprop=False,
                 precond_power=0.0, postcond_power=0.0)
        """
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, nesterov=nesterov,
                        backend=backend, scale_rms=scale_rms, eps=eps,
                        laprop=laprop, precond_power=precond_power, postcond_power=postcond_power,
                        backend_args=backend_args)
        super(Mango, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            nesterov = group['nesterov']
            scale_rms = group['scale_rms']
            eps = group['eps']
            laprop = group['laprop']
            precond_power = group['precond_power']
            postcond_power = group['postcond_power']
            backend = group['backend']
            if backend:
                assert backend in normalize_backends
                normalize_fn = partial(normalize_backends[backend], 
                                       **group['backend_args'])
            else:
                normalize_fn = None
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                
                state = self.state[p]
                # State initialization
                if not state:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if beta2:
                        state['grad_squared'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else:
                        state['grad_squared'] = None
                        
                state['step'] += 1
                
                # 1. Update the grad_squared preconditioner if beta2 is used.
                if beta2:
                    state['grad_squared'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 2. Optionally apply LaProp-style preconditioning.
                if beta2 and precond_power and laprop:
                    update = grad / (tensor_pow(state['grad_squared'], precond_power) + eps)
                else:
                    update = grad.clone()
                
                # 3. Update momentum: new_momentum = beta1 * old_momentum + update.
                momentum = state['momentum']
                momentum.mul_(beta1).add_(update)
                state['momentum'] = momentum  # (state update is in-place)
                
                # Use Nesterov lookahead if specified.
                if nesterov:
                    update = beta1 * momentum + update
                else:
                    update = momentum.clone()
                
                # 4. If not using LaProp, apply Adam-style preconditioning.
                if beta2 and precond_power and (not laprop):
                    update.mul_(1 / (tensor_pow(state['grad_squared'], precond_power) + eps))
                
                # 5. Optionally apply a normalization function.
                if normalize_fn is not None:
                    update = normalize_fn(update)
                    
                # 6. Apply post-conditioning if postcond_power is set.
                if beta2 and postcond_power:
                    update.mul_(1 / (tensor_pow(state['grad_squared'], postcond_power) + eps))
                
                # 7. Optionally apply RMS normalization.
                if scale_rms:
                    update.mul_(1 / (rms(update) + eps))
                
                # 8. Update the parameter.
                p.data.add_(update, alpha=-lr)
                
        return loss
