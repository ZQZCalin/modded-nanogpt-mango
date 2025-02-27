import torch
from torch.optim.optimizer import Optimizer, required
from functools import partial

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7, scale_dim=True):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    if scale_dim:
        X *= max(1, X.size(0)/X.size(1))**0.5
    return X

@torch.compile
def normalize_via_newtonschulzfunc(G):
    """Under construction..."""

def normalize_via_colmax(G, transpose=False):
    assert G.dim() == 2
    if transpose:
        axis = 1    # broadcast along rows
    else:
        axis = 0    # broadcast along cols
    max_norm = G.norm(dim=axis).max()
    return G / max_norm

def normalize_via_sign(G):
    return G.sign_()

normalize_backends = dict(
    svd=zeropower_via_svd,
    newtonschulz5=zeropower_via_newtonschulz5,
    colmax=normalize_via_colmax,
    sign=normalize_via_sign,
)

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

class Mango(Optimizer):
    def __init__(self, params, lr=required, beta1=0.95, beta2=0.95, nesterov=True,
                 normalize_fn=None, scale_rms=True, eps=1e-8, laprop=False,
                 precond_power=0.5, postcond_power=0.0, **backend_args):
        """
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
        """
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, nesterov=nesterov,
                        normalize_fn=normalize_fn, scale_rms=scale_rms, eps=eps,
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
            if group['normalize_fn']:
                normalize_fn = partial(normalize_backends[group['normalize_fn']],
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
                    update = momentum
                
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
