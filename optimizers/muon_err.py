# -----------------------------------------------------------------------------
# Muon optimizer with error feedback mechanism

import torch
from torch import Tensor
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
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
    return X

def newtonschulz_clipping(G: Tensor, tau: float) -> Tensor:
    """Newton-Schulz clipping that maps lambda -> max(lambda - tau, 0)."""
    raise NotImplementedError
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def error_feedback(G: Tensor, tau: float) -> Tensor:
    """Error feedback mechanism."""
    # For now, we use SVD instead of the faster newton-schulz implementation.
    assert G.ndim >= 2
    X = G.clone()
    if G.size(-2) > G.size(-1):
        X = X.mT
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    # S = S / (S.max(dim=-1, keepdim=True).values + 1e-7)
    # S = torch.clip(S - tau, min = 0.0)
    spectral_norm = S.max(dim=-1, keepdim=True).values + 1e-7
    S = torch.clip(S/spectral_norm-tau, min=0.0) * spectral_norm 
    X = U @ torch.diag_embed(S) @ Vh
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class MuonErr(torch.optim.Optimizer):
    """
    Muon with error feedback mechanism.

    Args:
        error_feedback: 1.0 recovers no momentum; 0.0 means no exponential downscaling on momentum term.
    """
    def __init__(self, params, lr=0.02, error_feedback=0.05, decay=True, momentum=0.95,
                 nesterov=True, nesterov_momentum=0.95, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, error_feedback=error_feedback, decay=decay, momentum=momentum, 
                        nesterov=nesterov, nesterov_momentum=nesterov_momentum, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    # replace momentum update with error-feedback update
                    # >> V1 updates:
                    # buf: Tensor = state["momentum_buffer"]
                    # buf = error_feedback(buf, tau=group["error_feedback"]).add(g)
                    # state["momentum_buffer"] = buf
                    # g = g.add_(error_feedback(buf, tau=group["error_feedback"])) if group["nesterov"] else buf
                    # >> V4 updates:
                    # incorporate momentum decay, disable error feedback for nesterov update
                    buf: Tensor = state["momentum_buffer"]
                    buf = error_feedback(buf, tau=group["error_feedback"])
                    buf = buf.lerp_(g, 1 - group["momentum"]) if group["decay"] else buf.add_(g)
                    state["momentum_buffer"] = buf
                    g = g.lerp_(buf, group["nesterov_momentum"]) if group["nesterov"] else buf
                    # >> Muon updates:
                    # buf.lerp_(g, 1 - group["momentum"])
                    # g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()