# -----------------------------------------------------------------------------
# Adam optimizer and its variants

import torch
from torch import Tensor
import torch.distributed as dist

class AdamW(torch.optim.Optimizer):
    """
    Implements adamw optimizer, modified based on Muon.
    """
    def __init__(self, params, lr=0.02, b1=0.95, b2=0.95, eps=1e-8, wd=0.0, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, b1=b1, b2=b2, eps=eps, wd=wd)
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
                    p_world.add_(g_world.view_as(p_world), alpha=-group["lr"])
                                #  alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    if "grad_squared" not in state:
                        state["grad_squared"] = torch.zeros_like(g)
                    if "count" not in state:
                        state["count"] = 0
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["b1"])
                    precond: Tensor = state["grad_squared"]
                    precond.lerp_(g**2, 1 - group["b2"])
                    count: int = state["count"]
                    count += 1
                    g = (buf/(1-group["b1"])**count) / ((precond/(1-group["b2"]**count)).sqrt() + group["eps"])
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()