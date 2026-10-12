"""Custom AdamW preserving the notebook's post-update decay convention."""

import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.0,
        correct_bias=True,
    ):
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError(
                "learning rate, epsilon and decay must be nonnegative"
            )
        if any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("betas must be in [0, 1)")
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                correct_bias=correct_bias,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )
                state = self.state[parameter]
                if not state:
                    state.update(
                        step=0,
                        exp_avg=torch.zeros_like(parameter),
                        exp_avg_sq=torch.zeros_like(parameter),
                    )
                state["step"] += 1
                average, square = state["exp_avg"], state["exp_avg_sq"]
                average.mul_(beta1).add_(grad, alpha=1 - beta1)
                square.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denominator = square.sqrt()
                step_size = group["lr"]
                if group["correct_bias"]:
                    denominator.div_(math.sqrt(1 - beta2 ** state["step"]))
                    step_size /= 1 - beta1 ** state["step"]
                parameter.addcdiv_(
                    average, denominator.add_(group["eps"]), value=-step_size
                )
                # retain the source update order for the stored reference test.
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
        return loss
