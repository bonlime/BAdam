import torch
from loguru import logger
from torch.optim import Optimizer
from collections import defaultdict


def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type).expand_as(x)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True).expand_as(x)


class BAdam(Optimizer):
    r"""BAdam - Better Adam is an optimizer based on Adam with couple important modifications

    1. decoupled weight decay (as in AdamW [2])
    2. epsilon is inside sqrt to avoid NaN in mixed precision
        default value is much larger than in Adam to reduce 'adaptivity' it leads to better and wider optimums [3]
        large epsilon works better than `amsgrad` version of Adam
    3. `exp_avg_sq` inits with large value, rather than with zeros. this removes the need for lr warmup and does the same
        thing as all the tricks from RAdam [4].
    4. Removed bias correction. It's not needed if exp_avg_sq is correctly initialized

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        avg_sq_init (float, optional): value to use for average square initialization
            smaller values lead to faster convergence at the begining but too small vaules degrade performance
            default should be good enough, no need to tune
        nesterov (bool): Flag  to use Nesterov accelerated momentum
        wd_type (str): one of `default`, `stable`. [5]

    Ref:
        [1] Adam: A Method for Stochastic Optimization
        [2] Decoupled Weight Decay Regularization
        [3] On the Convergence of Adam and Beyond
        [4] On the Variance of the Adaptive Learning Rate and Beyond
        [5] STABLE WEIGHT DECAY REGULARIZATION
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=1e-2,
        avg_sq_init=1e-3,
        nesterov=False,
        wd_type="default",
        lamb=False,  # normalize gradient to unit norm and scale by weight
        lamb_unitwise=False,
        center=False, # gradient centralization
        projection=False,  # projection on tangent space as in AdamP paper. only applies to convolutional weights
        adaptive_eps=False,
        # projection_wd_ratio=0.1,  # how much to discount weight decay for projected parameters. by order of magnitude is a good choice usually
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, avg_sq_init=avg_sq_init)
        super().__init__(params, defaults)
        self.nesterov = nesterov
        self.wd_type = wd_type
        self.lamb = lamb
        self.lamb_unitwise = lamb_unitwise
        # in PyTorch AdamW eps is inside state, but in this implementation it's global per optimizer
        self.eps = eps # eps
        self.lamb_eps = 1e-3
        self.center = center
        self.projection = projection
        self.adaptive_eps = adaptive_eps
        # self.adaptive_eps_every = 100
        # not supported now because i'm applying wd to all params at once
        # self.projection_wd_ratio = projection_wd_ratio

    @staticmethod
    def _channel_view(x):
        return x.view(x.size(0), -1)

    def _project(self, p, perturb):
        # apply only to 2d convolutions. second check to filter `gain` with shape `size, 1, 1, 1`
        if p.ndim < 4 or p.numel() == p.size(0):
            return perturb
        expand_size = [-1, 1, 1, 1]
        # projection onto the tangent space of W
        p_n = p.data / self._channel_view(p.data).norm(dim=1).view(expand_size).add_(1e-5) # eps
        perturb -= p_n * self._channel_view(p_n * perturb).sum(dim=1).view(expand_size)
        return perturb
    
    def update_epsilon(self) -> None:
        SUBSAMPLE = 100
        ADAPTIVE_EPS_EVERY = 100
        # don't want to update every step to avoid slowing down training
        if next(iter(self.state.values()))['step'] % ADAPTIVE_EPS_EVERY != 0:
            return
        prev_eps = self.eps + 1  # + 1 to make copy of tensor
        # subsampling model params gives the same estimate for median but requires fraction of memory 
        self.eps = torch.cat([v['exp_avg_sq'].flatten()[::SUBSAMPLE] for v in self.state.values()]).median()
        logger.info(f"Updating epsilon. Before: {prev_eps - 1:.2e}. After: {self.eps:.2e}")

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grads = []
            states = []
            exp_avg = []
            exp_avg_sq = []
            params_with_grad = []

            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError("BAdam does not support sparse gradients")

                    params_with_grad.append(p)
                    if self.center and p.ndim > 1:
                        p.grad.sub_(p.grad.mean(dim=tuple(range(1, p.ndim)), keepdim=True))
                    grads.append(p.grad)

            for p in params_with_grad:
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values. Torch init to zeros here
                    state["exp_avg_sq"] = torch.full_like(p, group["avg_sq_init"], memory_format=torch.preserve_format)

                exp_avg.append(state["exp_avg"])
                exp_avg_sq.append(state["exp_avg_sq"])

                state["step"] += 1
                states.append(state)

            beta1, beta2 = group["betas"]

            #
            # Decay the first and second moment running average coefficient
            #
            torch._foreach_mul_(exp_avg, beta1)
            torch._foreach_add_(exp_avg, grads, alpha=1 - beta1)

            torch._foreach_mul_(exp_avg_sq, beta2)
            torch._foreach_addcmul_(exp_avg_sq, grads, grads, 1 - beta2)

            exp_avg_sq_sqrt = torch._foreach_sqrt(torch._foreach_add(exp_avg_sq, self.eps))

            if self.nesterov:
                exp_avg = torch._foreach_mul(exp_avg, beta1)  # not inplace here!
                torch._foreach_add_(exp_avg, grads, alpha=1 - beta1)

            if self.lamb:
                adam_step = torch._foreach_div(exp_avg, exp_avg_sq_sqrt)
                # clipping min values to avoid zero initialized weights being stuck
                if self.lamb_unitwise:
                    # unitwise as in AGC paper
                    weight_norms = [unitwise_norm(w).clamp_min(self.lamb_eps) for w in params_with_grad]
                    adam_norms = [unitwise_norm(w).clamp_min(self.eps) for w in adam_step]
                    trust_ratio = [w / a * -group["lr"] for w, a in zip(weight_norms, adam_norms)]
                else:
                    # default as in LAMB paper
                    weight_norms = [w.norm(2).clamp_min(self.lamb_eps) for w in params_with_grad]
                    adam_norms = [w.norm(2).clamp_min(self.eps) for w in adam_step]
                    trust_ratio = [(w / a * -group["lr"]).item() for w, a in zip(weight_norms, adam_norms)]
                # p = p - lr * adam_step * || W || / ||adam_step||
                torch._foreach_mul_(adam_step, trust_ratio)
                torch._foreach_add_(params_with_grad, adam_step)
            else:
                if self.projection:
                    exp_avg = [self._project(p, perturb) for p, perturb in zip(params_with_grad, exp_avg)]
                # p = p - lr * exp_avg / exp_avg_sq_sqrt
                torch._foreach_addcdiv_(params_with_grad, exp_avg, exp_avg_sq_sqrt, -group["lr"])

            # Perform stepweight decay
            if self.wd_type == "default":
                # default AdamW weight decay. p *= 1 - lr * wd
                torch._foreach_mul_(params_with_grad, 1 - group["lr"] * group["weight_decay"])
            elif self.wd_type == "stable":
                adaptive_wd = [1 - group["lr"] * group["weight_decay"] / i.mean().item() for i in exp_avg_sq_sqrt]
                # weight decay scaled by effective learning rate. p *= (1 - lr * wd / exp_avg_sq_sqrt)
                torch._foreach_mul_(params_with_grad, adaptive_wd)
            elif self.wd_type == "norm_loss":
                # push weights to have unit norm. this definition is different from original NormLoss paper and follows
                # Ranger21 optimizer paper. p *= 1 - lr * wd * (1 - 1 / ||p_out||)
                weight_norms = [unitwise_norm(w).clamp_min(self.lamb_eps) for w in params_with_grad]  # ||p_out||
                decay_factor = [1 - group["lr"] * group["weight_decay"] * (1 - 1 / w)  for w in weight_norms]
                torch._foreach_mul_(params_with_grad, decay_factor)

            else:
                raise ValueError(f"Weight decay type not known: {self.wd_type}")

        if self.adaptive_eps:
            self.update_epsilon()

        return loss

    # TODO: refactor to a base class once foreach ops are in a good shape.
    def zero_grad(self, set_to_none: bool = False):
        per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)

                        if p.grad.is_sparse:
                            p.grad.zero_()
                        else:
                            per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad)

            for _, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    torch._foreach_zero_(grads)
