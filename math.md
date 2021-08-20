File for thoughts about new experiments

```
# Tensorflow order of ops for updating squared avg
square_avg.add_(one_minus_alpha, grad.pow(2) - square_avg)
sq += (1 - alpha) * (grad ^ 2 - sq)

sq = sq * alpha + (1 - alpha) * grad ^ 2
sq *= alpha
sq += (1-alpha) * grad * grad
# square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)  # PyTorch original
```