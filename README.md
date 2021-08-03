# BAdam
Modification of Adam [1] optimizer with increased stability and better performance. Tricks used:

1. Decoupled weight decay as in AdamW [2]. Such decoupling allows easier tuning of weight decay and learning rate. This implementation follows PyTorch and multiplies weight decay by learning rate, allowing simultenious sheduling. Due to this typical values passed to optimizer should be much higher that with SGD, if you use `1e-4` with SGD, good start would be to use `1e-2` with BAdam.
2. Epsilon is inside sqrt to avoid NaN in mixed precision. Default value is much larger than in Adam to reduce 'adaptivity' it leads to better and wider optimums [3]. Large epslion also works better than `amsgrad` version of Adam [5]
3. `exp_avg_sq` inits with large value, rather than with zeros. This removes the need for lr warmup and does the same thing as all the tricks from Radam [4], while being much simplier. 
4. Removed bias correction. It's not needed if `exp_avg_sq` is correcly initialized

# Practical Tips
Default values for this optmizer were tuned on Imagenet and work as good baseline for other computer vision tasks. Try them as is, before further tuning. 

# Installation
`pip install git+https://github.com/bonlime/badam.git@master`

### Reference:  
[1] Adam: A Method for Stochastic Optimization  
[2] Decoupled Weight Decay Regularization  
[3] On the Convergence of Adam and Beyond  
[4] On the Variance of the Adaptive Learning Rate and Beyond
[5] Adaptive Methods for Nonconvex Optimization