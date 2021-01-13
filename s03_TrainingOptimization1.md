# Training Optimization 1

Training neural networks is the most difficult optimization involved in deep learning and it differs from pure optimization in several ways. The cost function is usually non-convex which causes several problems and makes carefully choosing the initial points important. To conquer the problems several special optimization algorithms have been developed. The algorithms that are covered here are first order methods.

## 8.1 How Learning Differs from Pure Optimization

In machine learning, we usually want to optimize a performance measure P with respect to the test set. As P can only be optimized indirectly (in contrast to pure optimization, where we directly optimize a term of interest) and we do not know the underlying probability distribution of the data, the problem we need to solve is minimizing the empirical risk $E_{(x,y)\sim \hat{p}_data}[L(f(x;θ),y)] = \frac{1}{m} \sum_{i=1}^{m}L(f(x;θ),y), \textrm{ where } \hat{p}_data$ is the empirical distribution, L the loss function, f the predicted output for input x and y the actual output. Here, we will only look at the unregularized supervised case. <br />
Empirical risk minimization is rarely used in deep learning, because the loss functions do not have useful derivatives in many cases and it is likely that overfitting occurs.

Instead of the actual loss function we often minimize a surrogate loss function, which acts as a proxy and has more suitable properties for optimization. Minimizing the surrogate loss function halts when early stopping criterion is met. In particular, this means that training often halts when surrogate loss function still has large derivatives. Which is an other difference to pure optimization where we require the gradient to be zero at convergence. The early stopping criterion is based on true underlying loss function measured on the validation set.

An other difference to pure optimization is that in machine learning the objective function usually decomposes as a sum over training examples. We compute each update to the parameters based on an expected value of the cost function only on a small subset of the terms of the full cost function as computing the expectation on the whole dataset is very expensive.

In deep learning, we the optimization algorithms we use are usually so called minibatch or stochastic algorithms. That means we train our model on a batch of examples that is greater than one but also smaller than the whole training set. <br />
When we pick the minibatches, we have to consider the following points: The minibatches have to be selected randomly and subsequent minibatches should be independent of each other in order to get unbiased estimates that are independent of each other. Also, we have to shuffle examples if ordering is significant. In the special case of very large datasets the minibatches are constructed from shuffled examples rather than selected randomly. <br />
Factors influencing the size are: How accurate we want the estimate to be (larger batches yield more accurate estimates), the trade-off between regularization and  optimization, hardware and memory limitations and that multicore architectures are underutilized by very small batches, so it might make sense to define a minimum batch size.

## 8.2

## 8.3


## 8.4 Parameter Initialization Strategies

Training algorithms for deep learning are usually iterative. That means the user has to specify an initial point. Initial point affects the convergence, the speed of convergence and if we converge to a point with high or low cost. This last aspect is important as points of comparable cost can have different generalization error and the goal of training is to minimize the generalization error. 

Most initialization strategies are based on achieving good properties when the network isinitialized. There is no good understanding of how these properties are preserved during training. Certainly known is only that the initial parameters need to break symmetry between different units, which means that hidden units with same activation function and connection to same input parameters must have different initial parameters. This motivates to random initialization. <br /> 
More specifically, the weights are initialized randomly. The values are drawn either from a Gaussian or uniform distribution. The scale of the initial distribution has a large effect on the outcome, it influences optimization and generalization. Larger weights lead to stronger symmetry-breaking effect, but too large weights can cause exploding values during forward or backward-propagation or saturation of the activation function. Some heuristic initialization methods that are used in practice are: <br />

1. Sample each weight from $U(-\frac{1}{\sqrt{m}}, \frac{1}{\sqrt{m}})$, where m is the number of input layers. <br />
2. Normalized initialization: $W_{i, j} \sim U(-\sqrt{\frac{6}{m+n}}, \sqrt{\frac{6}{m+n}})$ <br />
3. Initialize to random orthogonal matrices with gain factor g that needs to be carefully chosen. <br />
4. Use sparse initialization: each unit is initialized to have exactly k nonzero weights. <br /> 
The second approach compromises between having the same activation variance and same gradient variance among all layers.<br /> 
An advantage of sparse initialization over approach 1. and 2. is that it does not scale with the number of inputs or outputs. An disadvantage is that it imposed a large prior on weights with large values.

Optimal criteria for initial weights do not lead to optimal performance. That is why in practice the it is useful to treat initial weights as hyperparameters and to treat the initial scale of the weights and whether to use sparse or dense initialization as hyperparameter aswell if not too costly.

The approach for setting the biases must be coordinated with the approach for setting the weights. Setting the biases to zero is compatible with most weight initialization schemes. E.g. it is important to set the biases to nonzero weights, if a unit controls whether other units are able to participate in a function or too avoid to much saturation at initialization time.

It is also possible to initialize model parameters using machine learning. This approach is not covered here.


 
## Questions
