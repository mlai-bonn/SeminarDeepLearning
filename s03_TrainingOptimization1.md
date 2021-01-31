# Training Optimization 1
In general, an optimization problem is the problem of finding an optimal value $x$ of a function $f(x)$ by maximizing or minimizing this function $f$. In the context of neural network training, optimization is the process 
of minimization of the loss function and accordingly, updating the parameters of the model such that the output accuracy of the neural network is maximized.    
Training neural networks is the most difficult optimization involved in deep learning and it differs from pure optimization in several ways.
 The cost function is usually non-convex which causes several problems and makes carefully choosing the initial points important. 
To conquer the problems several special optimization algorithms have been developed. The algorithms that are covered here are first order methods.

## 8.1 How Learning Differs from Pure Optimization

In machine learning, we usually want to optimize a performance measure P with respect to the test set. As P can only be optimized indirectly (in contrast to pure optimization, where we directly optimize a term of interest) and we do not know the underlying probability distribution of the data, the problem we need to solve is minimizing the empirical risk $E_{(x,y)\sim \hat{p}_data}[L(f(x;θ),y)] = \frac{1}{m} \sum_{i=1}^{m}L(f(x;θ),y),$ where $\hat{p}_data$ is the empirical distribution, L the loss function, f the predicted output for input x and y the actual output. Here, we will only look at the unregularized supervised case. <br />
Empirical risk minimization is rarely used in deep learning, because the loss functions do not have useful derivatives in many cases and it is likely that overfitting occurs.

Instead of the actual loss function we often minimize a surrogate loss function, which acts as a proxy and has more suitable properties for optimization. Minimizing the surrogate loss function halts when early stopping criterion is met. In particular, this means that training often halts when surrogate loss function still has large derivatives. Which is an other difference to pure optimization where we require the gradient to be zero at convergence. The early stopping criterion is based on true underlying loss function measured on the validation set.

An other difference to pure optimization is that in machine learning the objective function usually decomposes as a sum over training examples. We compute each update to the parameters based on an expected value of the cost function only on a small subset of the terms of the full cost function as computing the expectation on the whole dataset is very expensive.

In deep learning, we the optimization algorithms we use are usually so called minibatch or stochastic algorithms. That means we train our model on a batch of examples that is greater than one but also smaller than the whole training set. <br />
When we pick the minibatches, we have to consider the following points: The minibatches have to be selected randomly and subsequent minibatches should be independent of each other in order to get unbiased estimates that are independent of each other. Also, we have to shuffle examples if ordering is significant. In the special case of very large datasets the minibatches are constructed from shuffled examples rather than selected randomly. <br />
Factors influencing the size are: How accurate we want the estimate to be (larger batches yield more accurate estimates), the trade-off between regularization and  optimization, hardware and memory limitations and that multicore architectures are underutilized by very small batches, so it might make sense to define a minimum batch size.

## 8.2 Challenges in Neural Network Optimization
Several challenges arise when optimizing neural networks. Some of them are related to critical points on the surface of the underlying loss function, others depend on the architecture of the deep neural network.
 Despite the fact that these challenges complicate the optimization process in neural networks training, there exists techniques to overcome these limitations.
 In this section, some challenges facing the optimization process are presented as well as their mitigation techniques. <br />

### Ill-Conditioning <br />
Optimizing the training process of neural networks consists of finding the global minimum of the loss function and altering the parameters of the model based on this finding. In general, loss functions of deep neural networks show nonconvex surfaces consisting of a single global minimum and a huge number of local minima. The optimization process on such complex surfaces starts from a point and descends all along the surface of the cost function until the global minimum or a satisfying local minimum is reached. Formally, let $L(f(x;\theta), y)$ be the loss at point $x$, where $f(x;\theta)$ is the prediction delivered by the neural network, $\theta$ the set of parameters (e.g. weights) of the deep model and $y$ is the true label of the input $x$. The goal is to reach a point $x^{*}$ such that $L(f(x^{*};\theta), y)$ is minimal. Starting from a arbitrary point $x$ on the cost function $L$, the first order partial derivative with respect to $x$ $\frac{\partial L}{\partial x}$ is calculated to determine the slope of the loss function at point $x$. According to this slope, the gradient descent optimization method is applied iteratively on the cost function until it reaches a minimum at $\frac{\partial L(f(x;\theta),y)}{\partial x} = 0$, then $x = x^{*}$ is called a "critical point". In this case, the critical point can be either a local or a global minimum. Similarly, the second order partial derivative $\frac{\partial}{\partial x_{i} \partial x_{j}}L$ of the loss function $L$ at point $x$ can be calculated to measure the curvature of the function at that point. It tells how the first derivative (the slope) will vary when the input is changed when moving along the cost function. In addition, the second
derivative can be used to detect whether a critical point ($\frac{\partial L}{\partial x} = 0$) on the surface of the cost function is a local maximum, a local minimum or a saddle point (chapter 4). <br />
Generally, in case the input is high dimensional, there exists many first and second order derivatives that can be packed into matrices. Let $f$ be a real vector-valued function $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ consisting of $m$ functions $f_{1},\dots,f_{m}: \mathbb{R}^{n} \rightarrow \mathbb{R}$, then
the Jacobian matrix is defined as follows: <br />
$J \in \mathbb{R}^{m\times n}$, $(J)_{i,j} := \frac{\partial f_{i}}{\partial x_{j}}$ 
    $$
        J = 
        \left[
          \begin{array}{cccc}
            \vrule &      & \vrule \\
            \nabla_{x}f_{1}    & \ldots & \nabla_{x}f_{m}    \\
            \vrule &     & \vrule 
          \end{array}
        \right].
    $$ 
The first order optimization methods use the Jacobian matrix, i.e. the gradient of $f$ ($\nabla f$), to optimize the parameters of the neural models, whereas the second order optimization methods, e.g. Newton method, make use of
the Hessian matrix $H$ defined as follows: <br />
    
    $$H \in \mathbb{R}^{n\times n}, H(f)(x)_{i,j} := \frac{\partial}{\partial x_{i} \partial x_{j}}f(x).$$
The Hessian matrix encompasses many second order derivatives that hint at the possible directions $d$ which can be taken by the gradient at point $x$ to move along the cost function. Each second order derivative in direction $d$ is represented by $d^{T}H(f)(x)d$. The condition number of a Hessian $H$ is denoted by $\mathcal{K}(H) = |\frac{\lambda_{max}}{\lambda_{min}}|$, 
where $\lambda_{max}$ and $\lambda_{min}$ are respectively the largest and the smallest eigenvalues of $H$. $\mathcal{K}(H)$ at a point $x$ measures the change of the second derivatives in all directions. 
The Hessian matrix $H$ is called ill-conditioned if it has a poor condition number, e.g. a very high $\mathcal{K}(H)$. In this case, gradient descent looses its ability to determine the direction of descent in order to achieve the minimum faster and correctly. In addition, a poorly-conditioned Hessian leads to problems when the gradient-based optimization method wants to choose a suitable step size to move to the next point. More precisely, an ill-conditioned Hessian gives rise to strong curvatures on the cost function surface and gradient descent adapts to these changes by taking smaller steps towards the minimum. If the minimum is far from the gradient, small step sizes will slow down the learning process. <br />
To solve the challenges caused by an ill-conditioned Hessian matrices, Newton method has been proposed. Newton method (introduced earlier in chapter 4) is computed using the inverse of the hessian matrix. Accordingly, if the Hessian is strongly ill-conditioned, its inverse is also affected by the poor conditioning. This fact pushes towards calculating an approximation of the inverse of Hessian to solve the problem instead of considering the actual Hessian inverse. Nevertheless, Newton method is not widely applied in the context of neural networks because it is computationally expensive and it gets easily attracted to saddle points which may stop the optimization process. A second mitigation technique to overcome the problems of an ill-conditioned matrix is to adapt the "Momentum Algorithm" by ... dazu, ganz kurz was sagen(that will be explained in details in section 8.3). 

### Local Minima
A function is convex if it owns one single minimum that can be regarded as local and global minimum at the same time. However, the cost function of a neural network is nonconvex. 
Its surface presents a single global minimum but local minima are prolifrated, especially in high dimensional spaces(check this info one more time). Despite the huge number of local minima 
that can grow till infinity, many of them are equivalent to each others in cost value due to a property of neural networks, namely, the nonidentifiability. The nonidentifiability of deep 
networks can be proven in many ways. First, there are n! possibilities to select suitable weights during training and still get the same result (weight space symmetrie). Second, <idea 2>. 
Nonidentifiability of neural network allows the proliferation of local minima without affecting the optimization process. 
Although, it becomes challenging if there is a big number of local minima having cost values that differ considerably from the cost value of the global minimum. 
In this case, the learning can not be generalized accurately. <br /> 
The fact that most local minima present low loss value that do not deviate significantly from the global minimum loss value eliminates the generalization issues that may arise
 from the proliferation of such kinds of local minima. In addition, for an optimization method, it is sufficient to find a convenient local minimum that generalizes well on the data
 rather than finding the global minimum to update the model's parameters. 

### Plateaus, Saddle Points and other Flat Regions
### Cliffs
### Long-Term Dependencies
### Poor Correspondance between Local and Global Structure

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
