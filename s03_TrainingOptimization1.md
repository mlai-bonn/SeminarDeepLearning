# Training Optimization 1
In general, an optimization problem is the problem of finding an optimal value $x$ of a function $f(x)$ by maximizing or minimizing this function $f$. 
In the context of neural network training, optimization is the process 
of minimization of the loss function and accordingly, updating the parameters of the model such that the output accuracy of the neural network is maximized. 
In this chapter, section 8.1 shows how learning differs from pure optimization. Then, challenges facing the training optimization as well as their mitigation techniques are investigated in section 8.2. 
First order optimization algorithms as well as their parameters initialization strategies are presented in chapters 8.3 and 8.4, respectively.


## 8.1 How Learning Differs from Pure Optimization

Optimization algorithms used for training deep models differ from traditional optimization algorithms in several ways. Machine learning methods usually act indirectly. In most scenarios the goal is to optimize some performance measure $P$ that is definied with respect to the test set and is possibly intractable. $P$ is optimized indirectly by reducing a different cost function $J(\theta)$ in the hope that doing so will improve $P$. This is in contrast to pure optimization where the goal is to minimize $J$ itself. In many cases the cost function can be written as an average over the training set
$J(\theta) = E_{(x, y) \sim \hat{p}} L(f(x; \theta), y)$ <br />
where $\hat{p}$ is the empirical distribution, $L$ the loss function, $f$ the predicted output for input $x$ and $y$ the actual output for input $x$. In the above equation $J(\theta)$ is defined with respect to the training set, but we would usually prefer to minimize the corresponding objective function where expectation is taken across the data generating distribution $p$: <br />
$J^*(\theta) = E_{(x, y) \sim p} L(f(x; \theta), y)$.

### Empirical Risk Minimization <br />

The goal of machine learning is to minimize the expected generalization error $J^*(\theta)$, called risk. Since the data generating distribution $p$ is unknown the task cannot be solved by an optimization algorithm. Instead the problem is converted back into an optimization problem by replacing the true distribution by the empirical distribution $E_{(x, y) \sim \hat{p}} L(f(x; \theta), y) = \frac{1}{m} \sum\limits_{i=1}^{m} L(f(x^{(i)}; \theta), y^{(i)})$, <br /> where m is the number of training examples. The training process based on minimizing the average training error is called empirical risk minimization. In the context of deep learning empirical risk minimization is rarely used. The first reason for this is that empirical risk minimization is prone to overfitting. The second reason is that many loss functions do not have useful derivatives, but the most effective modern optimization methods are based on gradient descent which involves the derivative of the loss function. Instead of reducing the empirical risk, we optimize a more different quantity from the one that we actually want to optimize in deep learning.

### Surrogate Loss Functions and Early Stopping <br />

Instead of the actual loss function we often minimize a surrogate loss function, which acts as a proxy for the loss function and has more suitable properties for optimization. Minimizing the surrogate loss function halts when early stopping criterion is met. In particular, this means that training often halts when the surrogate loss function still has large derivatives. This is a second difference to pure optimization where we require the gradient to be zero at convergence. The early stopping criterion is based on true underlying loss function measured on the validation set.

### Batch and Minibatch Agorithms

In machine learning, the objective function usually decomposes as a sum over training examples. We compute each update to the parameters based on an expected value of the cost function only on a small subset of the terms of the full cost function as computing the expectation on the whole dataset is very expensive. In practice, the expectations are computed by randomly sampling a small number of examples and taking the average over only those examples.

In deep learning, the optimization algorithms we use are usually so called minibatch or stochastic algorithms. That means we train our model on a batch of examples that contains more than one but also less than all training data. <br />
When we pick the minibatches, we have to consider the following points: 
 - The minibatches have to be selected randomly and subsequent minibatches should be independent of each other in order to get unbiased estimates that are independent of each other. 
 - Shuffle examples if ordering is significant. 
 - In the special case of very large datasets the minibatches are constructed from shuffled examples rather than selected randomly. 
<br />

Factors influencing the size are: 
 - How accurate we want the estimate to be (larger batches yield more accurate estimates), 
 - Trade-off between regularization and  optimization, 
 - Hardware and memory limitations 
 - Multicore architectures are underutilized by very small batches, so it might make sense to define a minimum batch size. <br />

Different kinds of algorithms use diffent kinds of information from the minibatch in various ways. Some algorithms are more sensitive to sampling error that others, either because they use information that cannot be estimated accurately using few samples or becausethey use information in a way that amplifies errors. Gradient based methods are usually relatively robust and can handle smaller batch sizes, like 100.

One motivation for Stochastic Gradient Descent is that it follows the gradient of the true generalization error, if no examples are repeated. The fact that Stochastic Gradient Descent minimizes the true generalization error can be seen if we consider online learning, i.e. when minibatches are drawn from a stream of data such that every experience is a fair sample from $p$. If we assume discrete $x$ and $y$ the generalization error can be written as
$J^*(\theta)=\sum_x\sum_y p(x, y) L(f(x;\theta), y)$ with exact gradient <br /> 

$\nabla_{\theta}J^*(\theta)=\sum_x\sum_y p(x, y)\nabla_{\theta} L(f(x;\theta), y)$. <br />

Hence $\hat{g}=\frac{1}{m}\nabla_{\theta}\sum_i L(f(x^{(i)};\theta),y^{(i)})$ is an unbiased estimate of $\nabla_{\theta}J^*(\theta)$ if we sample a minibatch of examples $\{x^{(1)},.. ,x^{(m)}}$ with corresponding targets $y^{(i)}$ from $p$ and no example is repeated. Updating $\theta$ in direction of $\hat{g}$ performs SGD on the generalization error.


## 8.2 Challenges in Neural Network Optimization

Several challenges arise when optimizing neural networks. Some of them are related to critical points on the surface of the underlying loss function, others are caused by the architecture of the deep neural network.
 In this section, some challenges and their mitigation techniques are presented. 


### Definitions (Recap) <br />
Let $L(f(x;\theta), y)$ be the loss at point $x$, where $f(x;\theta)$ is the prediction function delivered by the neural network, $\theta$ is the set of parameters (e.g. weights) of the deep model and $y$ is the true label of the input $x \in \mathbb{R}^n$. <br />

The goal is to reach a point $x^{\ast}$ such that $L(f(x^{\ast};\theta), y)$ is minimal. To find $x^{\ast}:= \underset{x\in \mathbb{R}^n}{ \text{argmin } } L(f(x;\theta), y)$, the [gradient-based optimization](https://mlai-bonn.github.io/SeminarDeepLearning/s01_OptimizationMethods.html) ([4.3](https://mlai-bonn.github.io/SeminarDeepLearning/s01_OptimizationMethods.html)) is employed. <br />

Starting from an arbitrary point $x$ on the cost function $L$, the first order partial derivative with respect to $x$, $\frac{\partial L(f(x;\theta), y)}{\partial x}$, is calculated to determine the slope of the loss function at point $x$. According to this slope, the gradient descent optimization method is applied iteratively on the cost function until it reaches a minimum at $\frac{\partial L(f(x;\theta),y)}{\partial x} = 0$, then $x = x^{\ast}$ is called a *critical point*. <br /> 

In general, the critical point at $\frac{\partial L(f(x;\theta),y)}{\partial x} = 0$ can be either a local minimum, a local maximum or a saddle point. To figure out the exact critical point at $x$, the second order partial derivative $\frac{\partial}{\partial x_{i} \partial x_{j}}L$ of the loss function $L$ at point $x$ is calculated.

The second derivative determines the curvature of the loss function $L$ depending on its sign as well as on the first derivatives to the left and to the right of the point $x$. <br />

When the input is high dimensional, there exists several first and second order derivatives for a function $L$ which can be packed into matrices to ease the search for critical points. <br />

Let $g$ be a real vector-valued function $g: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ consisting of $m$ functions $g_{1},\dots,g_{m}: \mathbb{R}^{n} \rightarrow \mathbb{R}$, then

the Jacobian matrix is defined as $J \in \mathbb{R}^{m\times n}$ with $J_{i,j} :=$ $\frac{\partial g_{i}}{\partial x_{j}}.$ 
    
The first order optimization methods use the Jacobian matrix, i.e. [the gradient](https://mlai-bonn.github.io/SeminarDeepLearning/s01_OptimizationMethods.html) $\nabla g$ ([4.3](https://mlai-bonn.github.io/SeminarDeepLearning/s01_OptimizationMethods.html)), to optimize the parameters of the neural models, whereas the second order optimization methods, e.g. [Newton's method](https://mlai-bonn.github.io/SeminarDeepLearning/s01_OptimizationMethods.html) ([4.3](https://mlai-bonn.github.io/SeminarDeepLearning/s01_OptimizationMethods.html)), use the Hessian matrix $H$ defined as
$H \in \mathbb{R}^{n\times n}$ with $H_{i,j} :=$ $\frac{\partial}{\partial x_{i} \partial x_{j}}g(x).$



### Conditioning <br />
The Hessian matrix encompasses many second order derivatives. Each second derivative represents a possible direction that the gradient at point $x$ can take to move along the cost function.
The condition number of a Hessian $H$ at a point $x$ measures the difference between the second derivatives in all possible directions.  


### Ill-Conditioning <br />

The Hessian matrix $H$ is called ill-conditioned if it has a poor condition number. If the condition number is high, then the difference between the second derivatives in different directions is high as well. This leads the gradient descent to loose its ability to determine [the direction of the steepest descent](https://mlai-bonn.github.io/SeminarDeepLearning/s01_OptimizationMethods.html) ([4.3](https://mlai-bonn.github.io/SeminarDeepLearning/s01_OptimizationMethods.html)) to achieve the minimum fast and correctly. In addition, a poorly-conditioned Hessian leads to problems when the gradient-based optimization method wants to choose [a suitable step size](https://mlai-bonn.github.io/SeminarDeepLearning/s01_OptimizationMethods.html) ([4.3](https://mlai-bonn.github.io/SeminarDeepLearning/s01_OptimizationMethods.html)) to move to the next point on the loss function. The reason for this behaviour is that an ill-conditioned Hessian gives rise to strong curvatures on the cost function surface and the gradient descent adapts to these changes by taking smaller steps towards the minimum. If the minimum is far away from the gradient position, small step sizes will slow down the learning process. <br />

**Mitigation Techniques**
To solve the challenges caused by ill-conditioned Hessian matrices, [Newton's method](https://mlai-bonn.github.io/SeminarDeepLearning/s01_OptimizationMethods.html) ([4.3](https://mlai-bonn.github.io/SeminarDeepLearning/s01_OptimizationMethods.html)) is used after modification. 
The modification of the Newton's method introduced in [4.3](https://mlai-bonn.github.io/SeminarDeepLearning/s01_OptimizationMethods.html) is necessary because it computes the inverse of the Hessian matrix to arrive at an optimum. 
If the Hessian is strongly ill-conditioned, then its inverse is ill-conditioned too. Section [8.6](https://github.com/mlai-bonn/SeminarDeepLearning/blob/master/s04_TrainingOptimization2.md) motivates the Newton's method and explains how it can be modified ([Conjugate Gradient](https://github.com/mlai-bonn/SeminarDeepLearning/blob/master/s04_TrainingOptimization2.md)) in details. <br />

A second mitigation technique to overcome the problems of an ill-conditioned matrix is to adapt the [Momentum algorithm](https://mlai-bonn.github.io/SeminarDeepLearning/s03_TrainingOptimization1.html). 
This algorithm allows the gradient to traverse smoothly strong curvatures caused by an ill-conditioned Hessian using the *momentum parameter*. More information on how this algorithm works can be found in section [8.3](https://mlai-bonn.github.io/SeminarDeepLearning/s03_TrainingOptimization1.html). 


### Local Minima

The cost function of a neural network is nonconvex, it presents a single global minimum and a large number of local minima. 
The proliferation of local minima is not problematic due to the *non-identifiability* of neural networks. 
The *non-identifiability* property declares that altering model's parameters by scaling or permutating them returns an equivalent neural model.
Therefore, equivalent models produce local minima that have equivalent cost values on the loss function. 
However, it becomes challenging when the cost value of a big number of local minima deviate strongly from the cost value of the global minimum.
In this case, the learning process cannot generalize well to new data that was not involved in the training process. <br />

**Mitigation Techniques.**
Fortunately, choosing an appropriate model architecture to the learning task ensures that the majority of local minima have low loss value. 
In addition, it is sufficient to find a convenient local minimum that generalizes well on the task instead of finding the global minimum to update the model's parameters.


### Plateaus, Saddle Points and other Flat Regions
Along with local minima, saddle points are widely spread on the surface of cost functions of deep networks due to its nonconvex shape. 
A saddle point can be depicted as a local minimum when observing it from above, whereas it is a local maximum when observed from below. <br />
 The first and second order optimization methods deal differently with saddle points. When a first order optimization method, e.g. the [Gradient Descent algorithm](https://mlai-bonn.github.io/SeminarDeepLearning/s03_TrainingOptimization1.html) ([4.3](https://mlai-bonn.github.io/SeminarDeepLearning/s01_OptimizationMethods.html)), approaches a saddle point, it often decreases the gradient and moves with small steps downhill to escape this critical point. 
However, the second order optimization methods, e.g. [Newton's method](https://github.com/mlai-bonn/SeminarDeepLearning/blob/master/s04_TrainingOptimization2.md) ([4.6](https://mlai-bonn.github.io/SeminarDeepLearning/s04_TrainingOptimization2.html)), face challenges when dealing with saddle points on the surface of loss functions. They recognize the saddle point as a critical point with a zero gradient ($\frac{\partial L}{\partial x} = 0$) and may stop the optimization at this point instead of further descending to a minimal cost function value. In addition, deep neural networks of high dimensional spaces show that saddle points are much more proliferated than other critical points. This fact amplifies the challenge for the second order optimization methods to deal with saddle points. <br />

**Mitigation Technique.**
To mitigate this problem, the *saddle-free Newton method* was proposed to help the second order optimizers to quickly escape the saddle point. This method calculates the absolute value of the Hessian matrix. Consequently, the Hessian will treat this point as a local minimum and continue descending the loss function. 

In addition to saddle-points, plateaus and flat regions on the surface of the loss function cause problems during optimization because they are considered critical points ($\frac{\partial L}{\partial x} = 0$). 
If these flat regions have low cost value, they are treated as flat local minima with no drawbacks.
However, if flat regions have high cost values, the optimization algorithm adapts to make smaller steps and thus the learning process will slow down. <br />
Unfortunately, nonconvex loss functions contain a big amount of flat regions with high cost values and there are no mitigation techniques to solve this problem.


### Cliffs
A cliff is a region that undergoes a sharp fall or a sharp rise depending on the point of view with respect to this region. 
In both cases, it is dangerous to slide or to climb the cliff, and it is especially challenging to calculate the derivatives at such critical point because the gradient may surpass the cliff region to reach a point far away. 
The reason of this behavior is that the gradient at a cliff adapts only to the direction of the steepest descent when it moves forward and it disregards the optimal step size. <br />

**Mitigation Technique.** 
A mitigation technique for this unwanted behavior is to use *Gradient Clipping Heuristic* that reduces the step size to prohibit the gradient to jump over the cliff region.


#### Long-Term Dependencies
Neural networks process operations of the input vector over multiple layers forth and back.
When the network is very deep, e.g. [recurrent neural network](https://mlai-bonn.github.io/SeminarDeepLearning/s06_RecurrentNeuralNets.html) ([10](https://mlai-bonn.github.io/SeminarDeepLearning/s06_RecurrentNeuralNets.html)), the computation will result in a deep computational graph. 
During the computation, vanishing and exploding gradient descent appear. 
In the vanishing gradient descent situation, the gradients cannot decide in which direction to move to get into a convenient low cost value on the loss function.
 On the other hand, an exploding gradient descent makes the learning process inconsistent. <br />

**Mitigation Technique.**
A commonly used mitigation technique for both challenges is to drop uninteresting features in the input vector using the power method. <br />

**Example.** Suppose that a path of the computational graph applies a repeated multiplication with a matrix $W$, where $W = V diag(\lambda) V^{−1}$ is the eigendecomposition of $W$. <br />

 After $t$ multiplication steps, there are $W^{t}$ multiplications and the eigendecomposition becomes $W^{t} = V diag(\lambda)^{t} V^{−1}$. <br />

 The vanishing and exploding gradient descent problem arises from scaling $diag(\lambda)^{t}$. <br /> 

In this example, the power method detects the largest eigenvalue $\lambda_{i}$ of $W$ as well as its eigenvector and accordingly, 
it rules out all components that are orthogonal to $W$. The orthogonal components are the uninteresting features to be dropped.


### Poor Correspondence between Local and Global Structure
The presented mitigation techniques so far solved the optimization problem at a single point on the loss function. 
Although these methodologies improve the optimization process, it remains questionable whether the reached low cost value is sufficiently low with respect to other low cost values. Another question is whether the current low cost value drives the gradient into a much lower cost value or not. <br />

**Mitigation Techniques.** 
To answer these questions, experts advise to employ some heuristics. 
One heuristic is to force the gradient to start at good points on the loss function and thus ensure that it will converge to a convenient minimum quickly. Another heuristic is to seek a low cost value that generalizes well on the given task instead of seeking the global minimum on the loss function. 
In fact, the surrogate loss function presented in section [8.1](https://mlai-bonn.github.io/SeminarDeepLearning/s03_TrainingOptimization1.html) is usually used to calculate the gradient instead of using the true loss function. This fact amplifies the poor correspondence between local and global structures during optimization.



## 8.3 Basic Algorithms
A neural network adapts its parameters to reduce the loss arising from the difference between the estimated and true outputs. This optimization process is done using optimization algorithms. In this section, three first order optimization algorithms will be presented.

### Stochastic Gradient Descent (SGD)

#### SGD-Algorithm
The most popular optimization algorithm is the Stochastic Gradient Descent (SGD). 
SGD requires initial parameters $\theta$ as well as an adaptive learning rate $\epsilon_{k}$ at each iteration $k \in \mathbb{N}$. 
- First, SGD picks a minibatch consisting of $m$ examples from the training set $\left(\left(x^{(1)}, y^{(1)}\right), \dots, \left(x^{(m)}, y^{(m)}\right)\right)$. 
- On this minibatch, it computes a gradient estimate based on the normalized sum of the gradient of each example denoted as  $\hat{g} = \frac{1}{m} \nabla_{\theta} \sum_{i} L\left(f(x^{(i)}; \theta), y^{(i)}\right) $ for $i = 1, \cdots, m$. 
- Then, it applies the update step to the model's parameters $\theta = \theta - \epsilon_{k} \hat{g}$. 
- At each iteration $k$, a new $\epsilon_{k}$ is calculated and the algorithm runs until the convergence criterion is met, i.e. until a convenient low cost value is reached. <br />

#### SGD-Learning Rate $\epsilon_{k}$
In the SGD algorithm, the adaptive learning rate $\epsilon_{k}$ is essential because it determines the rate by which the model have to change the parameters according to the loss function. 
A suitable learning rate is chosen either by trial and error or by depicting the learning curve over time. 
Usually, $\epsilon_{k}$ decreases over time and in practice, a ratio $\alpha = \frac{k}{\tau}$ is defined to let $\epsilon_{k}$ decrease linearly until iteration $\tau$ by calculating
 $\epsilon_{k} = (1 - \alpha) \epsilon_{0} + \alpha \epsilon_{\tau},$ where
- $\tau$ is the number of iterations needed to make few hundred passes through the neural network,
- $\epsilon_{\tau} = \frac{\epsilon_{0}}{100}$  and 
- $\epsilon_{0}$ is the best performing $\epsilon_{k}$ in the first iterations. 

#### SGD-Convergence and Computation
The popularity of SGD is also related to the fact that it allows convergence even with a huge number of training examples. It needs $J(\theta) - \min_{\theta} J(\theta)$ to calculate the excess error for convergence. 
The excess error measures the difference between current cost function and the minimum cost value reached with optimal parameters $\theta$.

If SGD is applied to a convex problem, the excess error is $\mathcal{O}(\frac{1}{\sqrt{k}})$ after k iterations. Moreover, if it is applied to a strongly convex problem, the excess error becomes $\mathcal{O}(\frac{1}{k})$ after k iterations.



### Momentum

#### Momentum-Characteristics
The momentum algorithm is another optimization algorithm used during neural network training.
It adds to the gradient descent method a velocity parameter, also called momentum, to control the speed of the descent on the surface of the cost function. <br />

The velocity parameter improves the performance of gradient descent compared to SGD. 
If the region on the cost function shows high curvature or in the case of a small or a noisy gradient, SGD will take very small step sizes and the learning becomes slow. 
However, the momentum algorithm recognizes such regions and applies an additional force to the gradient descent to accelerate the learning along the cost function.
 While descending, the velocity increases due to the applied force and the gradient descent becomes faster. 
As a result of the increased velocity, the gradient overshoots a region or it experiences an oscillation on the two sides of a local minimum on the loss function. 
Therefore, it is important to have another force to help the gradient to stop at that local minimum. 
This force is called *viscous drag*, it calculates the negative of the velocity to resist to the force accelerating the gradient on the loss function. 

#### Momentum-Algorithm
The momentum algorithm requires an adaptive learning rate $\epsilon_{k}$, an adaptive momentum parameter $\alpha \in [0,1)$, an initial parameter $\theta$ and an initial velocity $v$. <br />
The velocity $v$ determines the direction and the speed by which the point on the loss function have to move. This velocity is the average of past gradients. Thus, the momentum algorithm considers previously calculated gradients and use them in the next move. To determine how much the contributions of previous gradients affect the current velocity, the momentum parameter $\alpha$  is used. In practice, $\alpha$ is set to $0.5$, $0.9$ or $0.99$ and it increases over time. <br /> 

- First, the momentum algorithm picks a minibatch consisting of $m$ examples from the training set $\left(\left(x^{(1)}, y^{(1)}\right), \dots, \left(x^{(m)}, y^{(m)}\right)\right)$. 
- On this minibatch, it computes a gradient estimate based on the normalized sum of the gradient of each example denoted as  $\hat{g} = \frac{1}{m} \nabla_{\theta} \sum_{i} L\left(f(x^{(i)}; \theta), y^{(i)}\right) $ for $i = 1, \cdots, m$. 
- Then, it computes the velocity update step  $v = \alpha v - \epsilon \hat{g}$. 
- After the velocity update, it applies the update step to the model's parameters $\theta = \theta + v$. 
- At each iteration $k$, a new $\epsilon_{k}$ is calculated and the algorithm runs until the convergence criterion is met, i.e. until a convenient low cost value is reached. <br />


### Nesterov Momentum


The Nesterov momentum is a third algorithm that can be used to optimize the training process of neural networks. It is an extension of the momentum algorithm because it adds a *correction factor* to the momentum. This correction is applied to the gradient estimation step by estimating $\hat{g} = \frac{1}{m} \nabla_{\theta} \sum_{i} L\left(f(x^{(i)}; {\bf \theta + \alpha v}), y^{(i)}\right)$ instead of $\hat{g}= \frac{1}{m} \nabla_{\theta}  \sum_{i} L \left(f(x^{(i)}; \theta), y^{(i)}\right)$, where $\theta + \alpha v$ is obtained after applying the momentum update step. The Nesterov momentum algorithm adds only one modification to the momentum algorithm in the gradient estimate step $g$, the remaining steps are the same. 



## 8.4 Parameter Initialization Strategies

Training algorithms for deep learning are usually iterative which means the user has to specify an initial point. The choice of the initial point affects convergence, the speed of convergence and if we converge to a point with high or low cost. This last aspect is important as points of comparable cost can have different generalization error and the goal of training neural networks is to minimize the generalization error. 

Most initialization strategies are based on achieving good properties when the network is initialized. There is no good understanding of how these properties are preserved during training. Certainly known is only that the initial parameters need to break symmetry between different units, which means that hidden units with same activation function and connection to same input parameters must have different initial parameters. This motivates to random initialization. <br /> 
More specifically, the weights are initialized randomly. The values are drawn either from a Gaussian or uniform distribution. We will denote the uniform distribution by $U$. The scale of the initial distribution has a large effect on the outcome, it influences optimization and generalization. Larger weights lead to stronger symmetry-breaking effect, but too large weights can cause exploding values during forward or backward-propagation or saturation of the activation function. Some heuristic initialization methods that are used in practice are: <br />

1. Sample each weight from $U(-\frac{1}{\sqrt{m}}, \frac{1}{\sqrt{m}})$, where m is the number of input layers. <br />
2. Normalized initialization: $W_{i, j} \sim U(-\sqrt{\frac{6}{m+n}}, \sqrt{\frac{6}{m+n}})$, where m and n are the number of inputs and outputs respectively and we assume fully connected layers.<br />
3. Initialize to random orthogonal matrices with gain factor g that needs to be carefully chosen. <br />
4. Use sparse initialization: each unit is initialized to have exactly k nonzero weights. <br /> 
The second approach compromises between having the same activation variance and same gradient variance among all layers.<br /> 
An advantage of sparse initialization over approach 1. and 2. is that it does not scale with the number of inputs or outputs. A disadvantage is that it imposed a large prior on weights with large values.

Optimal criteria for initial weights do not lead to optimal performance. This is why in practice the it is useful to treat initial weights as hyperparameters and to treat the initial scale of the weights and whether to use sparse or dense initialization as hyperparameter aswell if not too costly.

The approach for setting the biases must be coordinated with the approach for setting the weights. Setting the biases to zero is compatible with most weight initialization schemes. It is e.g. important to set the biases to nonzero weights, if a unit controls whether other units are able to participate in a function or to avoid too much saturation at initialization time.

It is also possible to initialize the model parameters using maching learning. This is not covered here but in part III of our source "Deep Learning".
 
# Questions

**Q:** Can you give an example for how networks can be nonidentifiable? 

Answer: For example by perturbing ingoing and outgoing weight vectors of two units in the same layer of a neural net we get two identical networks. This kind of nonidentifiability is called weight space symmetry. 

**Q:** Can you explain the concept of vicious drag? 

Answer: Vicious drag is the force in opposite direction of velocity that slows down a moving particle.

**Q:** Can you name advantages/disadvantages of a big/small momentum parameter for SGD?

Answer: The momentum parameter $\alpha$ in SGD with mom entum determines how fast the effect of previously compouted gradients decreases. As $\alpha$ gets smaller the previous gradients have exponentially less impact on the new direction of descent and the method approaches a usual SGD. The choice of $\alpha$ is highly specific to the application. Common values for $\alpha$ are 0.5, 0.9, and 0.99.

**Q**: Can you explain the Saddle-free Newton Method? 

Answer: The Newton Method tends to get stuck in saddle points. This can be fixed by employing a so called trust region approach. This approach consists of adding a constant value $\alpha$ to the eigenvalues of the Hessian matrix. Choosing $\alpha$ sufficiently large prevents the Newton Method from getting stuck in saddle points. 

**Q**: Why do we gradually decrease the learning rate over time in SGD? 

Answer: Since the gradient estimator produces a constant source of noise which leads to the gradients not vanishing entirely. 

**Q**: Can you give an example for exploding/vanishing gradients?

Answer: Repeated multiplications with the same weight Matrix $W$ can lead to vanishing or exploding gradients. An example of this can be found on Slide 38.

**Q**: How does the use of Nesterov Momentum improve the error rates?

Answer: For batch gradient descent on a convex function it can be shown that the use of Nesterov momentum improves the convergence of the excess error from $O(1/k)$ to $O(1/k^2)$. For SGD there are no such theoretical results.

**Q**: Can you come up with an example where shuffling the data is important before choosing the minibatches? 

Answer: Some data sets are arranged in a way such that subsequent examples are highly correlated which leads to the gradient estimations to not be independent. For example a list of blood sample test results might contain blood samples from single patients at different points in time in consecutive blocks. 
