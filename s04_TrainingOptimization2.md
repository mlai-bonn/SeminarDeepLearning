# Training Optimization II

tbd

## Batch Normalization

In order to understand batch normalization, lets first have a look at a very simple neural 
network, which consists of only linear operations on scalars:

$\hat{y} = xw_1w_2 \cdots w_k$

The gradient of one parameter, e.g. $w_1$, is always derived under the assumption that $w_2 \cdots
w_k$ remain unchanged. However, we usually update all parameters at the same time with the update
rule

$w \leftarrow w - \epsilon g.$

Suppose we want to tune $w$ such that the value of $\hat{y}$ decreases. Based on the gradient 
information, we expect $\hat{y}$ to decrease by $\epsilon g^Tg$ when applying the above update rule 
(first-order Taylor approximation). However, this estimate neglects all higher-order
effects (e.g. $\epsilon^2 g_1g_2 \prod_{i=3}^k w_i$). While this approximation is reasonable for
shallow networks, the higher-order effects might have a huge impact in deep networks.

We might try to mitigate this effects by using higher-order methods or a very low learning rate.
Unfortunately, both approaches come with huge disadvantages, especially in terms of computation
time. 

Instead we can use batch normalization. With batch normalization, activation values are normalized
after each linear transformation except for the last one.

**Algorithm: Batch Normalization**  
**Input:** $m$ examples $e^{(i)} \in \mathbb{R}^n$ ($n$ nodes in layer, batch size $m$)
1. calculate $n$ means $\mu_j = \frac{1}{m} \sum_{i=1}^m e^{(i)}_j$
2. calculate $n$ standard deviations $\sigma_j = \sqrt{\delta + \frac{1}{m} \sum_{i=1}^{m}
      (e^{(i)}_j - \mu_j)^2}$ with a small positive $\delta$ (e.g. $10^{-8}$)
3. replace $e^{(i)}$ by $e'^{(i)} = (\frac{e^{(i)}_1-\mu_1}{\sigma_1},
      \frac{e^{(i)}_2-\mu_2}{\sigma_2}, \dots, \frac{e^{(i)}_n-\mu_n}{\sigma_n})^T$

This normalization is taken into account when calculating gradients via back-propagation. Therefore
the gradient will never propose a change that solely changes mean or standard deviation of the
activation values. During testing $\mu$ and $\sigma$ are usually replaced by average values
collected during training. The algorithm above normalizes all activation values to zero mean and
unit standard deviation. A common expansion of this approach is to introduce additional parameters
$\mu'$ and $\sigma'$ for each layer and to shift the distribution of activation values to $\mu'$ mean 
and $\sigma'$ standard deviation after normalization. $\mu'$ and $\sigma'$ are optimized during 
training just like other parameters.

The benefit of batch normalization becomes apparent when looking at the simple example network 
introduced above ($\hat{y} = xw_1w_2 \cdots w_k$). If we assume $x \sim N(0, 1)$ (x is drawn from a normal
distribution), then $w_1$ cannot change the shape of the distribution, but only its parameters,
i.e. $xw_1 \sim N(0, \sigma^2)$. Batch normalization removes this change in standard deviation, so that
$xw_1 \sim N(0, 1)$ holds again. As batch normalization is applied to all weights $w_1$ to $w_{k-1}$, only 
$w_k$ can have an effect on the output value $\hat{y}$ (under most circumstances). The neural network with 
batch normalization can still express all functions it was able to express without batch normalization. 
However, it can be trained far easier as there are no chaotic higher-order effects anymore. 

To put it simply, one layer (the final layer) is sufficient to learn linear transformations and we are
only interested in the lower layers of a neural network because of their non-linear effects. Therefore it is 
acceptable to remove linear effects of lower layers by applying batch normalization.

Batch normalization has several advantages: It makes higher learning rates possible, makes the
initialization values of weights become less relevant and even acts as regularizer similar to
dropout in some cases. All in all it allows for an extreme speedup of training while increasing
performance as well.


## Supervised Pretraining

In some situations it might not be possible to train a model directly for the desired task, either
because of model characteristics (e.g. very deep models) or because of a difficult task. To overcome
this problem, we can try to give additional guidance for the training of parameters. This can be 
done with *greedy supervised pretraining*, *teacher student learning* or *transfer learning*.

### Greedy Supervised Pretraining

When applying greedy supervised pretraining, we perform one or multiple trainings before the actual
model training. In each of this pretrainings only a subset of model layers is trained and the
weights obtained during the pretraining are re-used to initialize the actual training. 

One approach would be to train a shallow network with one hidden layer. After the training, the
output layer is discarded and another hidden layer is added. Now only the weights of the newly added
hidden layer are trained and the weights of the lower hidden layer remain fixed. This can be repeated
to get the desired number of hidden layers before jointly training all layers for fine-tuning.

### Teacher Student Learning

In teacher student learning, a shallow and wide teacher network is used to aid the training of a 
deep and thin student network. During training the student network has the main objective to predict 
the correct outputs. In addition to this, it also has a secondary objective to predict the 
middle-layer values of the teacher network with one of its own middle layers. This is supposed to give 
some guidance as to how the student network should use its middle layers.

Using this approach *Romero et al.* were able to train a student network that outperformed the
teacher network on CIFAR-10 with 90% less parameters ([source](https://arxiv.org/abs/1412.6550)).

### Transfer Learning

In transfer learning a model is trained on one task and afterwards all or a subset of its parameters
are used to initialize the training of a model on a similar task. Transfer learning relies on the
assumption that networks learn some general abstraction, especially on higher layers, that is useful
for many tasks. 

This strategy is used for example in language processing, where it is common to train large general
purpose models that are later fine-tuned to the specific task.

## Designing Models to Aid Optimization

*Goodfellow et al.* claim that

> it is more important to choose a model family that is easy to optimize than to use a powerful 
> optimization algorithm. (["Deep Learning"](https://www.deeplearningbook.org/), p. 322)

But what is meant with "easy to optimize" in that case? In general, it is desireable that the local
gradient information occurring during training is useful for reaching a distant good solution.

To reach this goal it appears to be beneficial to have as much (near-)linearity in models as
possible. An example for this are ReLu activation functions, which are mostly preferred over sigmoid
or other activation functions today because they are  differentiable almost everywhere with a
significant slope.

In a deep network the model architecture should allow for large gradients on low layers that
show the direction in which to move in parameter space well. There are multiple options to improve
this.

One approach is to add *skip connections* in the network. This skip connections form "highways" which
pass unchanged activation information to lower layers while skipping several layers. 
Another approach is to add *auxiliary heads* at hidden layers. This additional nodes are trained to
perform like output nodes, but are discarded after training.

## Continuation Methods

For applying continuation methods, we need to generate a series of cost functions $J^{(0)}, J^{(1)},
\dots$, where the last cost function $J^{(n)}$ is the original cost function $J$ that we want to
optimize. This cost functions are supposed to be increasingly difficult to optimize. During training we 
start with the easy cost functions and proceed to the difficult ones as the training continues. 

The goal of this method is to ensure that the local optimization happens mostly in well-behaved
regions of parameter space. A region is regarded as well-behaved if the gradient information there
is useful for finding the desired (probably distant) solution.

The series of cost functions is usually created by smoothing or blurring the original cost function.
The intuition behind this process is that some non-convex cost functions might become convex when
blurred while still preserving their global minimum.

### Curriculum Learning

In curriculum leaning, the model first learns simple concepts before proceeding to more complicated
ones. This can be interpreted as generating a series of cost functions $J^{(i)}$ where functions
with a lower index depend on simpler examples. This method is successfully used in natural language
processing and computer vision and is close to how humans learn.

## Coordinate Descent

The main idea of coordinate descent is to only update a subset of parameters at each optimization
step instead of updating all parameters at once. The optimization algorithm alternates between all
subsets of parameters to ensure that all of them receive updates. 

Coordinate descent may be useful if the parameters can be clearly separated into subsets and if the
parameters in one subset have little to no influence on the optimal values of parameters in other
subsets.

An example for this is [Lloyd's
algorithm](https://en.wikipedia.org/wiki/K-means_clustering#Standard_algorithm_(na%C3%AFve_k-means))
for computing a k-means clustering, which alternates between updating the cluster centroids and 
assigning points to clusters.


## Polyak Averaging

In Polyak averaging the current location in parameter space is determined by averaging over all
previous locations visited by the optimization algorithm, i.e.

$\hat{\theta}^{(t)} = \frac{1}{t} \sum_{i=1}^t \theta^{(i)}$

This approach has strong convergence guarantees on some problem classes. For neural networks there
are no convergence guarantees, but it was found to perform well in practice.

For use with neural networks the unweighted average is usually replaced by a exponentially decaying
running average to decrease the influence of locations visited a long time ago:

$\hat{\theta}^{(t)} = \alpha \hat{\theta}^{(t-1)} + (1- \alpha)\theta^{(t)}$


## Questions

#### Q: What was the problem in SGD that we tried to address in Adaptive Learning Rate Methods (i.e RMSprop/ADAM optimizer)? <br/>
A: The problem in SGD is the constant learning rate, if we are using small learning rate during the whole training phase the convergce will be very slow (and may stuck to saddle point), while using large learning rate may cause overshooting the local minimum. Furthermore the SGD has a zigzag shape of updating the gradient, specially in the dimensions with large gradients the change in the gradient is going up and down during the iterations.

#### Q: What is the advantage of using a separate learning rate? <br/>
A:  In the optimization space, we have different gradients for each dimension, some of them are large need a small learning rate to avoid overshooting the local minimum, and others are small need a larger learning rate to get a faster convergence to the local minimumm. Therefore we need a separate learning rate for each dimension to avoid the mentioned problems.

#### Q: What is the advantage of using momentum ? <br/>
A: We can solve the zigzag motion in the SGD by smoothing the gradient in each dimension by calculating the momentum of the gradient. Which means instead of computing the gradient at each iteration we will compute the exponential weighted/moving average of the gradient.


#### Q: Why for the learning rate decaying we used exponential weighted/moving average rather than the normal averaging method? <br/>
A: The normal averaging method needs memory - of different variables - to save the history of the gradients and then compute the average of those values. While the moving/weighted average is only one variable that saves the average on the previous gradients. Furthermore the exponential weighted/moving average gives higher wieghts for the present gradients and lower weights for the past gradients, with decaying weights (until vanishing) for the far past gradients; so that the learning rate will not shrink before achieving the local minimum.

#### Q: Why graident descent optimisers use exponential moving average for the graident component and root mean square for the learning rate component ? <br/>
A:  Gradient descrent optimizers use exponential moving average for gradient component where recent gradient values are given higher weights or importance than the previous ones, because, most recent gradient values provide more information than the previous ones if we are approaching minimum. If we divide the gradient by mean square, it will make the learning process much better and efficient. In order for our updates to be better guided, we need to make use of previous gradients and for this we will take exponential moving average of past gradients (mean square) then taking its square root hence called root mean square. 

#### Q: How will using a separate learning rate help in the convergence of the optimization? <br/>
It will lead to a faster convergence as the learning rate changes through the optimization space to adapt with different structures of the space. Such that at steep area you need a low learning rate to not overshoot local minimum and at plane area you need a high learning rate to not stuck in saddle point.

#### Q: Can we solve the decrease in learning rate or slower learning problem via RMSProp <br/>
A: Yes, RMSprop helps with solving this issue. RMSprop uses exponential weighted average by modifying the gradient acculmulation step to remove history from extreme past. 

#### Q: What is the advantage of Conjugate Coordinates? How to avoid the repramatization caused by SGD? <br/>
A: The advantage of Conjugate Coordinates method that the gradient of the each step take into consideration the progress in the previous direction, to avoid computing a new direction that ruin the progress achieved by the previous one, which leads to less steps until convergence. Where ruining the previous gradient direction leads to the need of reminimizing it to retain the previous progress, which leads to the zigzag motion happens by the normal steepest decent method.

#### Q: How BFGS try to make use of Newton’s Method with less computational burden? And what is the advantage of using the limited memory edition of it? <br/>
A: The idea of BFGS is to approximate the inverse of the Hessian Matrix with a matrix M that is iteratively refined by a low rank updates. In its limited memory edition, it avoides storing the complete inverse Hessian approximation matrix M, and assumes that the matrix M in the previous step was an identity matrix.

#### Q: The goal of Batch normalization is to normalize the features to zero mean states with standard deviation.What is the advantage of this normalization? And how does non-zero mean affect the model training ? <br/>

A: The advantage of normalizing the features to zero mean, is that the distribution in all dimensions will be symmetric which leads to faster convergence. Non-zero mean data are used to have a problem called Covariate Shift of the distribution, where the distribution of the features changes by updating the previous layer parameters. Which causes deep neural networks struggle to converge.

#### Q: In Batch Normalization why do we transform the standard deviation distribution to arbitrary one (using gamma and beta learning parameters)? What if the resulted arbitrary distribution is the same as distribution before the batch normalization? Is the arbitrary distribution still symmetric? <br/>
A: To keep the flexiblity of the model to gain more capabilities by mapping to any arbitrary distribution that fits its training phase. There won't be a problem if the arbitraty distribution mapped back to the same distribution as before the batch normalization; as we already isolated the arbitrary distribution of this layer from the previous layer parameter updates and prevented the covariate shift problem. Yes the resulted arbitrary distribution is still symmetric in all dimensions as we mapped the distribution in all dimensions with the same gamma and beta parameters.

#### Q: What is the effect of the batch size on the Batch Normalization? <br/>
A: From statiscs point of view the mean and variance are meaningless if you have small number of data members; therefore to make use of the advantage of Batch Normalization, you should avoid using small batch sizes.

#### Q: What is the effect of Batch Normalization in the parameter initialization? Is it increasing its importance or decreasing it? <br/>
A: The Batch Normalization is robust to parameter initialization; as it already convert the distribution of the features to arbitrary distribution.

#### Q: What is the approach of the Batch Normalization in the test time? <br/>
A: In the test phase you may pass one example at a time, so there is no batch of examples to compute the mean and the variance, so it is better to estimate a mean and variance from the training examples using the exponential weighted average.

#### Q: What is the advantage of transfer learning? What is purpose of fine tunning? <br/>
A: it helps paramater intialization and making use of

#### Q: In Greedy Algorithms: does combining individual optimal solutions guarantee to get one complete optimal solution? <br/>
A: That is not guaranteed. However, the resulted solution is computationally much cheaper and still improved solution than a joint solution even (if it is not optimal).

#### Q: What is the advantage of transfer learning? What is purpose of fine tunning? <br/>
A: Having a good intialization of the parameters that leads to faster training phase, making use of larger datasets (i.e ImageNet), getting better performance by having your model pretrained by another dataset.

#### Q: In case of Continuation Methods: What if the cost function is not convex? <br/>
A: It will still give us an improved results even if this results are not global minimum.

#### Q: For simplicity, consider a model with only 2 parameters and both the initial gradients are 900. After some iteration, the gradient of one of the parameters has reduced to 200 but that of the other parameter is still around 650. Will the accumultion of square graidents from the very beginning affect the learning rate ? if so why and how? <br/>
A:  Accumulation from the very beginning of squared gradients can lead to excessive and premature decrease in the learning rate. So we will have accumulation at each update and because of this, there would be still the same value having by accumulated graident. Learning rate would be decreased by this, for both the parameters, learning rate would be reduced too much for the parameter having lower graident and thus leading to slower learning. 

#### Q: Consider a deep learning model that is learning to understand chess. In a standard way, Chess games are commonly divided into three levels a. Initial or Opening, b. Middle Stage, c. End Stage. So assume that the opening and end stages require very strong theoritical understanding, the middle stage is where most strategies and tacties are devised. Is it practical to train a single deep learning model to learn chess with the aforementioned scenario from a computation perespective ? if so, why ? <br/>
A: It is not practical, because to train individual subnetworks, supervised pre-training can be availed in opening, middle and End Stages and then combine them into deep learning model. It is because each of the hidden layer that is added is pretrained as part of a supervised multilayer perceptron and it is taking input as the output of the previously trained hidden layer. So in this case, instead of pretraining one layer at a time, we can pretrain a deep convolutional network and then use layers from this network to initialize even other deeper networks. The middle layers are initialized randomly of this new deep network.




