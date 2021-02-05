# Training Optimization II

tbd

## Batch Normalization

Lets have a look at a very simple neural network, which consists of only linear operations on
scalars:

$\hat{y} = xw_1w_2 \cdots w_k$

The gradient of one parameter, e.g. $w_1$ is always derived under the assumption that $w_2 \cdots
w_k$ remain unchanged. However, we usually update all parameters at the same time with the update
rule

$w \leftarrow w - \epsilon g.$

Based on the gradient information, we expect $\hat{y}$ to decrease by $\epsilon g^Tg$ (first-order
Taylor approximation) because of the update. However, this estimate neglects all higher-order
effects (e.g. $\epsilon^2 g_1g_2 \prod_{i=3}^k w_i$). While this approximation is reasonable for
shallow networks, this higher-order effects might have a huge impact in deep networks.

We might try to mitigate this effects by using higher-order methods or a very low learning rate.
Unfortunately, both approaches come with huge disadvantages, especially in terms of computation
time. 

Instead we can use batch normalization. With batch normalization, activation values are normalized
after each linear transformation except for the last one.

**Batch Normalization**  
**Input:** $m$ examples $e^{(i)} \in \mathbb{R}^n$ ($n$ nodes in layer)
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
$\mu'$ and $\sigma'$ for each layer to which the activation values are shifted after normalization.
$\mu'$ and $\sigma'$ are optimized during training.

The use of batch normalization becomes apparent when looking at the simple example network from
before ($\hat{y} = xw_1w_2 \cdots w_k$). If we assume $x ~ N(0, 1)$ (x is drawn from a normal
distribution), then $xw_1$ cannot change the shape of the distribution, but only its parameters,
i.e. $xw_1 ~ N(0, \sigma^2)$. Batch normalization removes this change in standard deviation, so that
$x_w1 ~ N(0, 1)$ holds again. Therefore (under most circumstances) only $w_k$ can have an effect on
the output value $\hat{y}$. The neural network with batch normalization can still express all
functions it was able to express without batch normalization. However, it can be trained far easier
as there are no chaotic higher-order effects anymore. 

In general, one layer (the final layer) is sufficient to learn linear transformations and we are
only interested in the lower layers because of their non-linear effects. Therefore it is acceptable
to remove linear effects of lower layers by applying batch normalization.

Batch normalization has several advantages: It makes higher learning rates possible, makes the
initialization values of weights become less relevant and even acts as regularizer similar to
dropout in some cases. All in all it allows for an extreme speedup of training while increasing
performance as well.


## Supervised Pretraining

In some situations it might not be possible to train a model directly for the desired task, either
because of model characteristics (e.g. very deep models) or because of a difficult task. To overcome
this problem, we can try to give additional guidance for the training of parameters. This can be done with *greedy supervised pretraining*, *teacher student learning* or *transfer learning*.

### Greedy Supervised Pretraining

When applying greedy supervised pretraining, we perform one or multiple trainings before the actual
model training. In each of this pretrainings only a subset of model layers is trained and the
weights of obtained during the pretraining are re-used to initialize the actual training. 

One approach would be to train a shallow network with one hidden layer. After the training, the
output layer is discarded and another hidden layer is added. Now only the weights of the newly added
hidden layer are trained and the weight of the lower hidden layer remain fixed. This can be repeated
to get the desired number of hidden layers before jointly training all layers for fine-tuning.

### Teacher Student Learning

In teacher student learning, at first a shallow and wide teacher network is trained that is later
used to aid the training of the deep and thin student network. During training the student network
has the main objective to predict the correct outputs. But it also has a secondary objective to
predict the middle-layer values of the teacher network with one of its own middle layers. This is
supposed to give some guidance as to how the student network should use its middle layers.

Using this approach *Romero et al.* were able to train a student network that outperformed the
teacher network on CIFAR-10 with 90% less parameters.

### Transfer Learning

In transfer learning a model is trained on one task and afterwards all or a subset of its parameters
are used to initialize the training of a model on a similar task. Transfer learning relies on the
assumption that networks learn some general abstraction, especially on higher layers, that is useful
for many tasks. 

This strategy is used for example in language processing, where it is common to train large general
purpose models that are later fine-tuned to the specific task.

## Designing Models to Aid Optimization

*Goodfellow et al.* claim that

> an easy to optimize model family is more important than a powerful optimization algorithm. (p. 326)

But what is meant with "easy to optimize" in that case? Generally it is desireable that the local
gradient information occurring during training is useful for reaching a distant good solution.

To reach this goal it appears to be beneficial to have as much (near-)linearity in models as
possible. An example for this are ReLu activation functions, which are mostly preferred over sigmoid
or other activation functions today because they are  differentiable almost everywhere with a
significant slope.

A model architecture is beneficial for training if it allows for large gradients on low layers that
show the direction in which to move in parameter space well. There are multiple options to improve
this in deep networks.

One approach is to add *skip connections* in the network. This skip connections form "highways" which
allow for passing unchanged activation information to lower layers while skipping several layers. 
Another approach is to add *auxiliary heads* at hidden layers. This additional nodes are trained to
perform like output nodes, but are discarded after training.

## Continuation Methods

For applying continuation methods, we need to generate a series of cost functions $J^{(0)}, J^{(1)},
\dots$, where the last cost function $J^{(n)}$ equals the original cost function $J$ that we want to
optimize. This cost functions are supposed to be increasing difficult to optimize. During training we 
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
subsets of parameters until convergence to ensure that all of them receive updates. 

Coordinate descent can be useful if the parameters can be clearly separated into subsets and if the
variables in one subset have little to no influence on the optimal value of variables in other
subsets.

An example for this is [Lloyd's
algorithm](https://en.wikipedia.org/wiki/K-means_clustering#Standard_algorithm_(na%C3%AFve_k-means))
for computing a k-means clustering, which alternates between updating the cluster centroids and the
assignment of points to clusters.


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

#### Q: For simplicity, consider a model with only 2 parameters and both the initial gradients are 900. After some iteration, the gradient of one of the parameters has reduced to 200 but that of the other parameter is still around 650. Will the accumultion of square graidents from the very beginning affect the learning rate ? if so why and how? <br/>
A:  Accumulation from the very beginning of squared gradients can lead to excessive and premature decrease in the learning rate. So we will have accumulation at each update and because of this, there would be still the same value having by accumulated graident. Learning rate would be decreased by this, for both the parameters, learning rate would be reduced too much for the parameter having lower graident and thus leading to slower learning. 

#### Q: Can we solve the decrease in learning rate or slower learning problem via RMSProp <br/>
A: Yes, RMSprop helps with solving this issue. RMSprop uses exponentially weighted moving average by modifying the gradient acculmulation step to remove history from extreme past. 

#### Q: Consider a deep learning model that is learning to understand chess. In a standard way, Chess games are commonly divided into three levels a. Initial or Opening, b. Middle Stage, c. End Stage. So assume that the opening and end stages require very strong theoritical understanding, the middle stage is where most strategies and tacties are devised. Is it practical to train a single deep learning model to learn chess with the aforementioned scenario from a computation perespective ? if so, why ? <br/>
A: It is not practical, because to train individual subnetworks, supervised pre-training can be availed in opening, middle and End Stages and then combine them into deep learning model. It is because each of the hidden layer that is added is pretrained as part of a supervised multilayer perceptron and it is taking input as the output of the previously trained hidden layer. So in this case, instead of pretraining one layer at a time, we can pretrain a deep convolutional network and then use layers from this network to initialize even other deeper networks. The middle layers are initialized randomly of this new deep network. 

#### Q: The goal of Batch normalization is to normalize the features to zero mean states with standard deviation. So how does non-zero mean affect the model training ? <br/>
A:  In non-zero mean the data is not distributed around the value of 0, but data would be greater or less than 0. Greater variance problem along with non-zero mean can cause the data to get very large or small. It is a common problem when we are training a neural network with deep layers. 

#### Q: Why graident descent optimisers use exponential moving average for the graident component and root mean square for the learning rate component ? <br/>
A:  Gradient descrent optimizers use exponential moving average for gradient component where recent gradient values are given higher weights or importance than the previous ones, because, most recent gradient values provide more information than the previous ones if we are approaching minimum. If we divide the gradient by mean square, it will make the learning process much better and efficient. In order for our updates to be better guided, we need to make use of previous gradients and for this we will take exponential moving average of past gradients (mean square) then taking its square root hence called root mean square. 




