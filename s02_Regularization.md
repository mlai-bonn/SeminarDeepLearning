# Regularization Methods

## Introduction
The main problem we were facing here is overfitting and methods to avoid that.
The definition of Reguralization is the following:

Define Regularization as “any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error.”

We talked about several Reguralization technics:

## Parameter Norm Penalties
The main idea here is to limit the  model complexity by adding a parameter norm penalty, denoted as Ω(θ), to the objective function J:
$ \tilde{J}(\theta;X, y) = J(\theta;X, y) + \alpha\Omega(\theta)$.

Most importantly, Parameter θ represents the weights only and not the biases.

## $L_1$, $L_2$ Regularization

$L_1$ (a.k.a Least Absolute Shrinkage and Selection Operation - LASSO) and $L_2$ (a.k.a Ridge Regression, Tikonohov Regularization, Weight Decay) Regularizations update the general cost function $J$ by adding a term known as the regularization term $\alpha$.
Because of $\alpha$ the values of weights decrease because it assumes that a neural network with smaller weights leads to simpler models.
Therefore, it will also help in reducing overfitting in an efficient manner.

In $L_2$, we have:

$ \tilde{J}(w;X, y) = \frac{\alpha}{2}w^Tw + J(w;X,y) $

Here, alpha is the regularization parameter.  It is the hyperparameter whose value is optimized for better results. $L_2$ regularization is also known as weight decay as it forces the weights to decay towards zero (but not exactly zero).

In $L_1$, we have:

$ \tilde{J}(w;X, y) = \alpha \| w\|_1 + J(w;X,y)$

Unlike $L_2$, the weights may be reduced to zero here


## Data Augmentation

Data Augmentation solve the problem of processing Limited Data with less diversity in order to get efficient results from the neural network.
Let us think of Data Augmentation as an added noise to our dataset. The idea here is to add new training examples when the data supply is limited.

Following are the most popular Data Augmentation Techniques:

1. Flip
2. Rotation
3. Scale
4. Crop

## Noise Robustness
Noise with extremely small variance imposes a penalty on the norm of the weights. Noise can even be added to the weights. This has several interpretations.
One of them is that adding noise to weights is a stochastic implementation of Bayesian inference over the weights, where the weights are considered to be uncertain, with the uncertainty being modelled by a probability distribution
It is also interpreted as a more traditional form of regularization by ensuring stability in learning.

For e.g. in the linear regression case, we want to learn the mapping y(x) for each feature vector x, by reducing the mean square error.

$J = E_{p(x,y)}{[}(\hat{y}{(x)}{-}y)^2{]} $

Now, suppose a zero mean unit variance Gaussian random noise, ϵ, is added to the weights. We still want to learn the appropriate mapping through reducing the mean square.

Minimizing the loss after adding noise to the weights is equivalent to adding another regularization term which makes sure that small perturbations in the weight values don’t affect the predictions much, thus stabilising training.

## Early Stopping

For a model with high representational model complexity, the training error continues to decrease but the validation error begins to increase (which we referred to as overfitting).
In such a scenario, a better idea would be to return back to the point where the validation error was the least. Thus, we need to keep calculating the validation metric after each epoch and if there is any improvement, we store that parameter setting. Upon termination of training, we return the last saved parameters.

The idea of Early Stopping is that if the validation error doesn’t improve over a certain fixed number of iterations, we terminate the algorithm.
This effectively reduces the capacity of the model by reducing the number of steps required to fit the model. The evaluation on the validation set can be done both on another GPU in parallel or done after the epoch. A drawback of weight decay was that we had to manually tweak the weight decay coefficient, which, if chosen wrongly, can lead the model to local minima by squashing the weight values too much. In Early Stopping, no such parameter needs to be tweaked which reduces the number of hyperparameters that we need to tune.
L²-regularization can be seen as equivalent to Early Stopping.


## Parameter Tying/Share and Dropout
Another way of implementing prior knowledge into the training process.

### Parameter Tying
Express that certain parameters should be close to each other taken from two different models $A$ and $B$.

Changing their loss functions with the additive Regularization term: $\Omega(w^{(A)}, w^{(B)}) := \vert\vert w^{(A)}-w^{(B)} \vert\vert_2^2$

### Parameter Sharing
Here we force sets of parameters in one (or multiple models) to be equal. Used for examples Heavy in CNN training, where the feature detectors of one layer get set on the same parameter set.

The Advantage here is the massive reduction of space (brings the ability to train larger models) and another implement of prior knowledge.

### Dropout
Here we improve generalization and speed up the model training by training the ensemble of all sub-networks of a NN.

Where a subnet is a subgraph of the original NN connecting (at least some) input neurons to the output.
When training a subnet its weights are shared with the original net.


Since there are too many subnets to train we sample a subnet by selecting each non ouput neuron $n$ in layer $l$ with probability $p_l$
($p_l$ for the input layer is usually high with $0.8$ and $0.5$ for the other layers).

Training routine is then: Sampling subnet, Training subnet, adjust wheigts in original net and iterate.

Once training is done one can predict new data points by usual forward propagation but with each weight $w$ in layer $l$ multiplied by $p_l$.

In conclusion Dropout provides a cheap Regularization method (implementable in $\mathcal{O}(n)$),
which forces the model to generalize since it is trained with subnets which have various topologies and a smaller capacity.



## Adverserial Training
Motivation: There a many models capable of reaching (or even exceeding)
human performance on specific tasks (as Chess or GO). But do they also gather a human understandment of the game?

No they don't. That can be seen very well on object recognition tasks.
Here often a little noise applied to the input image (barely distinguishable by a human), leads to a huge change in prediction.
Those examples are called Adversarial examples.


By training a model on many Adversarial examples one tries to force the model to implement a prediction stable plateau around each of the training data points,
because prior knowledge tells us that two different object in an image have a larger distance regarding the pixel values than a little noise could inject.



## Questions
