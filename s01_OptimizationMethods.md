# Optimization Methods
## Gradient-Based Optimization (4.3)
Optimization Methods are widely used and needed for for deep learning algorithms.
The general Task is the following: given a function $f : \mathbb{R}^n \to \mathbb{R}$ find $x^* := \underset{x\in \mathbb{R}^n}{ \text{argmin} } f(x)$ that minimizes $f$. How do we solve this problem?

**Idea:** 
We start at some initial value $x_{0}$ and iteratively move in the direction of steepest descent $u$ from there until convergence.
The update rule: $x_{i+1} \leftarrow x_{i} + \varepsilon u$

The following questions arise:
- How to find the direction of steepest descent $u$?
- How to find a good stepsize $\varepsilon$?

### Gradient and Directional Derivative
Recap of some basics from multidimensional analysis:
- **Partial derivative** $\frac{\partial}{\partial x_i}f(x)$:  derivative of $f$ with respect to $x_{i}$
- **Gradient** $\nabla_x f(x) := \left(\frac{\partial}{\partial x_1}f(x),\ldots,\frac{\partial}{\partial x_n}f(x)\right)^{\intercal}$: vector of all partial derivatives of $f$
- **Directional derivative in direction $u$**:  To obtain the directional derivative of $f$ in direction $u$ we compute $\frac{\partial}{\partial \alpha}f(x + \alpha u)\ \text{evaluated at } \alpha = 0$. This is equal to $u^\intercal \nabla_x f(x)$. We want to find the direction $u$ with minimal directional derivative in order to minimize $f$. 

$\rightarrow$ Hence, our task is to find $\underset{u, \|u\| = 1}{\text{argmin}}\ u^\intercal \nabla_x f(x)$.

### Direction of steepest descent
We compute: 
$\underset{u, \|u\| = 1}{\text{argmin}}\ u^\intercal \nabla_x f(x) =\ \underset{u, \|u\| = 1}{\text{argmin}}\ \| u\|_2\| \nabla_x f(x)\|_2 \cos(\alpha) 
		=\  \underset{u, \|u\| = 1}{\text{argmin}}\cos(\alpha)$
		
($\alpha$ denotes the angle between $u$ and $\nabla_x f(x)$)

The $cos(\alpha)$ is minimized when $u$ points into the opposite direction of the gradient.
$\rightarrow$ Set $u := - \nabla_x f(x)$.

### Jacobian and Hessian Matrix
- Consider a function $g: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$. This means that $g$ consists of $m$ functions $g_1, \ldots , g_m: \mathbb{R}^{n} \rightarrow \mathbb{R}$. The **Jacobian** matrix of $g$ is defined as: $J \in \mathbb{R}^{m\times n}$, $J_{ ij } :=$  $\frac{\partial f_{i}}{\partial x_{j}}$
- Consider $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$. Its Hessian is defined as $H \in \mathbb{R}^{n \times n}$, $H_{ ij } :=$ $\frac{ \partial }{ \partial x_{i} \partial  x_{j} } f$. It contains information about the curvature of $f$.

### The optimal stepsize $\varepsilon$
To find the optimal steptsize $\varepsilon$ we do a second order Taylor approximation of $f$:
$f(x^{(i+1)})$ 
$\approx  f(x^{(i)}) + (x^{(i+1)} - x^{(i)})g + \frac{1}{2}(x^{(i+1)} - x^{(i)})^{\intercal}H(x^{(i+1)} - x^{(i)})$
$= f(x^{(i)})  - \varepsilon g^{\intercal} g + \frac{1}{2} \varepsilon^{2} g^{\intercal} H g$,
where $g := \nabla_{x}f(x^{(i)})$ and $H := H(f)(x^{(i)})$.
Since we want to minimize $f$ we increase $\varepsilon$ if $g^{\intercal} Hg \leq 0$ else we set $\varepsilon := \frac{g^\intercal g}{g^\intercal H g}$.

### Issues of Gradient Descent
An ill-conditioned Hessian matrix leads to poor performance of the gradient descent algorithm. We can resolve this problem using Newton's method.

### Newton's Method
We again use a second order Taylor approximation of $f$:
$f(x^{(i+1)}) \approx  f(x^{(i)}) + (x - x^{(i)})^\intercal \nabla_{x}f(x^{(i)}) 
		 + \frac{1}{2}(x - x^{(i)})^\intercal H(f)(x^{(i)})(x - x^{(i)})$
Hence, the optimum is $x^{(i+1)} = x^{(i)} - H(f)(x^{(i)})^{-1}\nabla_{x}f(x^{(i)})$
			
## Constrained Optimization (4.4)
 We again want  to minimize $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ but this time with additional conditions.
 The constraints are given by:
 - $g_{i}(x) \leq 0$ for $i = 1,...,m$
- $h_{i}(x) = 0$ for $j = 1,...,k$ 
- with $g_{i}, h_{j}: \mathbb{R}^n \rightarrow \mathbb{R}$

We want to find $x$ that  minimizes $f$ under the given conditions. We are going to do that by translating our initial constrained optimization problem into an unconstrained one.
The KKT-approach uses the general Lagrangian 
$\mathcal{L}(x, \lambda, \mu) := f(x)+ \sum_{i}\lambda_{i}g_{i}(x) + \sum_{j}\mu_{j}h_{j}(x)$ where $\lambda \in \mathbb{R}^m_{\geq 0}, \  \mu\in\mathbb{R}^k$. 
Our intial problem is then equivalent to solving $\underset{x}{\min} \ \underset{\mu}{\max} \  \underset{\lambda, \lambda \geq 0}{\max} \ \mathcal{L}(x, \lambda, \mu)$.
It can be shown that necessary conditons for a local optimum $(x , \lambda , \mu )$ are:
- all constraints are satisfied by $(x , \lambda , \mu )$
- $\nabla_{x}\mathcal{L}(x , \lambda , \mu ) = 0$
- $\lambda_{i} \geq 0, \ \lambda_{i} g_{i}(x ) = 0$ for $i = 1,...,m$
 
## Example: Least Linear Squares (4.4)

Minimize $f(x) = \frac{1}{2}\|Ax-b\|^2$, $f: \mathbb{R}^n \rightarrow \mathbb{R}, \ A\in\mathbb{R}^{m\times n}, \ b\in\mathbb{R}^m$.
- Gradient descent:  $x^{i+1} = x^{i} - \varepsilon \nabla_{x}f(x)$ where $-\nabla_{x}f(x) = A^\intercal (Ax-b)$
- Newton's method: Converges in 1 step because the optimization problem is strictly convex.
- KKT-approach: Suppose we have an additonal constraint $x^\intercal x \leq 1$.
Then $\mathcal{L}(x, \lambda) = f(x) + \lambda (x^\intercal x - 1)$.

## Gradient-Based Learning (6.2)

In this section we want to apply the previously discussed optimization methods to deeplearning models. More precisely our goal is to approximate some function $f^\star$ by training a feedforward network $y = f(x; \theta)$ which learns the value of the parameters $\theta$. These feedforward networks are typically represented by a composition of different functions such as $f(x) = f_3(f_2(f_1(x)))$. We call $f_i$ the $i$-th layer, $x$ the input layer, and the last layer (e.g. $f_3$) the output layer.
The main difference in training these networks as opposed to linear models such as logistic regression or SVMs lies in the nonlinearity of neural networks. This increase in descriptive power leads to more complicated loss functions which are generally nonconvex. Minimizing nonconvex functions typically calls for an iterative, gradient-based approach such as the previously discussed gradient descent. This approach is highly sensitive to the values of the initial parameters and cannot guarantee any global convergence. We can merely hope to drive the value of our cost function to a very low value rather than a global optimum. 
In the following we first discuss how to choose an appropriate cost function. We then go on to discuss a way to represent the output according to our model. Finally we describe how to compute the gradient. 

### Cost Functions

In most cases the parametric model defines a distribution $p_{\text{model}}(y \vert x; \theta)$. We use the principle of maximum likelihood to arise at the cost function

- $J(\theta) := -E_{x,y : p_{\text{data}}} \log p_{\text{model}}(y \vert x),$

where $\theta$ is our model parameter and $p_{\text{data}}$ is the empirical distribution with respect to the training data. Minimizing this cost function corresponds to minimizing the cross-entropy between the traing data and the model distribution. The specific form of this cost function changes depending on the specific form of $p_{\text{model}}$.  

### Output Units

We assume that the feedforward network produces some hidden features $h = f(x; \theta)$. The goal of the output unit is to transform $h$ into some appropriate representation that completes the task according to our model. To illustrate this we present two small examples.

### Example 1: Sigmoid Units for Bernoulli Output Distributions

Assume we want predict the value of a binary variable $y$. We want our neural net to predict the probability $P(y = 1 \ | \ x)$ which means that the output unit must transform $h$ into some value in $[0,1]$. 
A naive approach would be to use a linear unit on $h$ and threshold its value:

- $P(y = 1 \ \vert \ x) = \max( 0, \min( 1, w^\intercal h + b ) )$

This turns out to be a poor decision for gradient descent since the gradient of this function is $0$ any time that $w^\intercal h + b$ lies outside of the unit interval.
A better solution would be the following. We first compute $z = w^\intercal h + b$ in a linear layer to then output

- $P(y = 1\ \vert \ x) = \sigma(z),$

where $\sigma(z) := \frac{1}{1+\exp(-x)}$ is the logistic sigmoid function. The sigmoid function can be motivated by starting from an unnormalized probability distribution $\tilde{P}(y)$ and the assumption $\log\tilde{P}(y) = yz$. We construct a normalized probability distribution $P(y)$ in the following way.

- $\tilde{P}(y)=\exp(yz),$
- $P(y) = \frac{\exp(yz)}{\sum_{y^\prime = 0,1}\exp(y^\prime z)},$
- $P(y) = \sigma((2y - 1)z).$

This approach yields a bernoulli distribution controlled by a sigmoidal transformation of $z$.
The cost function for maximum likelihood learning is given by 

- $J(\theta) = -\log P(y \vert x) = -\log \sigma((2y - 1)z) = \zeta((1-2y) z),$

where $\zeta(x) := \log(1 + \exp(x))$ is the softplus function. This cost function turns out to be very suitable for gradient descent. If we plug $y = 1$ into $J(\theta)$ (i.e. the correct classification is $1$) we get $J(\theta) = \zeta(-z)$ which saturates (i.e. gradient descent stops) for very positive values of $z$. For $y = 0$ we get $J(\theta) = \zeta(z)$ which saturates for very negative values of $z$. 

### Example 2: Softmax Units for Multinoulli Output Distributions

Assume we want to represent a probability distribution over $n$ discrete variables. In this case we need to produce a vector $\hat{y}$ with $\hat{y}_i = P(y = i | x)$ for $i = 1,\ldots,n$. 
Similarly to Example 1 we let a linear layer predict unnormalized $\log$-probabilities:

- $z = W^\intercal h + b$
- $z_i = \log \tilde{P}(y = i \vert x)$.

The output is then computed as 

- $\hat{y}_i = \text{softmax}(z)_i := \frac{\exp (z_i)}{\sum_j exp(z_j)}$.

Using the maximum likelihood approach we want to maximize

- $-J(\theta) = \log P(y = i \vert x) = \log \frac{\exp (z_i)}{\sum_j exp(z_j)} = z_i - \log \sum_{j} \exp(z_j)$.

To gain an intuition for this formula we observe that $log \sum_{j} exp(z_j) \approx max_j z_j$. If $y = i$ is the correct classification we can see that small values for $z_i$ (i.e. incorrect answers) are penalized the most whereas if $z_i = \max_j z_j$ (i.e. correct answer) both terms roughly cancel and we get a high log-likelihood.

## Backpropagation-Algorithms (6.5)

We now want to minimize the cost function $J(\theta)$ using gradient descent. It is thus necessary to compute the gradient $\nabla_{\theta}J(\theta)$. We will do this by letting information flow backwards through the layers of the network. This method is called backpropagation. 

### The Chain Rule of Calculus
Given to functions $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ and $g: \mathbb{R}^m \rightarrow \mathbb{R}$ we want to compute the partial derivatives of their composition.
Set $y = f(x)$, $z = g(y) = g(f(x))$.
The Chain Rule of Calculus claims $\frac{\partial z}{\partial x_i} = \sum_{j=1}^{m}\frac{\partial z}{\partial y_j}\frac{\partial y_j}{\partial x_i}$.

### Example 
Suppose we have $x = f(w)$,  $y = f(x)$ and $z = h(y)$ and want to compute $\frac{\partial z}{\partial w}$. 


$\rightarrow$ Then using the chain rule of calculus we get $\frac{\partial z}{\partial w} = \frac{\partial z}{\partial y} \frac{\partial y}{\partial x} \frac{\partial x}{\partial w}$.

### Forward Propagation in Fully Connected MLP's
The information first flows forward through the Network. For input $x$ the network outputs an estimate $\hat{y}$ of $y$ and the cost $J(\hat{y}, y, \theta)$ is computed.

**Input:** Network with depth $l$, $(x,y)$, $W^{(i)}, b^{(i)}$ for $i = 1,\ldots, l$
Set $h^{(0)} := x$
**For** $i = 1, \ldots, l$
- $a^{(i)} \leftarrow W^{(i)}h^{(i-1)} + b^{(i)}$
- $h^{(i)} \leftarrow f(a^{(i)})$

$\hat{y} \leftarrow h^{(l)}$
$J \leftarrow L(\hat{y}, y) + \lambda\Omega(\theta)$

### Back-Propagation in Fully Connected MLP's
We  now backpropagate the information through the network and recursively compute the gradient of the cost function $J$ with respect to $b_{i}, W_{i}$ in the i-th layer. We start at the last layer and proceed backwards through the network until we reach the first layer.

$g \leftarrow \nabla_{\hat{y}}J = \nabla_{\hat{y}}L(\hat{y},y)$
**For** $i = l, \ldots, 1$
- $g \leftarrow \nabla_{a^{(i)}}J = f^\prime(a^{(i)})^\intercal g$
- $\nabla_{b^{(i)}}J \leftarrow g \lambda\nabla_{b^{(i)}}\Omega(\theta)$
- $\nabla_{W^{(i)}}J \leftarrow g{h^{(i-1)}}^\intercal + \lambda\nabla_{W^{(i)}}\Omega(\theta)$
- $g \leftarrow \nabla_{h^{(i-1)}}J = {W^{(i)}}^\intercal g$


## Questions

### Q:  What problems can occur when choosing a too small or too large stepsize $\epsilon$?
A: A too large $\epsilon$ might cause that one might jump over the a (global) minimum without noticing and a too small one results normally in long computations for a small benefit or one can get easily get stuck in a local optima.

### Q: Why is it not common to use higher (or at least second) order Taylor Approximation to compute an as good as possible $\epsilon$?
A: Computing even the second derivative for a second order Taylor approximation is in general computational expensive or even intractable. E.g. computing the Hessian of an Neural net working on an $1000$x$1000$ pixel image one would have to compute a $(1000$x$1000)^2$ dimensional Matrix.

### Q: How to use the Gradient descent approach for constrained optimization?
A: E.g. one could implement one step and compute the next point disregarding the the constrains and then projecting the result onto the nearest possible one which obeys the the constraints and then iterate. Projecting here is needs to be defined in detail ad relies heavily on the constraints given (finding an computational cheap and meaningful method).

### Q: Where are these regularization or constraints usually needed in the scope of Neural Networks (continuation of the previous question)? 
A: Regularization or constraints are needed when the output relays within a certain boundary (upper or lower or both), eg. if the output is a probability.

### Q: Can you explain the method of taking small epsilon steps and projecting it back onto S?
A: Consider the case of contrained optimization. The idea behind that method is to optimize an $\epsilon$ step in the unconstrained problem with an result $r$. Then project $r$ onto the space of constrained solutions resulting in $r'$. This step is not trivial because it remains to solve the problem of finding an appropriate $r'$ given $r$. If possible (computational tractable) take the closest $r'$.
    
### Q: How would you classify new instances after training a neural net as discussed in example 1 (Sigmoid Units for Bernoulli Output Distribution) and example 2 ( Softmax Units for Multinoulli Distribution)? 
A: For the first example we calculate the probability that an input, x will be classified as $1$,  $P(y=1|x)$, eg. Output needs to lie in $[0,1]$. While classifying a new instance, the probability of that instance being a $1$ or $0$ will be checked. \newline
For the second example, the new instance will be generalized to a discrete variable y with n values. i.e. produce a vector $\hat{y}$ with $\hat{y_i} = P(y=i|x)$ 

### Q: How is the Krylov Method used to calculate higher order derivatives? 
A: In order to calculate higher order derivatives such as the Hessian Matrix, the inverse of the matrix might be needed, Krylov method does not invert the matrix but solves the system, it finds the best approximate solution in some subspace of the instance space and continues to make the subspace bigger until it becomes N dimensional and reaches optimal, then it can be solved in N steps. 
    




