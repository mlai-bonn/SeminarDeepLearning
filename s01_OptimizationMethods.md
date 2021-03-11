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

## Backpropagation-Algorithms (6.5)

We now want to minimize the cost function $J(\theta)$ using gradient descent. It is thus necessary to compute the gradient $\nabla_{\theta}J(\theta)$. We will do this by letting information flow backwards through the layers of the network. This method is called backpropagation. 

### The Chain Rule of Calculus
Given to functions $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ and $g: \mathbb{R}^m \rightarrow \mathbb{R}$ we want to compute the partial derivatives of their composition.
Set $y = f(x)$, $z = g(y) = g(f(x))$.
The Chain Rule of Calculus claims $\frac{\partial z}{\partial x_i} = \sum_{j=1}^{m}\frac{\partial z}{\partial y_j}\frac{\partial y_j}{\partial x_i}$.



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
    




