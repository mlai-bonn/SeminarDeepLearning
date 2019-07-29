## KERNEL METHODS

### §1. Embeddings into Feature Space

This is Commit 2. 

Not all labeled sets can be separated with a hyperplane. Consider the set 

$$ X = \{ -10, -9, .., 0, ...,9, 10 \} \subseteq \mathbb{R}$$

together with the labels $l(x) = +1 $ if $\rvert x \lvert > 2$ and $l(x) = -1$ otherwise for $x \in X$. Now $X$ cannot be separated with a hyperplane. However if we consider the map $\psi:\mathbb{R} \rightarrow \mathbb{R}^2, x \mapsto (x,x^2)$ then the image $\psi(X)$ without label changes can be separated using a hyperplane since $\psi$ aligns $X$ onto a parabola. This hyperplane is given by

$$ h(x) = \mathrm{sign}((w,\psi(x))-b) $$

where $w = (0,1)^T$ and $b=5$. The space $\psi(X)$ is called *feature space* and the map $\psi$ is called *feature map*. The general procedure for feature space embeddings is as follows.

1. For a given domain $X$ choose a map $\psi:X \rightarrow \mathcal{F}$ into some Hilber space $\mathcal{F}$, called *feature space*.
2. For $m$ labeled examples $S = \{(x_1, y_1), ...,(x_m,y_m) \}$ consider the embedded examples $\hat{S} =\{ (\psi(x_1), y_1), ..., (\psi(x_m), y_m) \}$.
3. Train a predictor $h$ on $\hat{S}$.
4. The classified label of a new point $x$ is given by $h(\psi(x))$.

Observe that for a given probability distribution $\mathcal{D}$ on $X \times Y$ we can define a probability distribution $\mathcal{D}^{\psi}$ on $\mathcal{F} \times Y$ by letting $\mathcal{D}^{\psi}(A) = \mathcal{D}(\psi^{-1}(A))$ for subsets $A \subseteq \mathcal{F} \times Y$. Hence for every predictor $h$ 

$$ L_{\mathcal{D}^{\psi}}(h) = L_{\mathcal{D}}(h \circ \psi). $$

In order to understand the expressive power of feature space embeddings we consider the case degree $k$ multivariate polynomials $p: \mathbb{R}^n \rightarrow \mathbb{R}$. They can be expressed as 

$$ p(x) = \sum_{ J \in [n]^r, \: r \leq k}  w_J \prod_{i=1}^r x_{J_i}$$ 

with coefficients $w_J \in \mathbb{R}$. We can write $p$ as a hyperplane $p(x) = (\psi(x),w)$ in some feature space $\mathbb{R}^d$. In order to do this we consider the feature map

$$ \psi : \mathbb{R}^n \rightarrow \mathbb{R}^d, \: \: x \mapsto \Big(\prod_{i \leq r} x_{J_i}\Big)_{J \in [n]^r, \: r \leq k}$$

and put $w = (w_J)_{J \in [n]^r, \: r \leq k}$.

### §2. The Kernel Trick

A *kernel* on a domain $X$ is a map $K: X \times X \rightarrow \mathbb{R}$ such that there exists a feature map $\psi : X \rightarrow \mathcal{F}$ into some Hilbert space $\mathcal{F}$ such that for all $x,y \in X$

$$ K(x,y)=(\psi(x), \psi(y))_{\mathcal{F}}$$ 

where $(-,-)_{\mathcal{F}}$ is the scalar product on $\mathcal{F}$.

**Theorem** (Representer Theorem)**:** Let $\psi :X \rightarrow \mathcal{F}$ be a feature map into a Hilbert space $\mathcal{F}$. Consider the optimization problem for $x_1, ..., x_m \in X $

$$ \underset{w \in \mathcal{F}}{\mathrm{min}}\Big[ f((w, \psi(x_1))_{\mathcal{F}}, ..., (w, \psi(x_m))_{\mathcal{F}}) + R(||w||) \Big]$$ 

where $f:\mathbb{R}^m \rightarrow \mathbb{R}$ is an arbitrary function and $$ R : \mathbb{R}_{\geq 0} \rightarrow  \mathbb{R}$$ is monotonically decreasing. Then there exists a vector $$\alpha \in \mathbb{R}^m$$ such that $$w^* = \sum_{i=1}^m \alpha_i \psi(x_i)$$ is an optimal solution.

**Proof:** Let $$w^*$$ be an optimal solution to the optimization problem. Since $$w^* \in \mathcal{F}$$ and $$\mathcal{F}$$ is a Hilbert space we can write 

$$ w^* = \sum_{i=1}^m \alpha_i \psi(x_i) + u$$ 

with $\alpha_i \in \mathbb{R}$ and $(u, \psi(x_i))_{\mathcal{F}}=0$ for all $i\leq m$. Let $w = w^* - u$. Using the polarization identities we find $||w^*||^2 = ||w||^2 + ||u||^2$ and hence $||w|| \leq ||w^*||$. By monotonicity of $R$ is follows $R(||w||) \leq R(||w^*||)$. Further we have 

$$ (w, \psi(x_i))_{\mathcal{F}} = (w^*-u, \psi(x_i))_{\mathcal{F}}=(w^*, \psi(x_i))_{\mathcal{F}}$$ 

for all $i \leq m$. It follows that for our minimization problem 

$$ f((w, \psi(x_1))_{\mathcal{F}}, ..., (w, \psi(x_m))_{\mathcal{F}}) + R(||w||) $$

$$ \leq f((w^*, \psi(x_1))_{\mathcal{F}}, ..., (w^*, \psi(x_m))_{\mathcal{F}}) + R(||w^*||) $$

which means that $w$ is an optimal solution. $\square$

Assume that $\psi$ is the feature map of a kernel $K$. We can rewrite the optimization problem in terms of kernel evaluations only. This is called the *kernel trick*. The Representer Theorem tells us that the optimal solution $w^*$ lies in the span 

$$ w^* \in \mathrm{lin}_{\mathbb{R}}\{ \psi(x_1), ..., \psi(x_m) \}.$$ 

This means that we can write $w = \sum_{j=1}^m \alpha_j \psi(x_j)$ and optimize over all $\alpha_j \in \mathbb{R}$ instead. We then have for all $i \leq m$

$$ (w, \psi(x_i))_{\mathcal{F}} = \Big( \sum_j \alpha_j \psi(x_j), \psi(x_i) \Big)_{\mathcal{F}} = \sum_j \alpha_j(\psi(x_j), \psi(x_i))_{\mathcal{F}} = \sum_j \alpha_j K(x_j, x_i).$$

In a similar way we find

$$ ||w||^2 = (w,w)_{\mathcal{F}} = \Big(\sum_i \alpha_i \psi(x_i), \sum_j \alpha_j \psi(x_j) \Big)_{\mathcal{F}} = \sum_{i,j} \alpha_i \alpha_j K(x_i, x_j).$$

All together the optimization problem in question can be rewritten as

$$ \underset{\alpha \in \mathbb{R}^m}{\mathrm{min}}\Big[f\Big( \sum_j \alpha_j K(x_j, x_1), ..., \sum_j \alpha_j K(x_j, x_m) \Big)  + R\Big( \sqrt{\sum_{i,j} \alpha_i \alpha_j K(x_i, x_j)} \Big) \Big] $$ 

### §3. Implementing Soft-SVM with Kernels

Let us consider the Soft-SVM optimization problem in the feature space $\mathcal{F}$

$$ \underset{w \in \mathcal{F}}{\mathrm{min}} \Bigg[  \frac{\lambda}{2}||w||^2 + \frac{1}{m} \sum_{i=1}^m \mathrm{max}\{0, 1-y_i(w, \psi(x_i))_{\mathcal{F}} \} \Bigg]$$

where $\lambda > 0$ is the margin and $y_i \in \{-1, +1\}$ is the label asssociated to observation $x_i$. Observe that for each iteration $w^t$ of SGD we have $ w^t \in \mathrm{lin}_{\mathbb{R}}\{ \psi(x_1), ..., \psi(x_m) \}$. Hence we can maintain the corresponding coefficients $\alpha_i^t$, $i \leq m$ instead. We write $K$ for the kernel induced by $\psi$. The Kernel Soft-SVM algorithm takes the following form.

**Definition (SGD for Solving Soft-SVM with Kernels):**
Input: $T \in \mathbb{N}$, $\lambda > 0$
* Initialize $\beta^1 =0$
* For $t = 1...T$ 
    * $\alpha^t = \frac{1}{\lambda t} \beta^t$
    * Choose $i \in [m]$ uniformly at random
    * For all $j \neq i$
        * $\beta_j ^{t+1}=\beta_j ^{t}$
    * If $y_i \sum_{j=1}^m \alpha_j ^t K(x_j, x_i) < 1$
        * $\beta_i ^{t+1}=\beta_i ^{t} + y_i$
    * Else
        * $\beta_i ^{t+1}=\beta_i ^{t}$
* Return $\overline{w} = \sum_{j=1}^m \overline{\alpha}_j\psi(x_j)$ where $\overline{\alpha} = \frac{1}{T}\sum_{t=1}^T \alpha^t$.

The following lemma tells us that this algorithm is equivalent to running SGD in the feature space.

**Lemma:** Let $\hat{w}$ be the output of SGD in the feature space. Let $\overline{w}=\sum_{j=1}^m \overline{\alpha}_j \psi(x_j)$ be the output of the above algorithm. Then $\hat{w}=\overline{w}$.

**Proof:** It is enough to show that for all $t\leq T$

$$ \theta^t = \sum_j \beta_j ^t \psi(x_j)$$

where $\theta^t$ is the result of running SGD in the feature space. By definition $\alpha ^t = \frac{1}{\lambda t}\beta^t$ and $w^t = \frac{1}{\lambda t} \theta^t$. It follows that 

$$ w^t = \sum_{j=1}^m \alpha_j ^t \psi(x_j).$$

To show that $ \theta^t = \sum_j \beta_j ^t \psi(x_j)$ we use induction on $t$. For $t=1$ the claim is obviously true. Assume the claim is true for $t \geq 1$ then 

$$y_i(w^t, \psi(x_i))_{\mathcal{F}} = y_i \Big(\sum_j \alpha_j ^t \psi(x_j), \psi(x_i) \Big)_{\mathcal{F}} = y_i \sum_{j=1}^m \alpha_j ^t K(x_j, x_i)$$
 
which means that the conditions checked in both algorithms are the same. For the update rule of $\theta^t$ we have

$$ \theta ^{t+1} = \theta ^t + y_i \psi(x_i) = \sum_{j=1}^m \beta_j ^t \psi(x_j) + y_i \psi(x_i) = \sum_{j=1}^m \beta_j ^{t+1} \psi(x_j)$$ 

which means that the claim is true for $t+1$. This concludes the proof. $\square$