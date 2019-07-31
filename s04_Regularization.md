# Regularization and Stability

## § 0 Overview
First we will define Regularized Loss Minimization and see how stability of learning algorithms and overfitting are connected. Then we are going to proof some general bounds about stability for Tikhonov regularization. To get useful bounds, we have to add further assumptions like a Lipschitz loss function, or a $\beta$-smooth loss function. However, only Lipschitz loss functions are considered here. We will proof that learning problems with convex-Lipschitz-bounded loss function and Tikhonov regularization are APAC learnable. We will also see (without proof) a similar result for Ridge Regression, which has a non-Lipschitz loss function.

## § 1 RLM Rule

#### Definition 1: Regularized Loss Minimization (RLM)
*Regularized Loss Minimization* is a learning rule in the form of
$\underset{w}{\text{argmin}} (L_S(w) + R(w))$,
with a *regularization function* $R: \mathbb{R}^d \to \mathbb{R}$ .
The case with $R(w) = \lambda \lVert w \rVert^2_2$ for $\lambda>0$ is called *Tikhonov regularization*.

## § 2 Stable Rules and Overfitting
#### Notations

|Symbol|Meaning|
|-|-|
|$U(m)$|uniform distribution over $[m]=\{1,\ldots,m\}$|
|$A$|a learning algorithm|
|$S = (z_1, \ldots,z_m)$| training set|
|$A(S)$|output of $A$ after processing $S$|
|$z'$|another training example|
|$S^{(i)}$| $(z_1,\ldots,z_{i-1},z',z_{i+1}, \ldots, z_m)$|

#### Definition 2: Stability
Let $\epsilon: \mathbb{N} \to \mathbb{R}$ be monotonically decreasing. A is *on-average-replace-one-stable* with rate $\epsilon(m)$ if for every distribution $\mathcal{D}$

$$\underset{(S,z')\sim\mathcal{D}^{m+1}, i\in U(m)}{\mathbb{E}}\left[l(A(S^{(i)}), z_i) - l(A(S),z_i)\right] \leq \epsilon(m) $$

holds.

##### Remark
For a reasonable learner, the term $l(A(S^{(i)}), z_i) - l(A(S),z_i)$ will typically be greater than zero, because $z_i$ was used during the learning of $A(S)$, but not while learning $A(S^{(i)})$.

#### Theorem 1
Let $\mathcal{D}$ be a distribution. Let $S = (z_1, \ldots,z_m)$, $z'$ be examples, independent and identically distributed according to $\mathcal{D}$. Then for any learning algorithm $A$:

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) - L_S(A(S))\right]
= \underset{(S,z')\sim\mathcal{D}^{m+1}, i\in U(m)}{\mathbb{E}}\left[l(A(S^{(i)}), z_i) - l(A(S),z_i)\right].$$

##### Proof
For all $i\in [m]$ we have:

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S) )\right] = \underset{(S,z')\sim\mathcal{D}^{m+1} }{\mathbb{E}}\left[l(A(S), z') \right] = \underset{(S,z')\sim\mathcal{D}^{m+1} }{\mathbb{E}}\left[l(A(S^{(i)}), z_i) \right] $$

For the first equation we used the definition of the true error $L_{\mathcal{D}}$ and for the second equation, we swapped the names of the i.i.d variables $z'$ and $z_i$. Using the definition of the empirical error $L_S$ as a weighted sum of terms of the form $l(-,z_i)$, which can be written as an expectation value as well, yields:

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{S}(A(S) )\right] =
\underset{S\sim\mathcal{D}^{m}, i\in U(m)}{\mathbb{E}}\left[l(A(S),z_i)\right].$$

Combining both equations finishes the proof. $\square$

##### Remark
As $\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) - L_S(A(S))\right]$
is a measurement of overfitting, Theorem 1 tells us, simply put, that "stable rules do not overfit".

## § 3 Strong Convexivity
#### Definition 3
A function $f$ is *$\lambda$-strongly convex*, if for all $w, u, \alpha\in(0,1)$ we have

$$f(\alpha w + (1-\alpha)u) \leq \alpha f(w) + (1-\alpha) f(u) - \frac{\lambda}{2}\alpha(1-\alpha) \lVert w-u \rVert^2.$$

#### Lemma 1
1. $f(w)=\lambda\vert\vert w\vert\vert^2$ is $2\lambda$ strongly convex.
2. If $f$ is $\lambda$ strongly convex and $g$ is convex then $f+g$ is $\lambda$ strongly convex.
3. If $f$ is $\lambda$ strongly convex and $u$ is a minimizer of $f$, then for any $w$:

$$f(w) -f(u) \leq \frac{\lambda}{2} \lVert w-u \rVert ^2.$$

##### Proof
1 and 2 are easy to check, so only a proof of 3 is provided here:
First we divide the definition of strong convexity by $\alpha$ and rearrange to get the following:

$$\frac{f(u +\alpha(w-u))-f(u)}{\alpha} \leq f(w) - f(u) - \frac{\lambda}{2}(1-\alpha) \lVert w-u \rVert^2$$

Now let $g(\alpha)=f(u+\alpha(w-u))$ and take the limit $\alpha \to 0$. Using that $u$ is a minimizer, we obtain

$$0 = g'(0) \leq f(w) -f(u) - \frac{\lambda}{2} \lVert w-u \rVert^2.$$

$\square$

## § 4 Tikhonov Regularization as a Stabilizer
From now on, we will assume our loss function to be convex. Our goal will be to bound $\vert A(S^{(i)})-A(S)\vert$ for Tikhonov regularization.

We define $f_S(w) = L_S(w) + \lambda\lVert w \rVert^2$ and $A(S)=\underset{w}{\text{argmin }} f_S(w)$.

By Lemma 1.2, $f_S$ is $2\lambda$-strongly convex. Now for any $v$ we have

$$f_S(v) - f_S(A(S)) \geq \lambda \lVert v-A(S)\rVert^2 \tag{1}$$

Also for any $u, v$ and $i$, we have

$$f_S(v)- f_S(u) = L_S(v) + \lambda\lVert v\rVert^2 -(L_S(u) + \lambda\lVert u \rVert ^2) \\ = L_{S^{(i)}}(v) + \lambda \lVert v \rVert^2 -(L_{S^{(i)}}(u) + \lambda \lVert u \rVert^2) +  \\ \frac{l(v,z_i) - l(u,z_i)}{m} + \frac{l(u,z') - l(v,z')}{m}$$

For $v=A(S^{(i)}), u=A(S)$, $v$ is a minimizer of $f_{S^{(i)}}$, so we obtain

$$f_S(A(S^{(i)})) - f(A(S)) \leq \frac{l(A(S^{(i)}),z_i) - l(A(S),z_i)}{m} + \frac{l(A(S),z') - l(A(S^{(i)}),z')}{m}.$$

By (1) it follows, that

$$\lambda \lVert A(S^{(i)})-A(S)\rVert^2  \leq \frac{l(A(S^{(i)}),z_i) - l(A(S),z_i)}{m} + \frac{l(A(S),z') - l(A(S^{(i)}),z')}{m} \tag{2}.$$

This is a general bound for Tikhonov regularization. To bound this further, we will now assume our loss function to be Lipschitz.


#### Theorem 2
Assume a convex, $\rho$-Lipschitz loss function. Then the RLM rule with $\lambda\lVert w \rVert ^2$ regularization is on-average-replace-one-stable with rate $\frac{2\rho^2}{\lambda m}$. By Theorem 1, this also implies, that

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) - L_S(A(S))\right] \leq\frac{2\rho^2}{\lambda m}.$$

##### Proof
Let $l(-,z_i)$ be $\rho$-Lipschitz. Then by definition

$$l(A(S^{(i)}),z_i) - l(A(S),z_i) \leq \rho \lVert A(S^{(i)})-A(S) \rVert \tag{3}
\\ l(A(S),z') - l(A(S^{(i)}),z')\leq \rho \lVert A(S^{(i)})-A(S) \rVert.$$

Plugging this into (2) gives us

$$\lambda \lVert A(S^{(i)})-A(S)\rVert^2\leq 2\rho\frac{\lVert A(S^{(i)})-A(S) \rVert}{m}$$

$$\Leftrightarrow \lVert A(S^{(i)})-A(S)\rVert\leq \frac{2\rho}{\lambda m}.$$

We now insert this into (3) and finally get

$$l(A(S^{(i)}),z_i) - l(A(S),z_i) \leq \frac{2\rho^2}{\lambda m}.$$

As this holds for any $S, z', i$, taking expectations will conclude the proof. $\square$

## § 5 Controlling the Fitting Stability Tradeoff
The theorem above shows, that the stability term decreases, when $\lambda$ increases. As the empirical risk also increases with $\lambda$, we face a tradeoff between fitting and overfitting. In this section, we will choose a value of $\lambda$ to derive a new bound for the true risk.

#### Theorem 3
Assume a convex, $\rho$-Lipschitz loss function. Then the RLM rule with Tikhonov regularization satisfies:

$$\forall w^* :  
\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) \right]
\leq L_{\mathcal{D}}(w^* ) + \lambda \lVert w^* \rVert^2 +\frac{2\rho^2}{\lambda m}$$

##### Remark
This is bound is also called *oracle inequality* . We may think of $w^{* }$ as hypothesis with low risk. $A(S)$ will then only be slightly worse than $w^{* }$.


##### Proof
$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) \right]
= \underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{S}(A(S)) \right]
+\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) - L_{S}(A(S)) \right] \tag{4}$$

We have $L_{S}(A(S)) \leq L_{S}(A(S)) + \lambda||A(S)||^2 \leq L_{S}(w^{* }) + \lambda\lVert w^{* }\rVert^2$,
as $A(S)$ is a minimizer of $L_S$.
Taking expectations and using $\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{S}(w^{* }) \right] =L_{\mathcal{D}}(w^* )$ yields

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_S(A(S)) \right]
\leq L_{\mathcal{D}}(w^* ) + \lambda \lVert w^* \rVert^2, $$

and using (4) we get

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) \right]
\leq L_{\mathcal{D}}(w^* ) + \lambda \lVert w^* \rVert ^2 +
 \underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) - L_{S}(A(S)) \right].$$

Applying Theorem 2 finishes the proof. $\square$

#### Corollary 1
For a convex-Lipschitz-bounded learning problem $(\mathcal{H},Z,l)$
with parameters $\rho, B$ and $\lambda := \sqrt{\frac{2\rho^2}{B^2m}}$ and Tikhonov regularization, we get

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) \right]
\leq \underset{w \in \mathcal{H}}{\min} L_{\mathcal{D}}(w) + \rho B \sqrt{\frac{8}{m}}.$$

In particular this implies, that for every $\epsilon > 0$ and distribution $\mathcal{D}$, we get

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) \right]
\leq \underset{w \in \mathcal{H}}{\min} L_{\mathcal{D}}(w) +\epsilon,$$

if $m \geq \frac{8\rho^2 B^2}{\epsilon^2}$.

(Note that bounded means: $\lVert w \rVert \leq B$ for all $w \in \mathcal{H}$)
##### Proof
The corollary follows directly by setting $w^{* }$ to $\underset{w \in \mathcal{H}}{\text{argmin }} L_{\mathcal{D}}(w)$, inserting $\lambda$ in Theorem 3, and using $\lVert w^{* } \rVert \leq B$.
$\square$

## § 6 APAC Learnability
Convex-Lipschitz-bound problems are APAC learnable, as Lemma 2 will show:

#### Lemma 2
If an algorithm A guarantees that for $m \geq m_{\mathcal{H}}(\epsilon)$ and every distribution $\mathcal{D}$ holds:

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) \right]
\leq \underset{h \in \mathcal{H}}{\min} L_{\mathcal{D}}(h) + \epsilon$$

then the problem is APAC learnable by $A$.

##### Proof
Let $\delta \in (0,1)$, $m\geq m_{\mathcal{H}}(\epsilon \delta)$.
Define $X=L_{\mathcal{D}}(A(S)) - \underset{h \in \mathcal{H}}{\min} L_{\mathcal{D}}(h)$.
Then $X\geq0$ and by our assumption $\mathbb{E}[X] \geq \epsilon \delta$. Using Markov's inequality, we obtain

$$\mathbb{P}\left[L_{\mathcal{D}}(A(S)) \geq\underset{h \in \mathcal{H}}{\min} L_{\mathcal{D}}(h) + \epsilon \right] = \mathbb{P}[X\geq\epsilon] \leq \frac{\mathbb{E}[X]}{\epsilon} \leq \frac{\epsilon \delta}{\delta} = \delta.$$

$\square$

## § 7 Ridge Regression
#### Definition 4
*Ridge Regression* is the following learning rule with squared loss:

$$\underset{w \in \mathbb{R}^d}{\text{argmin }}\left( \lambda\lVert w\rVert^2 +\frac{1}{m} \sum_{i=1}^{m}{\frac{1}{2}(\left<w,x_i\right> -y_i)^2} \right)$$

#### Remark
This is just the RLM rule with Tikhonov regularization and loss function $l(w,z') = {\frac{1}{2}(\left<w,x'\right> -y')^2}$. But the loss function is not Lipschitz, so we cannot apply the theorems for convex-Lipschitz-bound functions!
However we have, that

$$\nabla l(w,z') = \frac{1}{2}z'z'^{T}w - y'z',$$

is $\beta$-Lipschitz (for some value of $\beta$). Functions with this property, i.e. $\beta$-Lipschitz gradients, are called *$\beta$-smooth*.

#### Remark
For $\beta$-smooth functions, very similar results to the previously stated theorems and corollaries for Lipschitz functions hold. Especially we get:

#### Theorem 4 (without proof)
For a distribution $\mathcal{D}$ over $\chi \times [0,1]$ with $\chi = \{ x\in \mathbb{R}^d | x \leq 1\}$, $\mathcal{H}=\{w\in \mathbb{R}^d: \lVert w \rVert \leq B\}$, $\epsilon\in(0,1)$, $m \geq \frac{150 B^2}{\epsilon^2}$, $\lambda = \frac{\epsilon}{3B^2}$,
the Ridge Regression algorithm satisfies

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) \right]
\leq \underset{w \in \mathcal{H}}{\min} L_{\mathcal{D}}(w) + \epsilon.$$

This implies that there is an APAC learner for Ridge Regression.

## § 8 Questions

Q1: What does Theorem 3 tell us?  
A1: The error is bounded by the regularizer as $m \rightarrow \infty.$  
Q2: Why is $\lambda \lVert .  \rVert ^2$ $2 \lambda$-strongly convex?  
A2: This can be proven using the polarization identities.    Q3: In Corollary 1 why is $w$ bounded in a ball centered at $0$.  
A3: We can change the regularizer and get the same theorem with $w$ being bounded in a ball with different center.  
Q4: Are there practical applications to convergence theorems on Tikhonov regularization? 
A4: Yes, Rigde Regression.  