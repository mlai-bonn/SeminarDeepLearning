# Regularization and Stability

## § 1 RLM rule

#### Def 1: Regularized Loss Minimization (RLM)
*Regularized Loss Minimization* is a learning rule in the form of
$\underset{w}{\text{argmin}} (L_S(w) + R(w))$,
with a regularization function $R: \mathbb{R}^d \to \mathbb{R}$ .
The case with $R(w) = \lambda ||w||^2_2$ for $\lambda>0$ is called *Tikhonov regularization*.

## § 2 Stable rules and overfitting
#### Notations

|Symbol|Meaning|
|-|-|
|$U(m)$|uniform distribution over $[m]$|
|$A$|a learning algorithm|
|$S = (z_1, \ldots,z_m)$| training set|
|$A(S)$|output of $A$ after processing $S$|
|$z'$|another training example|
|$S^{(i)}$| $(z_1,\ldots,z_{i-1},z',z_{i+1}, \ldots, z_m)$|

#### Def 2: Stability
Let $\epsilon: \mathbb{N} \to \mathbb{R}$ be monotonically decreasing. A is *on-average-replace-one-stable* with rate $\epsilon(m)$ if for every distribution $\mathcal{D}$

$$\underset{(S,z')\sim\mathcal{D}^{m+1}, i\in U(m)}{\mathbb{E}}\left[l(A(S^{(i)}), z_i) - l(A(S),z_i)\right] \leq \epsilon(m) $$

holds.

#### Theorem 1
Let $\mathcal{D}$ be a distribution. (Let $S = (z_1, \ldots,z_m)$, $z'$ be i.i.d examples). Then for any learning algorithm $A$:

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) - L_S(A(S))\right]
= \underset{(s,z')\sim\mathcal{D}^{m+1}, i\in U(m)}{\mathbb{E}}\left[l(A(S^{(i)}), z_i) - l(A(S),z_i)\right].$$

##### Proof
For all $i\in [m]$ we have:

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S) )\right] = \underset{(S,z')\sim\mathcal{D}^{m+1} }{\mathbb{E}}\left[l(A(S), z') \right] = \underset{(S,z')\sim\mathcal{D}^{m+1} }{\mathbb{E}}\left[l(A(S^{(i)}), z_i) \right] $$

Also

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{S}(A(S) )\right] =
\underset{(s,z')\sim\mathcal{D}^{m+1}, i\in U(m)}{\mathbb{E}}\left[l(A(S),z_i)\right].$$

$\square$

##### Remark
$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) - L_S(A(S))\right]$
is a measurement of overfitting, so "stable rules do not overfit".

## § 3 Strong Convexivity
#### Def 3
A function $f$ *(from where to where?)* is *$\lambda$-strongly convex*, if for all $w, u, \alpha\in(0,1)$ we have

$$f(\alpha w + (1-\alpha)u) \leq f(w) + (1-\alpha) f(w) - \frac{\lambda}{2}\alpha(1-\alpha)||w-u||^2$$

#### Lemma 1
1. $f(w)=\lambda\vert\vert w\vert\vert^2$ is $2\lambda$ strongly convex.
2. If $f$ is $\lambda$ strongly convex and $g$ is convex then $f+g$ is $\lambda$ strongly convex.
3. If $f$ is $\lambda$ strongly convex and $u$ is a minimizer of $f$, then for any $w$:

$$f(w) -f(u) \leq \frac{\lambda}{2}||w-u||^2.$$

##### Proof
1 and 2 are easy to check, so only a proof of 3 is provided here:
First we divide the definition of strong convexivity by $\alpha$ and rearrange to get the following:

$$\frac{f(u +\alpha(w-u))-f(u)}{\alpha} \leq f(w) - f(u) - \frac{\lambda}{2}(1-\alpha)||w-u||^2$$

Now let $g(\alpha)=f(u+\alpha(w-u))$ and take the limit $\alpha \to 0$.

$$0 = g'(0) \leq f(w) -f(u) - \frac{\lambda}{2}||w-u||^2$$

$\square$

## § 4 Tikhonov Regularization as a Stabililizer
**Assumption:** Loss function is convex.

**Goal:** We want to bound $\vert A(S^{(i)})-A(S)\vert$ for Tikhonov regularization.

We define $f_S(w) = L_S(w) + \lambda\vert\vert w \vert\vert^2, A(S)=\underset{w}{\text{argmin }} f_S(w)$.

By Lemma 1.2, $f_S$ is $2\lambda$ strongly convex. Now for any $v$ we have

$$f_S(v) - f_S(A(S)) \geq \lambda ||v-A(S)||^2 \tag{1}$$

Also for any $u, v$ and $i$, we have

$$f_S(v)- f_S(u) = L_S(v) + \lambda||v||^2 -(L_S(u) + \lambda||u||^2)= L_{S^{(i)}}(v) + \lambda||v||^2 -(L_{S^{(i)}}(u) + \lambda||u||^2) +  \\ \frac{l(v,z_i) - l(u,z_i)}{m} + \frac{l(u,z') - l(v,z')}{m}$$

remark: one term leq 0

For $v=A(S^{(i)}), u=A(S)$, we obtain (because $v$ is a minimizer)

$$ f_S(A(S^{(i)})) - f(A(S)) \leq \frac{l(A(S^{(i)}),z_i) - l(A(S),z_i)}{m} + \frac{l(A(S),z') - l(A(S^{(i)}),z')}{m} $$

by (1) it follows, that:

$$\lambda ||A(S^{(i)})-A(S)||^2  \leq \frac{l(A(S^{(i)}),z_i) - l(A(S),z_i)}{m} + \frac{l(A(S),z') - l(A(S^{(i)}),z')}{m} \tag{2}$$


**Special Case**
#### Theorem 2
Assume a convex, $\rho$-lipschitz loss function. Then the RLM rule with $\lambda||w||^2$ regularization is on-average-replace-one-stable with rate $\frac{2\rho^2}{\lambda m}$. This also implies by (Theorem 1) that

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) - L_S(A(S))\right] \leq\frac{2\rho^2}{\lambda m}$$

##### Proof
Let $l(-,z_i)$ be $\rho$-lipschitz. Then by definiton

$$l(A(S^{(i)}),z_i) - l(A(S),z_i) \leq \rho ||A(S^{(i)})-A(S)|| \tag{3}
\\ l(A(S),z') - l(A(S^{(i)}),z')\leq \rho ||A(S^{(i)})-A(S)||$$

Plug this in (2):

$$\lambda ||A(S^{(i)})-A(S)||^2\leq 2\rho\frac{||A(S^{(i)})-A(S)||}{m}$$

$$||A(S^{(i)})-A(S)||^2\leq \frac{2\rho}{\lambda m}$$

inserting in (3) yields:

$$l(A(S^{(i)}),z_i) - l(A(S),z_i) \leq \frac{2\rho^2}{\lambda m}$$

This holds for any $S, z', i$.

$\square$

## § 5 Controlling the Fitting Stabilty Tradeoff
$\lambda$ large -> empirical risk increases, stability term will decrease.

#### Theorem 3
Assumptions as in Theorem 2. Then:

$$\forall w^* :  
\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) \right]
\leq L_{\mathcal{D}}(w^* ) + \lambda ||w^* ||^2 +\frac{2\rho^2}{\lambda m}$$

Remark: Oracle inequality. We may think of $w^{* }$ as hypothesis with low risk. $A(S)$ will be only slightly worse (than the rlm term) (depending on $\lambda$).


##### Proof
$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) \right]
= \underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{S}(A(S)) \right]
+\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) - L_{S}(A(S)) \right] \tag{4}$$

We have $L_{S}(A(S)) \leq L_{S}(A(S)) + \lambda||A(S)||^2 \leq L_{S}(w^{* }) + \lambda||w^{* }||^2$.
($A(S)$ is argmin)
Taking expectations and using $\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{S}(w^{* }) \right] =L_{\mathcal{D}}(w^* )$ yields:

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_S(A(S)) \right]
\leq L_{\mathcal{D}}(w^* ) + \lambda ||w^* ||^2 $$

using (4) we get

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) \right]
\leq L_{\mathcal{D}}(w^* ) + \lambda ||w^* ||^2 +
 \underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) - L_{S}(A(S)) \right]$$

applying Theorem 2 now gives us the desired result.

$\square$

#### Corollary 1
(For a convex-lipschitz-bounded learning problem (bounded, i.e.: $w\leq B$ for $w \in \mathcal{H}$)
with parameters $\rho, B$ and $\lambda := \sqrt{\frac{2\rho^2}{B^2m}}$ and Tikhonov regularization.)

Same assumpions as Theorem 2 + $w\leq B$ for all $w \in \mathcal{H}$.
Then for $\lambda := \sqrt{\frac{2\rho^2}{B^2m}}$, we get

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) \right]
\leq \underset{w \in \mathcal{H}}{\min} L_{\mathcal{D}}(w) + \rho B \sqrt{\frac{8}{m}}$$

##### Proof
The Corollary follows directly by setting $w^{* }$ to $\underset{w \in \mathcal{H}}{\text{argmin }} L_{\mathcal{D}}(w)$, inserting $\lambda$ in Theorem 3, and using $w^{* }\leq B$.

**TODO** insert the epsilon variant of the Corollary to!!

$\square$

## § 6 APAC learnability
Convex-lipschitz-bound problems are APAC learnable, as Lemma 2 will show:

#### Lemma 2
If an algorithm A guarantees that for $m \geq m_{\mathcal{H}}(\epsilon)$ and every distribution $\mathcal{D}$ holds:

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) \right]
\leq \underset{h \in \mathcal{H}}{\min} L_{\mathcal{D}}(h) + \epsilon$$

then the problem is APAC learnable by $A$.

##### Proof
Let $\delta \in (0,1)$, $m\geq m_{\mathcal{H}}(\epsilon \delta)$
Define $X=L_{\mathcal{D}}(A(S)) - \underset{h \in \mathcal{H}}{\min} L_{\mathcal{D}}(h)$.
Then $X\geq0$ and $\mathbb{E}[X] \geq \epsilon \delta$ (by assumption).
By Markov, we have:

$$\mathbb{P}\left[L_{\mathcal{D}}(A(S)) \geq\underset{h \in \mathcal{H}}{\min} L_{\mathcal{D}}(h) + \epsilon \right] = \mathbb{P}[X\geq\epsilon] \leq \frac{\mathbb{E}[X]}{\epsilon} \leq \frac{\epsilon \delta}{\delta} = \delta$$

$\square$

## § 7 Ridge Regression
#### Def 4
*Ridge Regression* is the following learning rule:

$$\underset{w \in \mathbb{R}^d}{\text{argmin }}\left( \lambda||w||^2 +\frac{1}{m} \sum_{i=1}^{m}{\frac{1}{2}(\left<w,x_i\right> -y_i)^2} \right)$$

with squared loss

#### Remark
$l(w,z') = {\frac{1}{2}(\left<w,x'\right> -y')^2}$ is not lipschitz!
but we have $\nabla l(w,z') = \frac{1}{2}z'z'^{T}w - y'z'$, which is $\beta$-lipschitz (for some value of $\beta$).
In this case, we call $l$ *$\beta$-smooth*.

#### Remark
For $\beta$-smooth funcions, very similar results to the previously stated Theorems and Corollarys for lipschitz functions hold. Especially we get (without proof)

#### Theorem 4
For a distribution $\mathcal{D}$ over $\chi \times [0,1]$ with $\chi = \{ x\in \mathbb{R}^d | x \leq 1\}$, $\mathcal{H}=\{w\in \mathbb{R}^d: ||w|| \leq B\}$, $\epsilon\in(0,1)$, $m \geq \frac{150 B^2}{\epsilon^2}$, $\lambda = \frac{\epsilon}{3B^2}$,
ridge Regression algorithm satisfies:

$$\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}\left[L_{\mathcal{D}}(A(S)) \right]
\leq \underset{w \in \mathcal{H}}{\min} L_{\mathcal{D}}(w) + \epsilon$$

This implies that there is an APAC learner for Ridge Regression.
