## STOCHASTIC GRADIENT DESCENT

### §1. Classical Gradient Descent

This is Commit 12. 

In Classical Gradient Descent we want to minimize $f \in \mathrm{C}^1(S, \mathbb{R})$ where $S \subseteq \mathbb{R}^n$ is open. The idea is to iteratively descent into the negative gradient direction of $f$ with small stepsize. 

**Definition (Gradient Descent):**
Input: $\eta > 0$, $T \in \mathbb{N}$, $f:S \rightarrow \mathbb{R}$
* Initialize $w^1 \in S$
* For $t = 1...T$ 
    * $w^{t+1} = w^t - \eta \nabla f(w^t)$
* Return $\overline{w} = \frac{1}{T}\sum_{t\leq T} w^t$.

Other options: Return $w^T$ or best performing $w^{t_0}$ such that $f(w^{t_0}) \leq f(w^t)$ for all $t \leq T$.

### §2. Subgradients

In this chapter we generalize Gradient Descent to non-differentiable functions. 

**Recall:** If $f:S\rightarrow \mathbb{R}$ is convex then for any point $w \in S$ if $f$ is differentiable in $w$ then 

$$ \forall u \in S: \quad f(u) \geq f(w) + (u-w, \nabla f(w)) $$

where $$ H_{f,w}= \{ f(w) + (u-w, \nabla f(w)) \mid u \in S\} $$ is the tangent hyperplane of $f$ at $w$.

**Lemma:** Let $S\subseteq \mathbb{R}^n$ be convex and open. Then a function $f:S \rightarrow \mathbb{R}$ is convex if and only if 

$$\forall w \in S \exists v \in  \mathbb{R}^n \forall u \in S: \quad f(u) \geq f(w) + (u-w, v). $$

**Definition:** Let $f:S \rightarrow \mathbb{R}$ be a function. A vector $v \in \mathbb{R}^n$ statisfying 

$$\forall u \in S: \quad f(u) \geq f(w) + (u-w, v)$$

at a given point $w \in S$ is called **subgradient** of $f$ at $w$. The set of all subgradients at $w$ is denoted with $\partial f(w)$.

**Facts:**

* If $f$ is convex then $\partial f(w) \neq \emptyset.$
* If $f$ is convex and differentiable at $w$ then $$\partial f(w) = \{ \nabla f(w) \}.$$

**Example:**

$$\partial |.|(x) = \begin{cases}
\{+1\}: \: x > 0\\
[-1,1]: \: x =0  \\
\{-1\}: \: x < 0.
\end{cases}
 $$

### §3. Lemma 14.1

For later convergence theorems we need the following lemma. 

**Lemma 14.1:** Let $v_1, ..., v_n \in \mathbb{R}^n$. Then any algorithm with initialization $w^1=0$ and update rule $w^{t+1} = w^t - \eta v_t$ for $\eta >0$ statisfies for all $w^* \in \mathbb{R}^n$

$$\sum_{t=1}^T (w^t - w^*, v_t) \leq \frac{\rVert w^* \lVert ^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \rVert v_t \lVert ^2.$$
In particular for every $B, \rho > 0$ if $ \rVert v_1 \lVert, ..., \rVert v_T \lVert \leq \rho $ and $w^* \in \overline{\mathbb{B}}_B(0)$ then with $\eta = \frac{B}{\eta} \frac{1}{\sqrt{T}}$ we have 

$$\frac{1}{T} \sum_{t=1}^T(w^t-w^*, v_t) \leq \frac{B\rho}{\sqrt{T}}.$$

**Proof:** Recall the polarization identities

$$(u,v) = \frac{1}{2}(||u||^2 + ||v||^2 - ||u-v||^2).$$

Hence 

$$(w^t - w^*, v_t) = \frac{1}{\eta}(w^t-w^*, \eta v_t)$$

$$=\frac{1}{2 \eta}(-||w^t-w^* - \eta v_t||^2 + ||w^t -w^*||^2 + \eta^2||v_t||^2)$$

$$= \frac{1}{2\eta}(||w^t - w^*||^2 - ||w^{t+1} - w^*||^2) + \frac{\eta}{2}||v_t||^2.$$

Summing over $t$ we get 

$$\sum_{t=1}^T (w^t - w^*, v_t) = \frac{1}{2\eta}\sum_{t=1}^T(||w^t-w^*||^2 - ||w^{t+1}-w^*||^2) + \frac{\eta}{2} \sum_{t=1}^T||v_t||^2$$

$$= \frac{1}{2\eta} (||w^1-w^*||^2 - ||w^{T+1} - w^*||^2) + \frac{\eta}{2} \sum_{t=1}^T ||v_t||^2 \leq \frac{||w^*||^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T ||v_t||^2$$

since the second sum is telescopic and $w^1=0$ by assumption. $\square$

### §4. Stochastic Gradient Descent

**Definition (Stochastic Gradient Descent):** 

Input: $\eta > 0$, $T \in \mathbb{N}, f:S \rightarrow \mathbb{R}$
* Initialize $w^1=0$
* For $t=1...T$
    * Draw $v_t$ according to some probability distribution such that $\mathbb{E}[v_t|w^t] \in \partial f(w^t)$
    * $w^{t+1} = w^t - \eta v_t$
* Return $\overline{w} = \frac{1}{T} \sum_{t=1}^T w^t.$

**Theorem:** Let $B, \rho >0$ and $f:S \rightarrow \mathbb{R}$ convex. Let $w^* \in \mathrm{argmin}_{||w|| \leq B} f(w)$ for $S \subseteq \mathbb{R}^n$. Assume that SGD runs with $T$ iterations and stepsize $\eta = \frac{B}{\rho}\frac{1}{\sqrt{T}}$ and assume that $||v_1||, ..., ||v_T|| \leq \rho$ almost sure. Then 

$$\mathbb{E}f(\overline{w})- f(w^*) \leq \frac{B\rho}{\sqrt{T}}.$$

Therefore for given $\epsilon > 0$ to achieve $\mathbb{E}f(\overline{w}) - f(w^*) \leq \epsilon$ one needs to run $T \geq (B\rho / \epsilon)^2$ iterations of SGD.

**Proof:** We use the notation $v_{1:T} = v_1, ..., v_T$. Since $f$ is convex we can apply Jensens inequiality to obtain

$$ f(\overline{w}) - f(w^*) \leq \frac{1}{T}\sum_{t=1}^T(f(w^t) - f(w^*))$$

for the iteratives $w^1, ..., w^T$ of SGD. The expectation preserves this inequality and we obtain 

$$ \underset{v_{1:T}}{\mathbb{E}}[f(\overline{w}) - f(w^*)] \leq \underset{v_{1:T}}{\mathbb{E}}\Big[\frac{1}{T}\sum_{t=1}^T(f(w^t) - f(w^*))\Big]. $$

Using lemma 14.1 we have 

$$ \underset{v_{1:T}}{\mathbb{E}} \Big[ \frac{1}{T} \sum_{t=1}^T(w^t - w^*, v_t) \Big] \leq \frac{B\rho}{\sqrt{T}}. $$

It is therefore enough to show that 

$$ \underset{v_{1:T}}{\mathbb{E}}\Big[\frac{1}{T}\sum_{t=1}^T(f(w^t) - f(w^*))\Big]\leq \underset{v_{1:T}}{\mathbb{E}} \Big[ \frac{1}{T} \sum_{t=1}^T(w^t - w^*, v_t) \Big]. $$ 

Using linearity of expectation we get

$$\underset{v_{1:T}}{\mathbb{E}} \Big[ \frac{1}{T} \sum_{t=1}^T(w^t - w^*, v_t) \Big] = \frac{1}{T} \sum_{t=1}^T \underset{v_{1:T}}{\mathbb{E}}[(w^t-w^*, v_t)]. $$ 

Next recall the law of total expectation: Let $\alpha$, $\beta$ be random variables and $g$ some function then $\mathbb{E}_{\alpha}[g(\alpha)]=\mathbb{E}_{\beta}[\mathbb{E}_{\alpha}[g(\alpha)|\beta]]$. Put $\alpha=v_{1:t}$ and $\beta=v_{1:t-1}$ then

$$ \underset{v_{1:T}}{\mathbb{E}}[(w^t-w^*,v_t)] = \underset{v_{1:t}}{\mathbb{E}}[(w^t-w^*, v_t)] $$

$$ = \underset{v_{1:t-1}}{\mathbb{E}}[\underset{v_{1:t}}{\mathbb{E}}[(w^t-w^*, v_t)|v_{1:t-1}] ] = \underset{v_{1:t-1}}{\mathbb{E}}[(w^t-w^*, \underset{v_{t}}{\mathbb{E}}[v_t|v_{1:t-1}])]. $$

But now $w^t$ is completely determined by $v_{1:t-1}$ since $w^t = -\eta(v_1 + ... + v_{t-1})$. It follows $\underset{v_{t}}{\mathbb{E}} [v_t|v_{t-1}] = \underset{v_{t}}{\mathbb{E}} [v_t|w^t] \in \partial f(w^t)$. Therefore

$$ \underset{v_{1:t-1}}{\mathbb{E}}[(w^t-w^*, \underset{v_{t}}{\mathbb{E}}[v_t|v_{1:t-1}])] \geq \underset{v_{1:t-1}}{\mathbb{E}}[f(w^t)-f(w^*)]$$

which also implies 

$$ \underset{v_{1:T}}{\mathbb{E}}[(w^t-w^*, v_t)] \geq \underset{v_{1:T}}{\mathbb{E}}[f(w^t)-f(w^*)].$$ 

Now summing over $t \leq T$ and dividing by $T$ gives the desired inequality. To achieve

$$ \mathbb{E}f(\overline{w}) - f(w^*) \leq \frac{B \rho}{\sqrt{T}} \leq \epsilon$$ 

we need $T \geq (B \rho / \epsilon)^2$ iterations. $\square$

### §5. Stochastic Gradient Descent for Risk Minimization

In Learning Theory we want to minimize 

$$ \mathcal{L}_{\mathcal{D}}(w) = \mathbb{E}_{z \sim \mathcal{D}}[l(w,z)].$$

SGD allows us to directly minimize $\mathcal{L}_{\mathcal{D}}$. For simplicity we assume that $l(-, z)$ is differentiable for all $z \in Z$. We construct the random direction $v_t$ as follows: Sample $z \sim \mathcal{D}$ and put

$$ v_t = \nabla l(w^t, z)$$ 

where the gradient is taken w.r.t. $w$. Interchanging integration and gradient we get 

$$ \mathbb{E}[v_t|w^t] = \mathbb{E}_z[\nabla l(w^t, z)] = \nabla \mathbb{E}_z[l(w^t, z)] = \nabla \mathcal{L}_{\mathcal{D}}(w) \in \partial\mathcal{L}_{\mathcal{D}}(w).$$ The same argument can be applied to the subgradient case. Let $v_t \in \partial l(w^t, z)$ for a sample $z \sim \mathcal{D}$. Then by definition for all $u$

$$ l(u,z) - l(w^t,z) \geq (u-w^t, v_t)$$

By applying the expectation on both sides of the inequality we get 

$$ \mathcal{L}_{\mathcal{D}}(u) - \mathcal{L}_{\mathcal{D}}(w^t)\geq \mathbb{E}_z[(u-w^t, v_t)| w^t] = (u-w^t, \mathbb{E}_z[v_t|w^t])$$

and therefore $\mathbb{E}[v_t|w^t] \in \partial \mathcal{L}_{\mathcal{D}}(w^t)$. This yields the following special form of SGD.

**Definition (Stochastic Gradient Descent for Risk Minimization):** 

Input: $\eta > 0$, $T \in \mathbb{N}$
* Initialize $w^1=0$
* For $t=1...T$
    * Sample $z \in \mathcal{D}$
    * Pick $v_t \in \partial l(w^t, z)$
    * $w^{t+1} = w^t - \eta v_t$
* Return $\overline{w} = \frac{1}{T} \sum_{t=1}^T w^t.$

Using Theorem 14.8. we get the following corollary. 

**Corollary:** Let $B, \rho > 0$, $\mathcal{L}_{\mathcal{D}}$ convex such that 

$$  ||\partial l(w^1, z)||, ..., ||\partial l(w^T, z)|| \leq \rho $$

almost sure. Let $w^* \in \mathrm{argmin}_{||w|| \leq B}f(w)$. Then to achieve 

$$\mathbb{E}f(\overline{w}) - f(w^*) \leq \epsilon$$ 

for given $\epsilon > 0$ we need to run SGD with stepsize $\eta = \frac{B}{\rho}\frac{1}{\sqrt{T}}$ and $T \geq (B \rho / \epsilon)^2$ iterations.