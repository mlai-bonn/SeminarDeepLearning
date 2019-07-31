# Model Selection and Validation
## § 1 Introduction

#### Definition 1
Model Selection is the task of choosing the best algorithm and parameters for a given learning problem.

#### Example 1
When fitting a polynomial with degree $d$ to a function $f:\mathbb{R}\to\mathbb{R}$, the choice of $d$ has a huge impact on the result.
- If $d$ is to low, the model will not fit the data well, i.e. the empirical risk is high.
- If the degree is to high (e.g. $d>m$ for a training set of size $m$), our model will overfit, i.e. the empirical risk is low and the true risk is high.
- But if an appropriate value of $d$ is chosen, the model will generalize well and have a low empirical and true risk.

## § 2 Validation
**Idea**: Use additional samples according to $\mathcal{D}$ as a *validation set*, to estimate the true risk of our algorithms. Usually in practice, we do not generate *additional* samples, but take an existing set of samples and divide it into a training set $S$ and a validation set $V$, that is not used for training and also called a *hold out set*.

### § 2.1 Validation for Model Selection
#### Theorem 1 (without proof)
Let $h$ be a predictor and $V = (x_1,y_1), \ldots, (x_{m_v}, y_{m_v})$ a validation set, sampled according to $\mathcal{D}$ (and not used during training of $h$).
Assume the loss function is in $[0,1]$. Then for every $\delta \in (0,1)$, with probability of at least $1-\delta$ over the choice of $V$, we have

$$
|L_V(h) - L_{\mathcal{D}}(h)| \leq \sqrt{\frac{\log(2/\delta)}{2m_v}}.
$$

##### Remark
This is a tight bound! The Fundamental Theorem of learning gives similar bound, which is not quite as good.
Theorem 1 bounds the true risk well, because $V$ is a "fresh" set not used in training, so $h$ and $V$ are independent.

#### Theorem 2 (without proof)
Let $\mathcal{H}=\{h_1, \ldots, h_r\}$ be a set of predictors and $V = (x_1,y_1), \ldots, (x_{m_v}, y_{m_v})$ a validation set, sampled according to $\mathcal{D}$ (and not used during training of a predictor).
Assume the loss function is in $[0,1]$. Then for every $\delta \in (0,1)$, with probability of at least $1-\delta$ over the choice of $V$, we have

$$
\forall h\in \mathcal{H}: |L_V(h) - L_{\mathcal{D}}(h)| \leq \sqrt{\frac{\log(2|\mathcal{H}|/\delta)}{2m_v}}.
$$
##### Remark
This is very similar to learning a finite hypothesis class, but here, the predictors $h_i$ are the output of different learning algorithms (or the same algorithm with different parameters).

If $|\mathcal{H}|$ is not to large, this theorem implies, that validation sets can be used to approximate the true error. Otherwise, we risk overfitting.

### § 2.2 Model-Selection curve
The model selection curve is a plot of the training and validation error against the complexity of a model
(e.g. for Example 1, we would plot the training and validation error against the degree of the polynomial.).

This tool makes it possible to see, for which level of complexity overfitting or underfitting occurs. Usually, we choose the complexity, for which the validation error is minimal.

## § 3 What to do if learning fails
When learning fails, we can do one of the following things:
- Get a larger sample
- Change hypothesis class by
  - enlarging it
  - reducing it
  - completely change it
  - Change the considered parameters
- Change the feature representation of the data
- Change the optimization algorithm used to apply the learning rule

But how can we decide what option to choose? To answer this question, we first need to understand the different causes of bad performance by decomposing the error terms.

### § 3.1 Error decomposition
Let $h_s$ be the hypothesis learned on set $S$ and define $h^* = \underset{h\in \mathcal{H}}{\text{argmin}}L_{\mathcal{D}}(h)$. We can decompose the true error as

$$
L_{\mathcal{D}}(h_S) =
( L_{\mathcal{D}}(h_S) - L_{\mathcal{D}}(h^* )) + L_{\mathcal{D}}(h^* ),
$$

where $L_{\mathcal{D}}(h^* )$ is called the *approximation error* and $L_{\mathcal{D}}(h_S) - L_{\mathcal{D}}(h^* )$ is called the *estimation error*. The approximation error is high if our hypothesis class is not capable of fitting the data, so underfitting necessarily occurs. However, we cannot directly measure these to errors.

Another decomposition is more practical:

$$
L_{\mathcal{D}}(h_S) =
( L_{\mathcal{D}}(h_S) - L_{V}(h_S )) + ( L_{V}(h_S )- L_{S}(h_S )) + L_{S}(h_S )
$$

Here, $L_{\mathcal{D}}(h_S) - L_{V}(h_S )$ is tightly bound by Theorem 1 and $L_{V}(h_S )- L_{S}(h_S )$ is a measurement of overfitting. For $L_{S}(h_S )$, we have another decomposing:

$$
L_{S}(h_S) =
( L_{S}(h_S )) - L_{S}(h^* )) + (L_{S}(h^* ) - L_{\mathcal{D}}(h^* )) + L_{\mathcal{D}}(h^* )
$$

By definition of $h^{* }$, we have $L_{S}(h_S )) - L_{S}(h^* ) \leq 0$.
In the term $L_{S}(h^* ) - L_{\mathcal{D}}(h^* )$, $h^{* }$ is independent of $S$, so by Theorem 1, the term is tightly bound as well. Now we know:


If $L_{S}(h_S)$ (the training error) is large, then $L_{\mathcal{D}}(h^* )$ (the approximation error) is large as well. In this case, you would change or enlarge the hypothesis class, or change the feature representation

Remark: The reverse does not hold. If the approximation error is large, the training error might be small.


### § 3.2 Learning curves
We will now consider the case of a small training error.

Assume that $L_S(h_S)=0$ in two different scenarios.

||$m < VC_{\dim}$ of learner, high approximation error | $m> 2VC_{\dim}$ of learner, approximation error is 0|
|-|-|-|
|Learning curve: training error | Constantly 0 | Constantly 0 |
|Learning curve: validation error| High validation error, as the model is only "learning by heart" | The validation error will be high at first, but then start to converge to 0|
|What to do | Change hypothesis class or feature representation | obtain more data or reduce complexity of hypothesis class|

for $m \to \infty$, validation and training error converge to approximation error.

### § 3.3 Summary
||Learning does not work $\\ \downarrow$|||
|-|-|-|-|
||Use Model Selection curve.$\\$ Check if parameters are ok $\\ \downarrow$|||
||training error: $\\$ large $\swarrow \quad \quad \searrow$ small |||
|Change or enlarge hypothesis class $\\$ or change feature representation||Plot learning curve! $\\$ Check approximation error $\\$ large $\swarrow \quad \quad \searrow$ small||
||Change hypothesis class or feature representation!||more data or alternatively $\\$ reduce complexity of hypothesis class|
