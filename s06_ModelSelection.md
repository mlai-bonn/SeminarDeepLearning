
# Model Selection and Validation
## § 1 Introducion

#### Def:
Model Selection is the task of choosing the best algorithm and parameters for a given learning problem

....

## § 2 Validation
Idea: use some of the training data as a "validation set", to estimate the accuracy of the algorithms. This Validation set is not used for training

### § 2.1 (Hold out set and) Validation for Model Selection
#### Theorem 1 (without proof)
**Gross V oder klein v im index??**
Let $h$ be a predictor and $V = (x_1,y_1), \ldots, (x_{m_v}, y_{m_V})$ a validation set, sampled according to $\mathcal{D}$ (and not used during training).
Assume the loss function is in $[0,1]$. Then for every $\delta \in (0,1)$, with probability of at least $1-\delta$ over the choice of V, we have

$$
|L_V(h) - L_{\mathcal{D}}(h)| \leq \sqrt{\frac{\log(2/\delta)}{2m_V}}.
$$

#### Remark
Tight bound! because V is "fresh"
Fundamental Theorem of learning gives similar bound, but with constant factor and VC dim
In practice: use hold out Selectio

#### Theorem 1 (without proof)
**Gross V oder klein v im index??**
Let $\mathcal{H}=\{h_1, \ldots, h_r\}$ be a set of predictors and $V = (x_1,y_1), \ldots, (x_{m_v}, y_{m_V})$ a validation set, sampled according to $\mathcal{D}$ (and not used during training).
Assume the loss function is in $[0,1]$. Then for every $\delta \in (0,1)$, with probability of at least $1-\delta$ over the choice of V, we have

$$
\forall h\in \mathcal{H}: |L_V(h) - L_{\mathcal{D}}(h)| \leq \sqrt{\frac{\log(2|\mathcal{H}|/\delta)}{2m_V}}.
$$
#### Remark
$h_i$ predictors= output of different algorithms (or different parameters)
similar to learning a finite hypothesis class
if $|\mathcal{H}|$ is not to large, this theorem implies, that validation sets can be used to approximate the true error. otherwise: danger of overfitting

### § 2.2 Model Selection cure
Plot training and validation error against complexity of Model
e.g. polynomial example

--graphs---

### § 2.3 Cross Validation
**omitted during presentation**

## § 3 What to do if learning fails
- Get a larger sample
- Change hypothesis class by
  - enlarging it
  - reducing it
  - completely change it
  - Change the considered parameters
- Change the feature representation of the data
- Change the optimization algorithm used to apply the learning rule

### § 3.1 Error decomposition
Let $h_s$ be the hypothesis learned on set $S$ and define $h^* = \underset{h\in \mathcal{H}}{\text{argmin}}L_{\mathcal{D}}(h)$.

$$
L_{\mathcal{D}}(h_S) =
( L_{\mathcal{D}}(h_S) - L_{\mathcal{D}}(h^* )) + L_{\mathcal{D}}(h^* )
$$

$$
L_{\mathcal{D}}(h_S) =
( L_{\mathcal{D}}(h_S) - L_{V}(h_S )) + ( L_{V}(h_S )- L_{S}(h_S )) + L_{S}(h_S )
$$

$$
L_{S}(h_S) =
( L_{S}(h_S )) - L_{S}(h^* )) + (L_{S}(h^* ) - L_{\mathcal{D}}(h^* )) + L_{\mathcal{D}}(h^* )
$$

If $L_{S}(h_S)$ is large $\Rightarrow$ $L_{\mathcal{D}}(h^* )$ = approximation error is large
$\rightsquigarrow$ change hypothesis class, enlarge it.
Remark: $\nLeftarrow$

### § 3.2 Learning curves
Assume $L_S(h_S)$ is 0 in two different scenarios.

|Scenario 1 | Scenario 2|
|-|-|
|$m < VC_{\dim}$ of learner, high approximation error (=bad model) | $m> 2VC_{\dim}$ of learner, approximation error is 0 (= good model)|
|graphics|graphics|
|model bad change it or features|model good enough. more samples or simpler, less complex model|

for $m \to \infty$, validation and training error converge to approximation error.

### § 3.3 Summary
||Learning does not work $\\ \downarrow$|||
|-|-|-|-|
||Use Model Selection curve.$\\$ Check if parameters are ok $\\ \downarrow$|||
||training error: $\\$ large $\swarrow \quad \quad \searrow$ small |||
|Change or enlarge hypothesis class $\\$ or change feature representation||Plot learning curve! $\\$ Check approximation error $\\$ large $\swarrow \quad \quad \searrow$ small||
||Change hypothesis class or feature representation!||more data or alternatively $\\$ reduce complexity of hypothesis class|
