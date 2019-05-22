---
title: Part I: An Overview
---


# What is Learning?

An insightful motivation in the first chapter is the following distinction between "good" learning and "bad" learning:

- **Bait Shyness of Rats:** Rats eat *poisoned* food, get sick, and afterwards avoid the same kind of food.
- **Pidgeon Superstition:** Pidgeons in a cage get food *in regular intervals of time* and subsequently repeat the things they were doing the first time the food appeared.

What is the difference that makes us consider the rats as "good" learners, and the pidgeons as "bad", superstitious learners?
One of the main reasons is our (prior) knowledge on what is going on behind the scenes of the situations:
We *know* that the food is *poisoned*, hence avoiding it is actually smart; on the other hands, the actions of the pidgeons do not influence the occurrence of food, hence the actions of the pidgeons are "useless".


# Our Setting

- **Supervised Learning:** We learn from a training set where we get some examples and corresponding target values and then try to come up with a rule how to generate a target value given a new example.
- **Passive Learner:** We have no influence on the environment and cannot e.g. request particular labeled examples.
- **Indifferent Teacher:** There is no particular teacher or adversary that selects the training examples for us.
- **Batch Learning:** We get some training set and some time to come up with a prediction rule. Then new examples need to be labeled using our rule. Our performance on the training set will not be punished.

Let's see how we can formalize this setting..


## Formal Model

#### Assumptions:
Let $\mathcal{X}$ be the *domain set* or *universe*, e.g. the set of all cookies. 
The domain set is often represented as $\mathcal{X} \subseteq \mathbb{R}^n$. 
Cookies, e.g. could be described by two dimensional real vectors where the first dimension measures a particular cookie's crunchiness, while the second dimension measures its chocolate content.

Let $\mathcal{Y}$ be the set of target labels / possible outcomes, e.g. whether the cookies are tasty, or not.
For the remainder of this section, we shall assume that $\mathcal{Y}$ only contains two values: $\mathcal{Y} = \\{ 0,1 \\}$.

There exists a *probability distribution* $\mathcal{D}$ over $\mathcal{X} \times \mathcal{Y}$ that is unknown, but fixed.
From this distribution, we will draw (labeled) training examples as well as (unlabeled) test examples.

There exists a *measure of success* that tells us, how well our predictions do. 
That is, the *error of a predictor* $h$ that computes a value $h(x) \in \mathcal{Y}$ for any $x \in \mathcal{X}$ is defined as 

$ L\_{\mathcal{D}} (h) := \mathbb{P}\_{(x,y) \sim \mathcal{D}} \left( h(x) \neq y \right) := \mathcal{D} \left( \\{ (x,y) : h(x) \neq y \\}  \right) $

#### Input:
A *training sequence* $S = ((x\_1, y\_1), \ldots, (x\_m, y\_m))$ with $x\_i \in \mathcal{X}$ and $y\_i \in \mathcal{Y}$ for all $i \in [m]$.
Each $(x\_i, y\_i)$ is drawn independently and identically from $\mathcal{D}$.
We denote this by $S \sim \mathcal{D}^m$.

*$S$ is all the learner sees. The learner has no access to $\mathcal{D}$, except via $S$.*

#### Output: 
A *prediction rule* $A(S)$ that can, given $x \in \mathcal{X}$, produce a value $A(S)(x) \in \mathcal{Y}$ such that $ L\_{\mathcal{D}}(A(S)) $ is minimized.

*$A(S)$ is not necessarily a function, but we write $A(S): \mathcal{X} \to \mathcal{Y}$, anyway.*

*Note further, that our goal is to minimize the true error over the whole distribution $\mathcal{D}$ given only a finite sample. This task is clearly impossible to solve exactly!*


# Empirical Risk Minimization (ERM):

A first, rather natural choice of learning paradigm is *empirical risk minimization*:

**Given** a training example $S \sim \mathcal{D}^m$, **return** a prediction rule $A(S)$ that minimizes the *empirical error* or *empirical error*

$ L\_{S} (A(S)) := \frac{ \\| \\{ i \in [m] : A(S)(x\_i) \neq y\_i\\} \\| }{m} $.

Just like this, however, the ERM paradigm is very keen to overfit. 
A possible solution would be, to remember every training example and corresponding target values and to return a constant on unseen examples:

$ A(S) := \begin{cases} y\_i & \text{ if } x = x\_i  \text{ for some } i \in [m] \\\\ 1 & \text{ otherwise} \end{cases} $

Intuitively, this is obviously no learning. 
There is no kind of generalization from the training examples to the unseen test examples going on.
**How can we fix this?** 

The answer is to include *prior knowledge* into the learning process. 
Formally, we will require our learning algorithm (here, ERM) to output a prediction rule from a *fixed set of prediction rules selected in advance*. 
We call this set of *candidate* prediction rules *hypothesis class* and will denote it by $\mathcal{H}$.

Then, ERM becomes the following:

**Given** a training example $S \sim \mathcal{D}^m$, and some finite representation of a hypothesis class $\mathcal{H} **return** a prediction rule $A(S) \in \operatorname{argmin}\_{h \in \mathcal{H}} L\_{S} (h) \subseteq \mathcal{H}$.

Intuitively, a more restricted hypothesis class $\mathcal{H}$ is a better safeguard against *overfitting*, but results in a stronger *inductive bias* or *underfitting*.

A hypothesis class $\mathcal{H}$ is called *realizable* if there exists $h \in \mathcal{H}$ with $L\_{\mathcal{D}} (h) = 0$.
This is a desirable property which will rarely be the case in practical learning scenarios.


# Probably Approximately Correct (PAC) and Agnostic Probably Approximately Correct (APAC) Learning

The introduction of a hypothesis class $\mathcal{H}$ allows us to ask "Which hypothesis classes are learnable?".
That is, can we come up with a learning algorithm that minimizes $L\_{\mathcal{D}} (A(S)) $ for any distribution $\mathcal{D}$ and any $S \sim \mathcal{D}^m$.
Again, in this exact formulation, we can be sure to fail, but there is hope if we allow a certain error bound (i.e., the performance of our learning rule being worse than the best hypothesis in $\mathcal{H}$) and a confidence parameter (i.e., we allow the algorithm to fail miserably in certain fraction of cases).

**Definition** (APAC):
A hypothesis class $\mathcal{H}$ of functions from $\mathcal{X}$ to $\mathcal{Y}$ is PAC learnable if there exists
- a function $m\_{\mathcal{H}}: (0,1)^2 \to \mathbb{N}$
- a learning algorithm $A$ that
  $\forall \epsilon, \delta \in (0,1)$, 
  $\forall$ distribution $\mathcal{D}$ over $\mathcal{X} \times \mathcal{Y}$
  when running $A$ on $m \geq m\_{\mathcal{H}}(\epsilon, \delta)$ i.i.d. examples from $\mathcal{D}$ then the algorithm returns $A(S) \in \mathcal{H}$ s.t. with probability $1-\delta$ (over the choice of the examples) 

  $L\_{\mathcal{D}}(A(S)) \leq \min\_{h' \in \mathcal{H}} L\_{\mathcal{D}}(h) + \epsilon$
  

**Corrollary**: 
Every *realizable* finite hypothesis class $\mathcal{H}$ is APAC learnable with sample complexity $m\_{\mathcal{H}}(\epsilon, \delta) \leq \left\lceil \frac{\log(|\mathcal{H}| / \delta)}{\epsilon} \right\rceil$ using ERM.

**Corrollary**: 
Every finite hypothesis class $\mathcal{H}$ is APAC learnable with sample complexity $m\_{\mathcal{H}}(\epsilon, \delta) \leq \left\lceil \frac{2 \log(2|\mathcal{H}| / \delta)}{\epsilon^2} \right\rceil$ using ERM.

Note that we are paying a price in terms of sample complexity if we want to learn using a hypothesis class that is not realizable.


# No Free Lunch

**Theorem**: (No Free Lunch)
Let $\mathcal{H}$ be some subset of $\\{ f | f:\mathcal{X} \to \\{0,1\\} \\}$ and let $A$ be any learning algorithm for binary classification with respect to $0-1$-loss over $\mathcal{X}$.
Let $m \leq \frac{\mathcal{X}}{2}$ if $\mathcal{X}$ is finite, and $m$ finite, if $\mathcal{X}$ is infinite.
Then there exists a distribution $\mathcal{D}$ over $\mathcal{X} \times \\{0,1\\}$ such that
- there exists a function $f:\mathcal{X} \to \\{0,1\\}$ with $L\_{\mathcal{D}}(f) = 0
- with probability at least $\frac{1}{7}$ over the choice of $S \sim \mathcal{D}^m$ we have $L\_{\mathcal{D}}(A(S)) \geq \frac{1}{8}$

**Corrollary**:
The class $\mathcal{H} = \\{f | f: \mathcal{X} \to \\{0,1\\} \\}$ is not APAC learnable for infinite $\mathcal{X}$.

The trick in the proof of the no free lunch theorem is to choose a distribution $\mathcal{D}$ for the learner $A$ on which it fails.
This can be done, as there is actually nothing we can possibly know about unseen data without making any assumptions!
However, usually we are saved by some reasonable assumptions that we can make about the data / the world.
Formally, this translates into restricting our hypothesis class $\mathcal{H}$ to a reasonable expressive power to restrict $\argmin\_{h \in \mathcal{H}} L\_{\mathcal{D}}(h) $ to something we can reach.
To this end, we introduce *shattering*

**Definition** (Shattering):
A hypothesis class $\mathcal{H}$ *shatters* a finite set $C \subseteq \mathcal{X}$ if the restriction of $\mathcal{H}$ to $C$ is the set of all functions from $C$ to $\\{0,1\\}$.

**Corrollary**
Let $\mathcal{H} \subseteq \\{ f | f: \mathcal{X}\to\\{0,1\\} \\}$ and suppose there is $C\subseteq\mathcal{X}$ with $|C|=2m$ that is shattered by $\mathcal{H}$. 
Then No Free Lunch! (i.e., the claim stated in the no free lunch theorem holds)

**Definition** (VC-Dimension)
The *VC-Dimension* of a hypothesis class $\mathcal{H} \subseteq \\{ f | f: \mathcal{X}\to\\{0,1\\} \\}$  is the cardinality of the largest set $C\subseteq \mathcal{X}$ that is shattered by $\mathcal{H}$.

**Theorem**
If $\mathcal{H}$ has infinite VC-Dimension then $\mathcal{H}$ is not APAC learnable.

**Theorem** (Fundamental Theorem of APAC Learning)
If $\mathcal{H} \subseteq \\{ f | f: \mathcal{X}\to\\{0,1\\} \\}$ has finite VC-Dimension $d$ then $\mathcal{H}$ is APAC learnable for the $0-1$ loss with 

$ C\_1 \frac{d+\log(1/\delta)}{\epsilon^2} \leq m\_{\mathcal{H}}(\epsilon, \delta) \leq C\_2 \frac{d + \log(1/\delta)}{\epsilon^2}\ .$
