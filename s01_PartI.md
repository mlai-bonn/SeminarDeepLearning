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

**Given** a training example $S \sim \mathcal{D}^m$, **return** a prediction rule that minimizes the *empirical error* or *empirical error*

$ L\_{S} (A(S)) := \frac{ \\| \\{ i \in [m] : A(S)(x\_i) \neq y\_i\\} \\| }{m} $.