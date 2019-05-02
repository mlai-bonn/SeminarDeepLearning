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

Let $\mathcal{X}$ be the *domain set* or *universe*, e.g. the set of all cookies. 
The domain set is often represented as $\mathcal{X} \subseteq \mathbb{R}^n$. 
Cookies, e.g. could be described by two dimensional real vectors where the first dimension measures a particular cookie's crunchyness, while the second dimension measures its chocolate content.

Let $\mathcal{Y}$ be the set of target labels / possible outcomes, e.g. whether the cookies are tasty, or not.
For the remainder of this section, we shall assume that $\mathcal{Y}$ only contains two values: $\mathcal{Y} = \{ 0,1 \}$.