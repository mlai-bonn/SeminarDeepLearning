# Neural Networks
A neural network is a computation paradigm based on a layered network of neurons which are capable of simple computations. For a given input, the network will
propagate signals down its layers and will output a result in its last one.
Each neuron does some simple computation with its inputs and outputs its result
to the next layer. A feed-forward network is one without cycles and will be the one that we work with.

A neuron's function can be divided into two steps: A first one calculating a weighted sum of its inputs (the outputs of the previous neurons or the input values if in the first layer) and a second applying an activation function to the result of the weighted sum. This activation function can vary (e.g. sign, threshold, sigmoid, identity).

There are different ways of setting up a network. Different numbers of layers can be used as well as different numbers of neurons in each layer and the activation function of they use. Each of these possible configurations is a different architecture. For each of these architectures, there is a set of parameters that correspond to the input weights of its neurons. The architecture of a network is then a hypothesis class and then each set of weights form a hypothesis in this class.

## Expressive Power -  Boolean Functions
### Claim
for every n there is a network of depth 2 such that the hypothesis class
contains all functions $$\{1,-1\}^n \rightarrow \{1,-1\}$$

Proof: for every vector $$u_1, ..., u_k$$ for which f outputs 1, we add a neuron in the hidden layer that checks if $$x = u_i$$. That is, implements:
$$g(x) = sign(<x,u> - n+1)$$. If any of these neurons is activated the network
outputs a 1, otherwise, 0. $$f(x)$$ can be written as
$$f(x) = sign(\sum_{i=1}^{k} g_i(x) + k-1)$$.

The size of this network can be exponentially large with n but it is capable
of representing any boolean function.


### Theorem
For every $$n$$, let $$s(n)$$ be the minimal integer such that $$\exists$$ graph (V,E)
with |V| = s(n) and whose hypothesis class contains all functions $$\{1,-1\}^n \rightarrow \{1,-1\}$$. Then, $$s(n)$$ is exposnential in $$n$$.

Proof: If the hypothesis class contains all such functions, then, it can shatter any input vector if size n. Hence, its VC dimension is $$2^n$$. Shortly we will show that the VC dimension of $$H_{V,E,sign}$$ is bounded by $$O(|E|log|E|) < O(|V|^3)$$. Then, we can state that $$2^n$$ is bounded by $$O(|V|^3)$$ and $$|V|\geq \Omega(2^{n/3})$$.

### Theorem
For $$T: N \rightarrow N, for every n, $$F_n$$ is the set of functions that can be
implemented using a turing machine with runtime bounded by $$O(T(n))$$. Then,
there is exists $$b,c \in \mathbb{R}_{+}$$ s.t. fir every b there is a graph
$$(V_n, E_n)$$ of size at most $$cT(n)^2 + b$$ such that $$H_{V,E,sign}$$ contains
$$F_n$$.

The proof of this theorem uses the relationship between time complexity and
circuit complexity. Every layer of the neural network acts as one processing
step. Each neuron is capable of NAND operations.


## Sample Complexity
### Theorem
For a hypothesis class H_{V,E,sign} with a single output neuron, its
VC dimension is $$O(|E|log(|E|))$$.

Proof: For a subset of $$X$$ of size m, the growth function $$\tau_H(m)
 = max_{C\in X:|C|=m} |H_C|$$ where $$H_C$$ is the restriction of the hypothesis
 space to the functions from C to Y, with Y being the set of output values.

 Each layer $$i$$ of a network is a mapping
 $$\mathbb{R}^|V_{i-1}| \rightarrow {+1,-1}^|V_i|$$. The hypothesis class H $$(H_{V,E,sign})$$ can be
  written as a composition of different hypothesis class corresponding to
  each of the layers: $$H = H^T \circ ... H^2 \circ H^1$$. Each of these classes
  has a corresponding growth function and we assume that the growth function
  of the composition is upper bounded by the product of the growth functions of its components:

  $$\tau_{H}(m) \leq \prod_{i=1}^{T} \tau_{H^i}(m)$$

Each layer $$i$$ contains $$|V_i|$$ neurons. Each of which can implement functions $$\{1,-1\}^{|V_t-1|} \rightarrow \{1,-1\}$$ and $$H^{i, j}$$ is the set of functions the $$j^{th}$$ neuron from layer $$i$$ can implement. Then, $$H^i = H^{i, 1} \times .... H^{i, j}$$. Assuming that the growth function of a hypothesis class formed by a product of classes is upper bounded by the product of the growth functions of its products, then :

$$\tau_{H^i}(m) \leq \prod_{j=1}^{|V_i|} \tau_{H^{i,j}}(m)$$

Considering that a neuron at layer $$i$$ has $$|V^{i-1}|$$ inputs, by Sauer's Lemma:

$$\tau_{H^{i,j}}(m) \leq (\frac{em}{|V_{i-1}|})^{|V_{i-1}|} \leq (em)^{|V_{i-1}|}$$

and considering that the number of all inputs is $$|E|$$:

$$\tau_{H}(m) \leq (em)^{|E|}$$

If we assume there are m shattered points, then, $$\tau_{H}(m) = 2^m$$, then,
$$2^m \leq (em)^{|E|}$$

which implies $$m \leq |E| log(em)/log(2)$$ and we prove our claim.

## Learning Network Weights

Solving the problem of adjusting the input weights with ERM, is a NP-Hard problem. Therefore, stochastic gradient descent is often used. For one layer networks its application is very simple. For deeper networks, however, calculating the gradient over the error surface for neurons not in the last layer is nos as straight forward. For this, Backpropagation is used.

### The Backpropagation Algorithm
We define the result of a weighted sum performed by the $$j^th$$ neuron on layer $$i$$, $$\a_{i,j}$$. It's output is the result of applying the transfer function to the $$a$$ : $$\sigma(a_{i,j})$$. The input weights of the $$j^th$$ neuron on layer $$i$$ is a vector $$\mathbf{W_{i,j}}$$

Calculating the gradient for neuron $$j$$ in the last layer $$T$$ is simple. We assume that the error function is $$E(h_w(X), y) = \frac{1}{2}||h_w(x) -y||^2$$. Then we to adjust each weight in the $$\mathbf{W}_{T,j}$$ vector. For every weight $$k$$ in the vector:
$$\frac{\partial E}{\partial W_{T,j,k}} = \frac{E_j}{\sigma(a_{T,j})}\frac{\partial \sigma(a_{T,j})}{\sigma(a_{T,j})}\frac{\sigma(a_{T,j})}{\partial W_{T,j,k} } = (\sigma(a_{T,j})-y_j)\sigma'(a_{T,j})\sigma(a_{T-1,k})$$

In hidden layers, the weight of an input can affect the error in several of next layers neurons. Therefore, the error gradient must take that into account. This is done using the concept of the error signal $$\delta$$. The error signal provided by the last layer corresponds to the first term of the product shown previously $$(\sigma(a_{T,j})-y_j)$$ or simply the difference between the expected result and the one obtained. For an intermediate layer, the error signal is a weighted sum of the next layer's error signals considering the weights connecting the neuron to the ones in the next layer:

$$\delta_{i,j} = \frac{\partial E}{\partial \sigma(a_{i,j})}=  \sum_{k=1}^{|V_i+1|}\frac{\partial E}{\sigma(a_{i+1,k})}\frac{\partial \sigma(a_{i+1,k})}{\partial a_{i+1,k}}\frac{a_{i+1,k}}{\sigma({i,j})}$$

$$\delta_{i,j} = \sum_{k=1}^{|V_i+1|} W_{i,j,k}\delta_{i+1, k}\sigma'(a_{i,j})$$

Then we can just insert this in the chain derivation and obtain the gradient for hidden layers:

$$\frac{\partial E}{\partial W_{i,j,k}} = \frac{\partial E}{\partial \sigma(a_{i,j})}\frac{\sigma(a_{i,j})}{a_{i,j}}\frac{a_{i,j}}{W_{i,j,k}} = \delta_i \sigma'(a_{i,j})\sigma(a_{i-1, k})$$



# Questions

### What is the benefit of implementing a conjunctive normal formula (CNF) or disjunctive normal formula (DNF) using a neural network?

### Are feed-forward neural networks Turing-complete?

### How is the growth function in the proof of the VC-dimension defined?
