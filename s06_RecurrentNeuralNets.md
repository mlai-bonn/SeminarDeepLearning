# Recurrent Neural Networks

## Introduction
Recurrent Neural Networks (RNNs) are a class of neural networks for processing sequential data. RNNs are designed to recognize a data's sequential characteristics and use patterns to predict the next likely scenario. They use feedback loops to process a sequence of data that informs the final output, which can also be a sequence of data. These feedback loops allow information to persist; the effect is often described as memory.
Compared to a traditional fully connected feedforward network which has separate parameters for each input feature, (so it would need to learn all the rules of the language separately at each position in the sentence), a RNN shares the same weights across several time steps. Recurrent networks share parameters in a diﬀerent way. Each member of the output is a function of the previous members of the output. Each member of the output is produced using the same update rule applied to the previous outputs.This recurrent formulation results in the sharing of parameters through a very deep computational graph.

## §10.1. Unfolding Computational Graphs

## §10.2 RNNs by examples
Motivated by the graph-unrolling and parameter-sharing ideas we can design different kinds of RNN. As a first example (slide 7), we have an RNN that has recurrent connections between hidden units. In this computational graph we have an input sequence of x values which are mapped to a sequence of output o values. And there is a loss function which computes how far each o is from the corresponding training target y. The network has input to hidden connections parametrized by a weight matrix U, while the recurrent connections parametrized by a weight matrix W, and hidden-to-output connections parametrized by a weight matrix V.
We can develop the forward propagation equations (slide 8) based on some assumptions. First, we assume that for the hidden units we will use hyperbolic tangent as the activation function and the output is discrete, as if the RNN is used to predict words or characters. Then we apply the softmax operation as a post-processing step to obtain a vector y of normalized probabilities over the output. 
The total loss for a given sequence of x values paired with a sequence of y values would then be just the sum of the losses over all the time steps.
As a second example (slide 9) we have an RNN that produces an output at each time step and has recurrent connections only from the output at one time step to the hidden units at the next time step.
The RNN in the previous one can choose to put any information it wants about the past into its hidden representation h and transmit h to the future. The RNN in the second example is trained to put a speciﬁc output value into o, and o is the only information it is allowed to send to the future. There are no direct connections from h going forward. Unless o is very high-dimensional and rich, it will usually lack important information from the past. This makes the latter RNN less powerful, but it may be easier to train because each time step can be trained in isolation from the others, allowing greater parallelization during training.

## §10.2.1 Teacher Forcing
Models that have recurrent connections from their outputs leading back into the model may be trained with teacher forcing. Teacher forcing is a procedure that emerges from the maximum likelihood criterion (slide 10). The main idea is that during training, instead of feeding the model’s own output back into itself, these connections should be fed with the target values specifying what the correct output should be.
The advantage of teacher forcing is that it allows us to avoid back-propagation through time in models that lack hidden-to-hidden connections.

## §10.2.2 Computing the Gradient in a RNN
Computing the gradient through a RNN is straightforward.One simply applies the generalized back-propagation algorithm to the unrolled computational graph.
First, we compute the gradients on the internal nodes of the computational graph (slide 12), then we can obtain the gradients on the parameter nodes (slide 13).
We do not need to compute the gradient with respect to input value for training because it does not have any parameters as ancestors in the computational graph deﬁning the loss.

## §10.2.3 Recurrent Networks as Directed Graphical Models
How can we interpret the RNNs as Directed Graphical Models so we can see direct dependencies? One way to do it is just by ignoring the hidden layer (slide 14). Parametrizing the graphical model directly according to the graph in slide 14 might be very inefficient, with an ever growing number of inputs and parameters for each element of the sequence.
Including the hidden units in the graphical model reveals that the RNN provides an efficient parametrization of the joint distribution over the observations. Incorporating the hidden nodes in the graphical model decouples the past and the future, acting as an intermediate quantity between them. The structure of the graph on slide 15 shows that the model can be efficiently parametrized by using the same conditional probability distributions at each time step, and that when the variables are all observed, the probability of the joint assignment of all variables can be evaluated efficiently.
Even with the efficient parametrization of the graphical model, some operations remain computationally challenging. One of them is determining the length of the sequence while drawing samples. This can be achieved in various ways:

1. Special symbol at the end of the sequence
2. Extra Bernoulli output
3. Predicting sequence length 𝛕

## §10.2.4 Modeling Sequences Conditioned on Context with RNNs
In general, RNNs allow the extension of the graphical model view to represent not only a joint distribution over the y variables but also a conditional distribution over y given x. This can be achieved in different ways. The most common two choices are the followings:

1. To take only a single vector x as input. When x is a ﬁxed-size vector, we can simply make it an extra input of the RNN that generates the y sequence. This RNN is appropriate for tasks such as image captioning, where a single image is used as input to a model that then produces a sequence of words describing the image.
2. Rather than receiving only a single vector x as input, the RNN may receive a sequence of vectors as input. To remove the conditional independence assumption, we can add connections from the output at time t to the hidden unit at time t+ 1, as shown in ﬁgure on slide 18. The model can then represent arbitrary probability distributions over the y sequence. This kind of model representing a distribution over a sequence given another sequence still has one restriction, which is that the length of both sequences must be the same.

## §10.7 Long Term Dependencies

## §10.10 Gated RNNs


## Questions


### How are LSTM trained?

They can be trained like each other type of RNN.

### How are text inputs for LSTMs parsed?

We can use word embeddings for that, or split the input an input the single characters to the network.

### Is there an alternative to BPTT?

No, we always have to unfold the network for training.

### Can RNNs be used for Computer Vision?

Maybe. It's possible to give an image as input. The difficulty is, to design a proper RNN, to get the desired outputs.

### Is there a limit for the length of an input?

Not really. If we design a RNN in a proper way, it can process very long inputs. But we also need to consier the computational costs.

### Can we use batch normalization in RNNs?

Yes. But LSTM are doing already a kind of normalization.

### What are the similarites between RNNs and CNNs?

Both can work with big data and can process long seqences. Also they both can process data of varying length.

### Where are LSTMs used?

Audio processing, Speech-to-text, Text-to-Speech, Image-to-text, Video subtitling

### Why do the gradients vanish or explode over time?

Because the same weights get multiplied with itself over and over in each time step. This leads to a vanishing when the values are small or to a explosion when they are big.

### How to avoid gradient vanishing/ exploding?

Using gradient clipping, define a maximum value use LSTMs which avoid this by it's design.

### What are the advantages and disadvantages of hidden-to-hidden and output-to-input recurrence?

Hidden-to-hidden connections allow for sending any information to the future with less loss of information and are more powerful in general (Turing complete). Input-to-output connections allow the use of teacher forcing, which avoids BPTT.

### Are there types of sequential data for which RNNs are not suitable?

RNNs can in principle be used for any sequential data, but they have difficulties with learning long-term dependencies.

###  When predicting the sequence length, do we predict it only once in the beginning or at each time step?

It is predicted again at each time step.

### What is the advantage of employing both hidden-to-hidden and output-to-input recurrence?

It is useful to compensate for missing information.

### Why might a network trained with teacher forcing work poorly in open-loop mode?

Because it the input values received during testing might differ a lot from the input values received during training. 

### Is transfer learning possible with RNNs?

Yes.
