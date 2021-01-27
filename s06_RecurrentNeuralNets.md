# Recurrent Neural Networks

## Introduction
Recurrent Neural Networks (RNNs) are a class of neural networks for processing sequential data. RNNs are designed to recognize a data's sequential characteristics and use patterns to predict the next likely scenario. They use feedback loops to process a sequence of data that informs the final output, which can also be a sequence of data. These feedback loops allow information to persist; the effect is often described as memory.
Compared to a traditional fully connected feedforward network which has separate parameters for each input feature, (so it would need to learn all the rules of the language separately at each position in the sentence), a RNN shares the same weights across several time steps. Recurrent networks share parameters in a diﬀerent way. Each member of the output is a function of the previous members of the output. Each member of the output is produced using the same update rule applied to the previous outputs.This recurrent formulation results in the sharing of parameters through a very deep computational graph.

## §10.1. Unfolding Computational Graphs

## §10.2 RNNs by examples

## §10.2.1 Teacher Forcing

## §10.2.2 Computing the Gradient in a RNN

## §10.2.3 Recurrent Networks as Directed Graphical Models

## §10.2.4 Modeling Sequences Conditioned on Context with RNNs

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








