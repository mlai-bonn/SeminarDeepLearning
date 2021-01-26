



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








