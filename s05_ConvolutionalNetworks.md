# Convolutional Networks



## Questions

### Are convolutional layers also equivariant to translation like the convolution operation in general? (Slide B.7)

This depends on whether padding is used or not. If padding is used, behavior on the border can differ if the input is translated.

### Why do we use zeroes for padding and not other values?

Other paddings are possible, but zeroes are a good choice as they do not introduce more noise. Zeroes generally mean the absence of signal.

### What if we have different input sizes to the whole network AND want to have output of fixed size, for example a class label? (Slide B.12)

We can use a reshape layer, global pooling, and scalable stride in pooling layers according to the input.

### How do we get to use the advantage of having separable kernels if we can not set the weights ourselves because the network learns them? (Slide B.14)

We cannot be sure if kernels will be separable. Instead, we can lay the groundwork by introducing 1D convolution layers (see MobileNet for instance).

### Is the Fourier Transform useful for the forward as well as for the backward pass?

The FFT speeds up convolution operations in general. During backpropagation, kernel weights are also updated using convolution, which means that FFT can be applied as well.

### Is there a guideline for which types of pooling to use?

Max pooling is the most widely used pooling, but it depends on the application. Guidelines exist, see the Deep Learning book by Goodfellow.

### Does the order of ReLU and pooling matter? (Slide A.8)

Yes.

### Can pooling be trained?

Not by backpropagation, as there are no trainable parameters. That is why they are called hyperparameters.

### What kind of features are captured in earlier layers, and what are captured in later layers?

Complex features are recognized in later layers, simple features such as lines, shapes and edges in early layers.

### Is the pictured network a representative example of CNNs? (Slide A.8)

Yes, but e.g. ResNet and other networks have hundreds of layers, the depth depends on what kind of data or problem is to be solved, and how much data is available. More parameters require more data to train. 

### What layers are responsible for pixels that are close to each other, and what layers are responsible for pixels that are away from each layer?

The receptive field of feature maps grows larger as the network goes on (see slide 11). Pooling layers enable us to gather information about pixels that are far away from each other, as they produce statistical summaries rather than spatial encodings like convolutional layers.

### Why are the Toeplitz matrices sparse? (Slide A.7)

Neurons in convolutional layers are sparsely connected and often have small kernel sizes, so most entries in the Toeplitz matrices would be zero.

### What are tied weights? (Slide A.12)

Kernel weights are the same for every pixel. Another interpretation is that changing one weight changes the other "tied" weights to the exact same value.

### Why can convolutional layers be better than fully connected layers? (Slide A.13)

There are fewer weights that need to be stored, which leads to faster training and a better memory footprint.

### How does prior help with the parameter determination?

A strong prior means we give more weight to prior assumptions about model parameters.

### Why does convolution and pooling cause underfitting? (Slide A.20)

They are translation equivariant and rotation/scale invariant respectively, and have also fewer parameters than fully connected layers.

### How does L2-Norm pooling work? (Slide A.15)

Instead of the max or average, we take the sum of squares of pixels in a region as reduction technique.

### Are there other advantages of strided convolution other than performance?

There is a big difference to statistic summation; if you use pooling, the result is affected by all the input in an area. See receptive field.

### Is convolution with multiple channels commutative? 

The commutative property only holds for same number of output channels and input channels.

### How can invariance be obtained using max pooling?

The actual neuron that produced the max result, i.e. its spatial location does not matter exactly, only that it is part of the pooling region (e.g. the upper left quadrant). This enforces spatial invariance.

### Is a varying input size the same as rescaling?

No, because rescaling requires interpolation or reduction, which can vary greatly depending on the method.
