# Practical Methodology


## Questions

### For real-life tasks, it's impossible to achieve absolute zero error because of the so called *Bayes error*. How is Bayes error defined and how can it be estimated?

Bayes error is the minimum error that is possible to achieve and is analogous to the irreducible error. We can use human-level error as a proxy for estimating the Bayes error. For example the performance of a team of specialists could serve as the human-level error for the specific task.

### What techniques can be used to improve a model in case we have a large gap between the human-level error and our training error?

We could use *bias reduction* techniques, e.g. increasing model capacity, decreasing regularization, or if our model is already complex enough but still poorly performing, then we should consider collecting a larger and richer dataset.

### Consider a network outputting the probability of an event to be positive (e.g. with a single sigmoid output unit). Then we can take a threshold in range (0,1) and report a detection only when the probability is above this threshold. How does the choice of this threshold control the model's Precision and Recall metrics?

The smaller the threshold, the more likely we are to output false positives and less likely to miss detecting a positive event. Thus, the smaller Precision and the higher Recall will be. And vice versa, the bigger the threshold is, the smaller Recall and the higher Precision will be.

### Consider a model that is trained to generate samples according to some data distribution (e.g. a GAN trained on human face images). What could be a good evaluation metric for this kind of task?

*Inception score*, proposed in [this](https://arxiv.org/abs/1606.03498) paper, is an example metric for evaluating the performance of such generative models.
It evaluates both the quality and diversity of the generated images. It includes use of an extra pretrained neural network for classifying the generated images.

### When does it make sense to combine supervised and unsupervised learning methods? 

Often we have lack of *labeled* training data but have plenty of unlabeled data. Obviously, pure supervised approach might not give satisfactory results. In this case, we could e.g. train an *Autoencoder* model in unsupervised way on the large dataset available, thus extracting useful feature representations. The encoder part can then be extended with some additional layers (e.g. a fully-connected layer yielding the output class scores) and further trained on the smaller labeled dataset in supervised manner.

### How can we decide on how much data we need to collect?

It is a good idea to monitor how the generalization error decreases with the growth of the training dataset. By plotting sich a curve and extrapolating it, we can hope to predict how much more data we would need to achieve the desired performance.

### Does optimizing for different hyperparameters separately guarantee an overall optimal set of hyperparameters?

In general, no. That is why hyperparameter optimization is considered such a hard problem.

### Name some hyperparameters for which overfitting occurs when the value is large and some for which it occurs when the value is small.

An example of a hyperparameter whose very large value may cause overfitting is the *number of epochs*. And an example of a hyperparameter whose very small value may cause overfitting is regularization (e.g. weight decay) strength.

### What is the difference between *primary* and *secondary* hyperparameters in the context of automatic hyperparameter tuning?

Hyperparameter optimization algorithms have their own parameters that can be called "secondary" hyperparameters, e.g. the range of values (and the step) to be explored for each hyperparameter. So in this context, we have primary hyperparameters of the learning algorithm automatically tuned by the hyperpameter optimization algorithm, and secondary hyperparameters which should be tuned manually. 

### Does the "number of epochs" hyperparameter also control the model capacity?

It doesn't control the *representational capacity* of the model, but controls its *effective capacity*, since the longer we train the better our model tends to fit the training data and eventually overfit it.

### What are some popular hyperparameter optimization packages?

Hyperopt, Ray-Tune, Optuna, etc.

### What is the main idea of Bayesian hyperparameter optimization?

Bayesian optimization, in contrast to Random search and Grid search, keeps track of the previous evaluation results which are used to form a probabilistic model from hyperparameter values to a probability of achieving certain scores for the objective function.

### We often encounter the term “Percent error” in the literature. How is it defined?

$PE = \frac{|Accepted\_value - Experimental\_value|}{Accepted\_value} x 100\%$

For example, in case of 100% accuracy taken as the accepted value for performance, the Percent error will be $100 - model_accuracy\%$.

### What if we have low test error, but after deploying the model we get bad reviews from the customer complaining about the system performance at real-time? What could be the reason and what steps should we undertake?

The possible reason could be the inconsistency between the data that we used for training/testing our model and the actual data used at inference-time. So, it would be a good idea to start collecting a richer dataset that better reflects the actual data distribution.  

### When monitoring the paramter values together with their update magnitudes at each iteration, what should the relation between these values be?

As proposed by Bottou (2015), the magnitude of a parameter update over a minibatch should be around 1 percent of the parameter magnitude.

### In the example project of Street View transcription system, people used a CNN with multiple output softmax units to predict a sequence of n characters. But this assumes predicting a fixed number of maximum digits in the image. Wouldn’t it be more suitable to use an RNN instead of a CNN in this case?

They could probably also use a CNN in combination with an RNN in this project. For example, a CNN encoder to get a useful feature representation, followed by an RNN decoder to output the sequence of digits. But the reason for not considering more complex models could be that the achieved performance was already acceptable enough.
