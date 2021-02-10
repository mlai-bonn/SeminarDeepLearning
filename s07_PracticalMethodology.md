# Practical Methodology


## Questions

### For real-life tasks, it's impossible to achieve absolute zero error because of the so called *Bayes error*. How is Bayes error defined and how can it be estimated?

Bayes error is the minimum error that is possible to achieve and is analogous to the irreducible error. We can use human-level error as a proxy for estimating the Bayes error. For example the performance of a team of specialists could serve as the human-level error for the specific task.

### What techniques can be used to improve a model in case we have a large gap between the human-level error and our training error?

We could use *bias reduction* techniques, e.g. increasing model capacity, decreasing regularization, or if our model is already complex enough but still poorly performing, then we should consider collecting a larger and richer dataset.

### Consider a network outputting the probability of an event to be positive (e.g. with a single sigmoid output unit). Then we can take a threshold in range (0,1) and report a detection only when the probability is above this threshold. How does the choice of this threshold control the model's Precision and Recall metrics?

The smaller the threshold, the more likely we are to output false positives and less likely to miss detecting a positive event. Thus, the smaller Precision and the higher Recall will be. And vice versa, the bigger the threshold is, the smaller Recall and the higher Precision will be.

### How to estimate desired error rate before beginning with the project? 
In research, previous work on the same problem may help to estimate the error. In the real world applications, we have some ideas on how good humans are at some tasks. For instance, the street view task needs to have less than 5% error rate due to its sensitive application.
 	
### What is the difference between performance metric and loss function? 
Loss functions are functions that show the model performance during the training of a machine learning model. However, metrics are used to monitor and measure the performance of a model during training and testing. 	 	

### What does the term “coverage” in the book denote? Why is the accuracy 100% if the coverage is 0?
Coverage is a performance metric, it describes the amount of samples that a ML model can cover/classify. A zero coverage means that the algorithm did not yield any result and that all samples are human-classified, therefore, accuracy=100%.

### Do you know other widely used classification metrics than confusion matrix, Precision, recall? 
Receiver Operator Characteristics (ROC) curves plot the false positive rates vs. true positives at various threshold settings.

### You have cited some performance metrics like Confusion Matrix, Precision, Recall and so on. Are these performance metrics suitable to be used in unsupervised learning? 
No, these metrics are widely applied in supervised learning models and specifically, in classification tasks. But for some unsupervised learning techniques, e.g. clustering, one could transform the problem into a supervised learning problem by preparing a data set labeled by hand for calculating the metrics.


### Consider a model that is trained to generate samples according to some data distribution (e.g. a GAN trained on human face images). What could be a good evaluation metric for this kind of task?

*Inception score*, proposed in [this](https://arxiv.org/abs/1606.03498) paper, is an example metric for evaluating the performance of such generative models.
It evaluates both the quality and diversity of the generated images. It includes use of an extra pretrained neural network for classifying the generated images.

### When does it make sense to combine supervised and unsupervised learning methods? 

Often we have lack of *labeled* training data but have plenty of unlabeled data. Obviously, pure supervised approach might not give satisfactory results. In this case, we could e.g. train an *Autoencoder* model in unsupervised way on the large dataset available, thus extracting useful feature representations. The encoder part can then be extended with some additional layers (e.g. a fully-connected layer yielding the output class scores) and further trained on the smaller labeled dataset in supervised manner.

### The authors mentioned that NN research progresses rapidly and that some of the default algorithms in the book would be replaced by others soon. Now, after 5 years of publishing this book, can you prove this statement by some examples? 
For Computer Vision/Topological structure, Graph Neural Networks and Capsule Neural Networks are used nowadays to overcome the limitations of CNN. For sequence based input/output, Transformers/AttentionNeuralNetworks (2017), on which BERT and XLNet are based, have been successfully used. In addition, Momentum LSTM is a new technique proposed in 2020. In general, almost every task proposes its own neural network, optimization and regularization techniques in order to achieve the intended results. For this reason, novel neural networks and optimization strategies regularly appear in the research field.
 	
### What are default optimization techniques mentioned in the book? Are they still the same in 2021? If not, what algos should we try out first? 
Basically, the optimization techniques, e.g. SGD with momentum, Adam and batch normalization, are still widely applied nowadays. In addition, there is progress to help apply Newton’s method.
 
### What default regularization techniques can we use if we have millions of training 	data? Are these techniques still the same in 2021? 
Early stopping, dropout and batch normalization are still leading in the regularization of neural networks. Although, there are other regularization strategies that are very popular, e.g. Bagging/Ensemble Voting, Adversarial Training, Noise Robustness and L1 and L2 norms.
 	
### In what cases can we choose unsupervised learning as default? 
First, when it is known and proven that the problem to tackle works better if using unsupervised learning, e.g. Natural Language Processing (NLP). Second, if the task belongs to the unsupervised learning paradigm. Third, after applying the supervised methodology, one could attempt to try out the unsupervised one to see if better results can be achieved.

### How can we decide on how much data we need to collect?

It is a good idea to monitor how the generalization error decreases with the growth of the training dataset. By plotting sich a curve and extrapolating it, we can hope to predict how much more data we would need to achieve the desired performance.

### Overall, you showed 2 cases when there is a need to collect more data, can you cite them? 
Case1: If the performance of the algorithm is less on testing data than on testing data. 
Case2: If after collecting more data and regularizing the model, we still have poor performance. 
 	
### What sectors always benefit in general from the choice of collecting more data? And what sectors do not profit from such a setting? 
Big internet companies with millions of users profit from gathering more data, because they have enough (labeled) data from their users at their disposal. However, in medical areas, where experiments are conducted for a long time and at high cost, it is hard to collect more data.
 	
### If we decided to gather more data, how much data should we add? 
The first step is to plot the relationship between training error and generalization error and conclude the amount of data needed. Then, increase successively the size of datasets to be added, e.g. on a logarithmic base. Note that very small datasets do not improve the performance dramatically. 

### Solving any task with research is known to be very time consuming and demands money and funding. When should it be considered in your diagram? 
At the end under the constraint that it is infeasible to get good results even after a second data collection.

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

### On slide 26, you showed a U-shaped curve showing how the generalization error behaves with respect to training error. This shape is feasible because the hyperparameter examined is continuous. How does the U-shaped curve change if the hyperparameter is discrete? Binary? 
If the hyperparameter is discrete, e.g. number of units in a layer, it is only possible to plot some points along the U-shaped curve. If the hyperparameter is binary, they can only explore two points on the U-shaped curve.
 	
### How does the training error change when increasing the learning rate? 
The training error decreases and it forms a U-shaped curve if depicted with an increasing learning rate under assumption that it was chosen correctly. Otherwise, the training error strongly depends on the problem we are trying to solve, the model’s architecture and other hyperparameters beside the learning rate. 	
### The training error is low when the capacity is high and the test error depends on the gap between training and test error. Can you explain the idea behind error trade-off in successful neural networks models? 
The gap must decrease (regularization params) faster than the increase of training error.  


### Who can manually tune hyperparameters? 
Experienced people in hyperparameter optimization or people that worked on the same project before.


### GridSearch and RandomSearch are two hyperparameter optimization algorithms that belong to the exhaustive search of space class. Can you name some other algorithms that are responsible to optimize a model’s hyperparameters? What is the clue with these algorithms in general? 
Besides the exhaustive search, there exists the sequential model based optimization, e.g. Bayesian Optimization (BO) and Tree-structured Parzen Estimator (TSPE). In addition, the Hyperband algorithm is an extension of the RandomSearch and Population-based training. The clue with these algorithms is that they have parameters that need to be adjusted for each model. 


### Do you think that it is possible to use GridSearch for unsupervised learning tasks? 
Yes, but we have to get rid of the cross validation first because it requires labeled data.
### On slide 32, to which hyperparameter(s) such a logarithmic scale corresponds?  Does it have to be the same for all the hyperparameters? 
On slide: learning rate. No, it is not the same, e.g. hidden units hyperparameters range from the following set {50, 100, 200, 500, 1000, 2000}. 	
### Compare grid search and random search.
The complexity of grid search grows exponentially in the number of hyperparameters. For different experiments, grid search delivers the same result even if the values of hyperparameters are different (see slide 35). Unlike grid search, random search does not repeat experiments, it only tests a unique value for each interesting hyperparameter and reduces validation set error faster than grid search. The major benefit of grid search is that it can be parallelized even though this parallelization poses a new computational problem.
### Does it make sense in your opinion to run grid search more than one time to get the best model? If yes, is this idea also applicable for random search? 
Yes, grid search improves its findings by running its experiments repeatedly while focusing on successful parameter sets defined from the previous GridSearch results. The same concept is also applicable for random search.
	
### What is the difference between hyperparameter and model parameter? 
The model parameters are part of the neural network model which derives from data automatically, e.g. weights. Whereby hyperparameters are parameters that are chosen manually by an expert or heuristically, they are not part of the model and do not relate directly with data but they help estimate the model parameters, e.g. the learning rate of a neural network is a hyperparameter.
### Can you explain what is the “gradient” mentioned on slide 36? Can we calculate the gradient for hyperparameters of discrete type? 
The gradient here is the derivative of the error during backpropagation with respect to (thousands) hyperparameters. It is not feasible to calculate this gradient for discrete hyperparameters because it will not be differentiable.
 	

### Can you explain the terms Exploration and Exploitation as well as their relationship to Bayesian regression model? 
The Bayesian regression model makes an expectation on the error of the validation set when using a hyperparameter and draws the uncertainty around this expectation. Optimization using Bayesian regression models can be seen as a tradeoff between exploration and exploitation. On one hand, exploration is a searching operation where hyperparameters are chosen that yield high uncertainty leading to either a large improvement or a poor performance. On the other hand, exploitation is a refinement operation in which the model chooses a hyperparameter with which it is confident, the confidence comes from the previously seen examples because of the assumption that this hyperparameter would perform at least as well as the previous ones. 
 	
### The authors did not recommend using Bayesian Optimization in their book. Has it been widely used after 2016?
No, mainely in research and big companies like Facebook and Amazon because using it requires knowledge in Bayesian techniques, and they are time and computationally costly
 	
### What is the benefit of randomSearch over model-based hyperparameter optimization? What has been a mitigation technique for it? 
Hyperparameter optimization algorithms must completely run a training experiment in order to extract any information from the experiment, unlike random search that have the opportunity to only experiment with promising hyperparameters. The proposed algorithm for mitigation can choose to begin a new experiment, freeze a running experiment that appears not so important/promising or to unfreeze/thaw a previously frozen experiment. 


### What is the main idea of Bayesian hyperparameter optimization?

Bayesian optimization, in contrast to Random search and Grid search, keeps track of the previous evaluation results which are used to form a probabilistic model from hyperparameter values to a probability of achieving certain scores for the objective function.

### What if we have low test error, but after deploying the model we get bad reviews from the customer complaining about the system performance at real-time? What could be the reason and what steps should we undertake?

The possible reason could be the inconsistency between the data that we used for training/testing our model and the actual data used at inference-time. So, it would be a good idea to start collecting a richer dataset that better reflects the actual data distribution.  

### When monitoring the paramter values together with their update magnitudes at each iteration, what should the relation between these values be?

As proposed by Bottou (2015), the magnitude of a parameter update over a minibatch should be around 1 percent of the parameter magnitude.

### Debugging neural networks is a tough task. Why? 
One reason is that we have no idea on how the algorithm should behave. Another reason is that the parts of machine learning models are adaptive and depend on each other during training, loss function, weighting and adjusting hyperparameters. If a part failed, others part do not stop, instead, they continue their calculations with false measurements.
 	
### Cite some debugging strategies. Which debugging strategy was used in the StreetView project? 
1. Visualize the model in action to look beyond quantitative performance and evaluate the output of the model. But here, make sure that good results may be misleading. 
2. Visualize the worst mistakes to unveil mistakes, e.g. in data preprocessing and labels (used in the Street View project). 
3. Reason about software using training and test error to detect clues in software implementations e.g. reloading a model after saving it does not work properly. 
4. Fit a tiny dataset: by fitting only 1 example to the model which should classify it correctly. If it does not, use autoencoders to regenerate the same example. If it does not work, there is a problem with the software. 
5. Compare back-propagated derivatives to numerical derivatives. 
6. Monitor histograms of activations and gradient: tells if the units of activation functions saturate and how much they do that.
### Is it sufficient to look at the model’s output to make sure that the model works? 
No, because the bug is not necessarily seen from the output. Sometimes, the output looks accurate, but there are still mistakes in the model.

### What accuracy did they achieved in the street view project at google at the end?
At the end, the accuracy was 98%, greater than their intended accuracy.

### In the example project of Street View transcription system, people used a CNN with multiple output softmax units to predict a sequence of n characters. But this assumes predicting a fixed number of maximum digits in the image. Wouldn’t it be more suitable to use an RNN instead of a CNN in this case?

They could probably also use a CNN in combination with an RNN in this project. For example, a CNN encoder to get a useful feature representation, followed by an RNN decoder to output the sequence of digits. But the reason for not considering more complex models could be that the achieved performance was already acceptable enough.
