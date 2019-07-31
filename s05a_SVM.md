# Support Vector Machines
Support Vector Machine is a useful learning paradigm for higher-dimensional problems where a linear separation (and predictor) of a data distribution is desired. It scales well to larger dimensions if constrained by a regularization measure.

## Hard SVM
Given a set of linearly separable labeled points, what is the *best* separating hyperplane? One way to define *best* is to seek to maximize the distance from the hyperplane to the points. If this distance is larger, we can be more certain about are classification. This distance is defined as margin. Given a set of labeled points and a separating hyperplane, the margin is the minimum distance of any point to the plane.

Under Hard SVM we wish to find a viable solution that maximizes the margin. We can state this as a optimization problem:
$$w_0, b_0 = argmin_{(w,b)} ||w||^2 s.t. \forall i y_i(<w, x_i> + b) \geq 1$$

$$w_0 $$ encodes the length of the margin. If the vector $$w$$ were to be normalized the previous inequality would be defined as $$\geq margin$$. The length of the margin $$\gamma$$ is $$\frac{1}{w_0}$$

Bias can be included in the regularization by adding a new dimension to the data and learning a homogenous half-space. Then, the bias is included in $$w$$ and will be regularized

### Sample complexity
For halfspaces, the VC dimension is d+1 which does not scale well with larger dimensions. By assuming that the generating distribution is separable with margin $$\gamma$$, then the sample complexity is $$O(\frac{1}{\gamma})$$. The scale of $$\gamma$$, however, is relevant and it must be proportional to the scale of the data. One can double the margin of a separator by multiplying the points by two.

If the margin is relatively small with relation to the data, there can be a lot of possible suboptimal separators. A distribution D is separable with a $$(\gamma, p)$$-margin if $$y(<w, x> + b) \geq \gamma$$ for every x s.t. $$\vert\vert x \vert \vert \leq p$$ and $$\vert \vert w \vert \vert = 1$$

Theorem: Let $$D$$ be a distribution over $$\mathbb{R}^d \times \{1,-1\}$$ separable with a homogenous halfspace and a $$(\gamma, p)$$-margin. With probability at least $$1-\delta$$ the choice of a training set of size m drawn i.i.d from D the 0-1 error of the output of Hard SVM is bounded by $$\sqrt{\frac{4(p/\gamma)^2}{m}}+\sqrt{\frac{2log(2/\delta)}{m}}$$.

Then, with the restriction on the norm of x and the size of the norm, we obtain a sample complexity that does not depend on the dimension of the input data.

## Soft SVM
Soft SVM allows for missclassification. For this,  slack variables $$\xi_i$$ are added to the optimization problem which represent how far are each data point from the correct margin:

$$min_{\mathbf{w}, b, \mathbf{\xi}} \lambda ||\mathbf{w}|| + \frac{1}{m} \sum_{i=1}^m \xi_i$$

s.t.
$$\forall i, y_i(<\mathbf{w}, \mathbf{x_i}> + b) \geq 1 - \xi_i, \xi_i \geq 0$$
