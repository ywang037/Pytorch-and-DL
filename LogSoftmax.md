# LogSoftmax vs Softmax

One can use both Softmax or Log Softmax to generate the logits at the end of any NN, or use them as the loss function. However, that Log Softmax is advantageous over Softmax and is recommended for the following reasons:

> There are a number of advantages of using log softmax over softmax including practical reasons like **improved numerical performance and gradient optimization**. These advantages can be extremely important for implementation especially when training a model can be computationally challenging and expensive. At the heart of using log-softmax over softmax is the use of log probabilities over probabilities, which has nice information theoretic interpretations. 

>When used for classifiers the log-softmax **has the effect of heavily penalizing the model when it fails to predict a correct class**. Whether or not that penalization works well for solving your problem is open to your testing, so both log-softmax and softmax are worth using.

See [here](https://datascience.stackexchange.com/questions/40714/what-is-the-advantage-of-using-log-softmax-instead-of-softmax) for the original post of the above information.

> **Numerical Stability**: Because log-softmax is a log over probabilities, the probabilities does not get very very small. And we know that because of the way computer handles real numbers, they led to numerical instability.

> **Cheaper Model Training Cost**: By using the log-softmax the model training efficiency can be greatly increased while reducing the training time. This is due to the fact that we need to do less aritematic oprations while computing the gradients.

> **Penalises Larger error**: The log-softmax penalty has a exponential nature compared to the linear penalisation of softmax. i.e More heavy peanlty for being more wrong.

See [here](https://deepdatascience.wordpress.com/2020/02/27/log-softmax-vs-softmax/) for the original post of the above information.