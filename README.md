# Unsupervised Concept Drift Detection using Unmasking Algorithm

The purpose of this project is to detect the distribution drifts in the provided data sets, using [Unmasking algorithm](https://www.jmlr.org/papers/v8/koppel07a.html), which implies removing features iteratively to determine how similar two batches of data are. This project was inspired by the implementation of [Concept Drift Detection with a Discriminative Classifier (D3)](https://github.com/ogozuacik/d3-discriminative-drift-detector-concept-drift).


    
For each iteration, the algorithm performs the following steps:

1. Fit the classifier to the training data where label 0 corresponds to the old data and label 1 corresponds to the new data.
2. Predict the probabilities of the test data belonging to class 1 (new data) using the classifier.
3. Calculate the AUC score using the true and the predicted labels.
4. If the mean AUC score exceeds a predefined threshold, return True, meaning that a concept drift was detected.
5. If the mean AUC score doesn't exceed the threshold, select the features with the smallest coefficients and remove them from the feature matrix.
6. If the loop completes without meeting the threshold, return False, indicating that no significant concept drift was detected.

### Validation technique

To validate the performance of the drift detector, 4 custom drifted datasets were used. Each dataset was created through the following steps:

1. Beginning with two datasets, the first was utilized to train a ResNet18 CNN for up to 25 epochs to ensure effective prediction of values.
2. The second dataset was then combined with the first in known batches, facilitating identification of concept drift locations.
3. This merged dataset was input into the ResNet18 model, and features were extracted from the average pooling layer.

With the datasets prepared, the drift detector was applied to identify the drifts. The results are presented in `Results.pdf` and were compared against [D3 drift detector](https://github.com/ogozuacik/d3-discriminative-drift-detector-concept-drift).

The original datasets utilized to generate the custom datasets, according to the explanation above, are as follows:

- First dataset: [SVHN](http://ufldl.stanford.edu/housenumbers/) (labels 0-4), Second dataset: [SVHN](http://ufldl.stanford.edu/housenumbers/) (labels 5-9).
- First dataset: [SVHN](http://ufldl.stanford.edu/housenumbers/), Second dataset: [MNIST](https://paperswithcode.com/dataset/mnist). 
- First dataset: [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) (labels 0 - 49), Second dataset: [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) (labels 50 - 99).
- First dataset: [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) (labels 0 - 49), Second dataset: [MNIST](https://paperswithcode.com/dataset/mnist). 

