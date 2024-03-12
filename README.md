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

To validate the results of the model, two custom metrics were created: `Average Drift Gap` and `Maximum Drift Gap`.

The data sets used for the validation step are:

- [Spam](https://github.com/vlosing/driftDatasets)
- [Elec2](https://github.com/vlosing/driftDatasets)
- [CovType](https://github.com/vlosing/driftDatasets)
- [SVHN](http://ufldl.stanford.edu/housenumbers/) (preprocessed with a ResNet18 CNN)

Finally, a fair comparison was made between "Unmasking" drift detector and "D3" drift detector.