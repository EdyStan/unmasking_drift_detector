# Unsupervised Concept Drift Detection using Unmasking Algorithm

The purpose of this project is to detect the distribution drifts in the provided data sets, using [Unmasking algorithm](https://www.jmlr.org/papers/v8/koppel07a.html), which implies removing features iteratively to determine how similar two batches of data are. This project was inspired by the implementation of [Concept Drift Detection with a Discriminative Classifier (D3)](https://github.com/ogozuacik/d3-discriminative-drift-detector-concept-drift).



The algorithm operates through the following steps:
    
1. A model is applied over the old+new data points, and evaluates the AUC score between the true labels(which mark the old data as 0 and the new data as 1) with the predicted ones. 
    
2. If this is the first iteration, the AUC score is stored in a variable, else, the AUC score is compared with the first one. 

3. In case the difference between the AUC scores reaches a threshold, a drift is detected and the loop is interrupted. 

4. Else, an arbitrary number of the most distinctive features is removed and the iteration is continued until a certain percentage of the initial features is left.

### Validation technique

To validate the results of the model, two custom metrics were created: `Average Drift Gap` and `Maximum Drift Gap`.

The data sets used for the validation step are:

- [Spam](https://github.com/vlosing/driftDatasets)
- [Elec2](https://github.com/vlosing/driftDatasets)
- [CovType](https://github.com/vlosing/driftDatasets)
- [SVHN](http://ufldl.stanford.edu/housenumbers/) (preprocessed with a ResNet18 CNN)

Finally, a fair comparison was made between "Unmasking" drift detector and "D3" drift detector.