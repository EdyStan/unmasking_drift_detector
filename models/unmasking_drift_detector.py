import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import train_test_split




# Drift Detector
# S: Source (Old Data)
# T: Target (New Data)
# ST: S&T combined
def drift_detector(S,T,threshold = 0.56, min_feature_percentage_remaining=0.3, iterations=6):
    # Compute the number of features that are going to be dropped with each iteration
    # as a function of the minimum percentage of features that we want in the end, and the number of iterations
    percentage_per_batch = np.round((1-min_feature_percentage_remaining) / iterations, 2)
    features_dropped = np.floor(T.shape[1] * percentage_per_batch).astype(int)
    T = pd.DataFrame(T)
    S = pd.DataFrame(S)
    T['in_target'] = 0 # in target set
    S['in_target'] = 1 # in source set
    ST = pd.concat( [T, S], ignore_index=True, axis=0)
    labels = ST['in_target'].values
    ST = ST.drop('in_target', axis=1).values

    clf = LogisticRegression(solver='liblinear')
    idx = np.arange(ST.shape[0])
    np.random.shuffle(idx)
    train_idx, test_idx = np.split(idx, 2)
    predictions = np.zeros(labels.shape)
    all_auc = []

    # For each iteration, the code performs the following steps:
    # 1. Fit the classifier (`clf`) to the training data (`ST[train_idx]`) 
    # where label 0 corresponds to the old data and label 1 corresponds to the new data.
    # 2. Predict the probabilities of the test data (`ST[test_idx]`) belonging to class 1 (new data) using the classifier.
    # 3. Calculate the AUC score using the true and the predicted labels.
    # 4. If the mean AUC score (stored in `all_auc`) exceeds a predefined threshold, return True, 
    # meaning that a concept drift was detected.
    # 5. If the mean AUC score doesn't exceed the threshold, select the features with the 
    # smallest coefficients (as determined by `clf.coef_`) and remove them from the feature matrix ST.
    # 6. If the loop completes without meeting the threshold, return False, 
    # indicating that no significant concept drift was detected.

    for _ in range(iterations):
        clf.fit(ST[train_idx], labels[train_idx])
        probs = clf.predict_proba(ST[test_idx])[:, 1]
        predictions[test_idx] = probs
        auc_score = AUC(labels, predictions)
        all_auc.append(auc_score)
        if np.mean(all_auc) > threshold:
            return True
        removed_features_idx = np.argsort(np.abs(clf.coef_))[0,-features_dropped:]
        ST = np.delete(ST, removed_features_idx, axis=1)
    return False
