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
def drift_detector(S,T,threshold = 0.05, min_feature_percentage_remaining=0.3, no_batches=6):
    # Compute the number of features that are going to be dropped with each iteration
    # as a function of the minimum percentage of features that we want in the end, and the number of iterations
    percentage_per_batch = np.round((1-min_feature_percentage_remaining) / no_batches, 2)
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

    # Predict the given features using the chosen model. If the index is at least 1,
    # check if the current AUC score is different from the one at index 0. If yes,
    # return true. Otherwise, remove an arbitrary number of features(=`features_dropped`), namely the ones with 
    # the highest values. Repeat for an arbitrary number of iterations (=`no_batches`).
    for i in range(no_batches):
        clf.fit(ST[train_idx], labels[train_idx])
        probs = clf.predict_proba(ST[test_idx])[:, 1]
        predictions[test_idx] = probs
        auc_score = AUC(labels, predictions)
        if i == 0:
            first_score = auc_score
        elif first_score - auc_score > threshold:
            return True
        removed_features_idx = np.argsort(np.abs(clf.coef_))[0,-features_dropped:]
        ST = np.delete(ST, removed_features_idx, axis=1)
    return False