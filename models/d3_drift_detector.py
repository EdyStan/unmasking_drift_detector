import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score as AUC


# Drift Detector
# S: Source (Old Data)
# T: Target (New Data)
# ST: S&T combined
def drift_detector(S,T,threshold = 0.75):
    T = pd.DataFrame(T)
    S = pd.DataFrame(S)
    # Give slack variable in_target which is 1 for old and 0 for new
    T['in_target'] = 0 # in target set
    S['in_target'] = 1 # in source set
    # Combine source and target with new slack variable 
    ST = pd.concat( [T, S], ignore_index=True, axis=0)
    labels = ST['in_target'].values
    ST = ST.drop('in_target', axis=1).values
    # You can use any classifier for this step. We advise it to be a simple one as we want to see whether source
    # and target differ not to classify them.
    clf = LogisticRegression(solver='liblinear')
    predictions = np.zeros(labels.shape)
    # Divide ST into two equal chunks
    # Train LR on a chunk and classify the other chunk
    # Calculate AUC for original labels (in_target) and predicted ones
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    for train_idx, test_idx in skf.split(ST, labels):
        X_train, X_test = ST[train_idx], ST[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        predictions[test_idx] = probs
    auc_score = AUC(labels, predictions)
    # Signal drift if AUC is larger than the threshold
    if auc_score > threshold:
        return True
    else:
        return False

