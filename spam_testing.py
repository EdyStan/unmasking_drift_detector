import models.unmasking_drift_detector as unm
import models.d3_drift_detector as d3
from models.utils import split_dataframe, detect_drifts
from sklearn.datasets import load_svmlight_file


# Load data into np.array and then split it in chunks of 25 elements each.
data = load_svmlight_file("driftDatasets/realWorld/spam/spam.libsvm")
spam_data = data[0].toarray()
data_splitted = split_dataframe(spam_data, 25)

# Apply the drift detectors over the datasets. Store the results for a final comparison
summary_d3 = detect_drifts(data_splitted, d3.drift_detector)
summary_unm = detect_drifts(data_splitted, unm.drift_detector, min_feature_percentage_remaining=0.15)

# Print and compare the final results: average drift gap and maximum drift gap
# gap = the number of chunks between two drift detections
print("d3", summary_d3, "unmasking", summary_unm, sep='\n')

# Results:
# d3
# Average drift gap: 1.189102564102564, Maximum drift gap: 19
# unmasking
# Average drift gap: 1.1450617283950617, Maximum drift gap: 8