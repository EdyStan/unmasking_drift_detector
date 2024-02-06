import pandas as pd
import models.unmasking_drift_detector as unm
import models.d3_drift_detector as d3
from models.utils import split_dataframe, detect_drifts

# Load data into pd.DataFrame and then split it in chunks of 2000 elements each.
covtype_data = pd.read_csv('driftDatasets/realWorld/covType/covType.csv')
data_splitted = split_dataframe(covtype_data, 2000)

# Apply the drift detectors over the datasets. Store the results for a final comparison
summary_d3 = detect_drifts(data_splitted, d3.drift_detector)
summary_unm = detect_drifts(data_splitted, unm.drift_detector, min_feature_percentage_remaining=0.35)

# Print and compare the final results: average drift gap and maximum drift gap
# gap = the number of chunks between two drift detections
print("d3", summary_d3, "unmasking", summary_unm, sep='\n')

# Results:
# d3
# Average drift gap: 2.919191919191919, Maximum drift gap: 8
# unmasking
# Average drift gap: 2.5350877192982457, Maximum drift gap: 8