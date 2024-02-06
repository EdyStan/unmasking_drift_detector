import pandas as pd
import models.unmasking_drift_detector as unm
import models.d3_drift_detector as d3
from models.utils import split_dataframe, detect_drifts


# Load data into pd.DataFrame and then split it in chunks of 1500 elements each.
elec2_data = pd.read_csv('driftDatasets/realWorld/Elec2/elec2_data.csv')
elec2_labels = pd.read_csv('driftDatasets/realWorld/Elec2/elec2_label.csv')
elec2_data = pd.concat([elec2_data, elec2_labels], axis=1)
data_splitted = split_dataframe(elec2_data, 1500)

# Apply the drift detectors over the datasets. Store the results for a final comparison
summary_d3 = detect_drifts(data_splitted, d3.drift_detector)
summary_unm = detect_drifts(data_splitted, unm.drift_detector, min_feature_percentage_remaining=0.3)

# Print and compare the final results: average drift gap and maximum drift gap
# gap = the number of chunks between two drift detections
print("d3", summary_d3, "unmasking", summary_unm, sep='\n')

# Results:
# d3
# Average drift gap: 2.25, Maximum drift gap: 4
# unmasking
# Average drift gap: 2.076923076923077, Maximum drift gap: 4