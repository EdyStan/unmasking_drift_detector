import numpy as np


# Splits data in chunks of an arbitrary number of instances. 
def split_dataframe(df, chunk_size = 1000): 
    chunks = list()
    num_chunks = len(df) // chunk_size
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return np.array(chunks)


# Calls the chosen drift detector.
# In order to determine if there's a drift, we compare the i'th chunk with 
# the mean of all chunks from the last drift detected, until the (i-1)'th chunk.
def detect_drifts(data_splitted, drift_detector, show_drifts=False, **kwargs):
    past_data = data_splitted[0]
    drift_index = 0
    drift_gaps = []
    for i in range(len(data_splitted)):
        if drift_detector(past_data, data_splitted[i], **kwargs):
            drift_gaps.append(drift_index)
            drift_index = 1
            past_data = data_splitted[i]
            if show_drifts:
                print(f"Change detected at index {i}")
        else:
            past_data = (drift_index * past_data + data_splitted[i]) / (drift_index + 1)
            drift_index += 1
    return f"Average drift gap: {np.average(drift_gaps)}, Maximum drift gap: {np.max(drift_gaps)}"