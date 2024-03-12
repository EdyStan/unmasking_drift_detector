from keras.datasets import cifar100    
from models.utils import ResNet18, split_dataframe, detect_drifts
from scipy.io import loadmat
import keras
import numpy as np
import models.unmasking_drift_detector as unm
import models.d3_drift_detector as d3
import matplotlib.pyplot as plt


# load the main dataset
svhn_data = loadmat('/realWorld/train_32x32.mat')
svhn_images, svhn_labels = svhn_data['X'], svhn_data['y']
svhn_images = np.moveaxis(svhn_images, -1, 0)
svhn_labels -= 1  # normally, svhn labels range from 1 to 10, so we solve this issue for later

# select only the images that correspond to the first 5 labels
train_indices = np.where(np.isin(svhn_labels, range(5)))[0]

# define a cap so that we can choose just a percentage of the whole data to train/validate the neural network
cap = len(train_indices)

# split the dataset in train/test values
idx = int(0.8 * cap)
X_train, X_test = svhn_images[train_indices][:idx], svhn_images[train_indices][idx:]
y_train, y_test = svhn_labels[train_indices][:idx], svhn_labels[train_indices][idx:]

# Initialize, compile and fit the ResNet18 neural network
model = ResNet18(input_shape=(32, 32, 3), n_classes=50)
model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_test, y_test))


# define the unseen data as the images corresponding to the last 50 labels. 
# these images were not used to fit the model.
unseen_data_indices = np.where(np.isin(svhn_labels, range(5, 10)))[0]
unseen_data = svhn_images[unseen_data_indices]


# Next, we generate drift over the dataset.
# The drifted dataset will have alternating batches of unseen_data and X_train
# The length of each batch is also flexible
drifted_data = np.empty((0, 32, 32, 3))
i = 0
last_index = 0
current_index = 0
ground_truth_drifts = []

dimension_limit = 0.7 * svhn_images.shape[0]
while drifted_data.shape[0] < dimension_limit:
    if i % 2 == 0:
        current_index += np.random.choice([400, 800, 1200, 1600])
        if i != 0:
            ground_truth_drifts.append(drifted_data.shape[0])
        known_batch = X_train[last_index:current_index]
        drifted_data = np.concatenate((drifted_data, known_batch), axis=0)
    else:
        ground_truth_drifts.append(drifted_data.shape[0])
        batch = unseen_data[last_index:current_index]
        np.random.shuffle(batch)
        drifted_data = np.concatenate((drifted_data, batch), axis=0)
        last_index = current_index
    i += 1


# create an auxiliary model so we can extract data from the avg_pool layer of the already trained neural network
# extract features from the drifted dataset
extract_features_model = keras.Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
features = extract_features_model.predict(drifted_data)

# split the dataset in reasonable sized batches
# modify the true drifts so that they are assigned to the current batches
batch_size = 400
data_splitted = split_dataframe(features, batch_size)
ground_truth_drifts_batch = np.array(ground_truth_drifts) / batch_size
ground_truth_drifts_batch = ground_truth_drifts_batch.astype(int)


# Detect drifts using D3 algorithm
print("D3:")
sim_d3 = detect_drifts(data_splitted, d3.drift_detector, ground_truth_drifts_batch, accepted_error=1, show_drifts=False)
print(sim_d3)

# Perform grid search over the unmasking algorithm.
thresholds = [0.5, 0.53, 0.54, 0.55, 0.56, 0.57, 0.6]
jaccard_sim_unm = []

print("Unmasking:")
for thr in thresholds:
    print(f"Threshold = {thr}")
    sim_unm = detect_drifts(data_splitted, unm.drift_detector, ground_truth_drifts_batch, show_drifts=False, accepted_error=1, threshold=thr, min_feature_percentage_remaining=0.05)
    print(sim_unm)
    jaccard_sim_unm.append(sim_unm)



# Plot the results
plt.xlabel('Threshold')
plt.ylabel('Jaccard Similarity')
plt.title('Drift detection over SVHN + MNIST dataset')

plt.scatter(thresholds, jaccard_sim_unm, marker='o', color='red', label='accepted_error=1')
plt.grid()
plt.legend()
plt.savefig('/images/svhn_jaccard_error1.png')
plt.show()