from keras.datasets import mnist    
from scipy.io import loadmat
from models.utils import ResNet18, split_dataframe, detect_drifts
from skimage.transform import resize
import keras
import numpy as np
import models.unmasking_drift_detector as unm
import models.d3_drift_detector as d3
import matplotlib.pyplot as plt


# load the dataset we'll generate the drift with
# resize the images from its original shape so that it will fit in the neural network 
(mnist_images, _), (_, _) = mnist.load_data()
mnist_images = resize(mnist_images, (mnist_images.shape[0],32,32,3))

# load the main dataset
svhn_data = loadmat('/realWorld/train_32x32.mat')
svhn_images, svhn_labels = svhn_data['X'], svhn_data['y']
svhn_images = np.moveaxis(svhn_images, -1, 0)
svhn_labels -= 1  # normally, svhn labels range from 1 to 10, so we solve this issue for later

# define a cap so that we can choose just a percentage of the whole data to train/validate the neural network
cap = svhn_images.shape[0] // 2

# split the dataset in train/test values
idx = int(0.8 * cap)
X_train, X_test = svhn_images[:idx], svhn_images[idx:cap]
y_train, y_test = svhn_labels[:idx], svhn_labels[idx:cap]

# Initialize, compile and fit the ResNet18 neural network
model = ResNet18(input_shape=(32, 32, 3), n_classes=10)
model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_test, y_test)) 
# Epoch 1/3
# 458/458 [==============================] - 390s 843ms/step - loss: 1.0530 - accuracy: 0.6609 - val_loss: 0.8744 - val_accuracy: 0.7341
# Epoch 2/3
# 458/458 [==============================] - 380s 831ms/step - loss: 0.4994 - accuracy: 0.8451 - val_loss: 1.3445 - val_accuracy: 0.6866
# Epoch 3/3
# 458/458 [==============================] - 374s 818ms/step - loss: 0.4001 - accuracy: 0.8756 - val_loss: 0.5081 - val_accuracy: 0.8440


# Next, we generate drift over the SVHN dataset.
# The drifted dataset will have alternating batches of [SVHN] and [2/3 SVHN + 1/3 MNIST]
# The length of each batch is also flexible
drifted_data = np.empty((0, 32, 32, 3))
i = 0
last_index = 0
current_index = 0
ground_truth_drifts = []
dimension_limit = 0.6 * (svhn_images.shape[0] + 0.5*mnist_images.shape[0])

while drifted_data.shape[0] < dimension_limit:
    current_index += np.random.choice([400, 800, 1200, 1600])
    if i % 2 == 0:
        if i != 0:
            ground_truth_drifts.append(drifted_data.shape[0])
        svhn_batch = svhn_images[3*last_index:3*current_index]
        drifted_data = np.concatenate((drifted_data, svhn_batch), axis=0)
    else:
        ground_truth_drifts.append(drifted_data.shape[0])
        svhn_batch = svhn_images[2*last_index:2*current_index]
        batch = np.concatenate((svhn_batch, mnist_images[last_index:current_index]), axis=0)
        np.random.shuffle(batch)
        drifted_data = np.concatenate((drifted_data, batch), axis=0)
    i += 1
    last_index = current_index

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