import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from scipy.io import loadmat
import models.unmasking_drift_detector as unm
import models.d3_drift_detector as d3
from models.utils import split_dataframe, detect_drifts

# Load data into np.array
data = loadmat('/home/edstan/Desktop/master_AI/practice/driftDatasets/realWorld/train_32x32.mat')
images= data['X']
images = np.moveaxis(images, -1, 0)


# ResNet18 architecture

# Residual block. Introduces shortcut connections that allow the gradient to flow more easily during training.
# Such structures represent a solution to the vanishing gradient problem (gradients become extremely small 
# as they are propagated backward through the layers)
def residual_block(x, filters, strides):
    shortcut = x

    # First convolutional layer of the residual block. The output is supposed to have the following shape: 
    # (n, N, N, `filters`), where
    # n = the number of input instances
    # N = 1 + [(`input_volume` - `kernel_size` + 2*`padding`) / `strides`]
    # `padding='same'` adjusts padding such that the output has the same size as the input
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same')(x)
    # re-centre and re-scale data
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # `filters` need to match in order to execute the shortcut connection. In addition, coincidentally, `strides` change to 
    # the value of (2,2), when the filters increase.
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add the current layer with the shortcut layer to ensure a better gradient flow
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

# input shape of the model
input_tensor = tf.keras.Input(shape=(32,32, 3))

x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(input_tensor)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

x = residual_block(x, filters=64, strides=(1, 1))
x = residual_block(x, filters=64, strides=(1, 1))
x = residual_block(x, filters=128, strides=(2, 2))
x = residual_block(x, filters=128, strides=(1, 1))
x = residual_block(x, filters=256, strides=(2, 2))
x = residual_block(x, filters=256, strides=(1, 1))
x = residual_block(x, filters=512, strides=(2, 2))
x = residual_block(x, filters=512, strides=(1, 1))

# Collapses the spatial dimensions to a single value per channel
# Since this is the layer we want to extract the features from, we skip creating a dense layer
x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

# Create and inspect the model
model = tf.keras.Model(inputs=input_tensor, outputs=x)
model.summary()

# Extract features from avg_pool layer and split them in chunks of 70 elements each.
features = model.predict(images)
data_splitted = split_dataframe(features, 70)

# Apply the drift detectors over the datasets. Store the results for a final comparison
summary_d3 = detect_drifts(data_splitted, d3.drift_detector)
summary_unm = detect_drifts(data_splitted, unm.drift_detector, min_feature_percentage_remaining=0.5)

# Print and compare the final results: average drift gap and maximum drift gap
# gap = the number of chunks between two drift detections
print("d3", summary_d3, "unmasking", summary_unm, sep='\n')

# Results:
# d3
# Average drift gap: 43.208333333333336, Maximum drift gap: 121
# unmasking
# Average drift gap: 50.45, Maximum drift gap: 111