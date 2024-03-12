import numpy as np
import tensorflow as tf
from keras import layers


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
def detect_drifts(data_splitted, drift_detector, true_drifts, beta=0.5, accepted_error=0, show_drifts=False, **kwargs):
    past_data = data_splitted[0]
    detected_drifts = set()
    true_drifts = set(true_drifts)
    for i in range(len(data_splitted)):
        if drift_detector(past_data, data_splitted[i], **kwargs):
            past_data = data_splitted[i]
            if accepted_error < i and accepted_error != 0:
                for err in range(accepted_error + 1):
                    if i-err in true_drifts:
                        detected_drifts.add(i-err)
                        break
                    elif i+err in true_drifts:
                        detected_drifts.add(i+err)
                        break
                else:
                    detected_drifts.add(i)
            else:
                detected_drifts.add(i)
            if show_drifts:
                print(f"Change detected at index {i}")
        else:
            past_data = (1 - beta) * past_data + beta * data_splitted[i]
    inter = len(detected_drifts.intersection(true_drifts))
    union = len(detected_drifts.union(true_drifts))
    jaccard_sim = inter/union
    return jaccard_sim


# ResNet18 architecture

# Residual block. Introduces shortcut connections that allow the gradient to flow more easily during training.
# Such structures represent a solution to the vanishing gradient problem (gradients become extremely small 
# as they are propagated backward through the layers)
def _residual_block(x, filters, strides):
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


def ResNet18(input_shape, n_classes):
    # input shape of the model
    input_tensor = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = _residual_block(x, filters=64, strides=(1, 1))
    x = _residual_block(x, filters=64, strides=(1, 1))
    x = _residual_block(x, filters=128, strides=(2, 2))
    x = _residual_block(x, filters=128, strides=(1, 1))
    x = _residual_block(x, filters=256, strides=(2, 2))
    x = _residual_block(x, filters=256, strides=(1, 1))
    x = _residual_block(x, filters=512, strides=(2, 2))
    x = _residual_block(x, filters=512, strides=(1, 1))

    # Collapses the spatial dimensions to a single value per channel
    # Since this is the layer we want to extract the features from, we skip creating a dense layer
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    output_tensor = layers.Dense(n_classes, activation='softmax')(x)

    # Create and inspect the model
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model