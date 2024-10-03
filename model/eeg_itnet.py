import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Add, Dropout, BatchNormalization, DepthwiseConv2D
from tensorflow.keras.layers import Activation, Concatenate, Dense, Flatten, AveragePooling2D
from tensorflow.keras.constraints import MaxNorm


class EEGItNetModel:
    # Inception Temporal Convolutional Network
    def __init__(self, chans, samples):
        self.chans = chans
        self.samples = samples
        self.frequency_filters = [2, 4, 8]
        self.spatial_filters = [1, 1, 1]

    def build(self):
        input_block = Input(shape=(self.chans, self.samples, 1))

        block_1 = Conv2D(
            self.frequency_filters[0], (1, 16), use_bias=False, activation='linear', padding='same'
        )(input_block)
        block_1 = BatchNormalization()(block_1)
        block_1 = DepthwiseConv2D(
            (self.chans, 1), use_bias=False, padding='valid', depth_multiplier=self.spatial_filters[0],
            activation='linear', depthwise_constraint=MaxNorm(max_value=1), name='Spatial_filter_1'
        )(block_1)
        block_1 = BatchNormalization()(block_1)
        # TODO
        _ = Activation('elu')(block_1)

        return


def inception_temporal_convolutional_net(chans, samples):
    drop_rate = 0.4
    n_ff = [2, 4, 8]
    n_sf = [1, 1, 1]

    input_block = Input(shape=(chans, samples, 1))

    block1 = Conv2D(n_ff[0], (1, 16), use_bias=False, activation='linear', padding='same',
                    name='Spectral_filter_1')(input_block)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((chans, 1), use_bias=False, padding='valid', depth_multiplier=n_sf[0], activation='linear',
                             depthwise_constraint=MaxNorm(max_value=1),
                             name='Spatial_filter_1')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)

    block2 = Conv2D(n_ff[1], (1, 32), use_bias=False, activation='linear', padding='same',
                    name='Spectral_filter_2')(input_block)
    block2 = BatchNormalization()(block2)
    block2 = DepthwiseConv2D((chans, 1), use_bias=False, padding='valid', depth_multiplier=n_sf[1], activation='linear',
                             depthwise_constraint=MaxNorm(max_value=1),
                             name='Spatial_filter_2')(block2)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)

    block3 = Conv2D(n_ff[2], (1, 64), use_bias=False, activation='linear', padding='same',
                    name='Spectral_filter_3')(input_block)
    block3 = BatchNormalization()(block3)
    block3 = DepthwiseConv2D((chans, 1), use_bias=False, padding='valid', depth_multiplier=n_sf[2], activation='linear',
                             depthwise_constraint=MaxNorm(max_value=1),
                             name='Spatial_filter_3')(block3)
    block3 = BatchNormalization()(block3)
    block3 = Activation('elu')(block3)

    block = Concatenate(axis=-1)([block1, block2, block3])

    block = AveragePooling2D((1, 4))(block)
    block_in = Dropout(drop_rate)(block)

    paddings = [[0, 0], [0, 0], [3, 0], [0, 0]]
    block = tf.keras.ops.pad(block_in, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = tf.keras.ops.pad(block, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_in, block])

    paddings = [[0, 0], [0, 0], [6, 0], [0, 0]]
    block = tf.keras.ops.pad(block_out, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = tf.keras.ops.pad(block, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_out, block])

    paddings = [[0, 0], [0, 0], [12, 0], [0, 0]]
    block = tf.keras.ops.pad(block_out, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = tf.keras.ops.pad(block, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_out, block])

    paddings = [[0, 0], [0, 0], [24, 0], [0, 0]]
    block = tf.keras.ops.pad(block_out, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = tf.keras.ops.pad(block, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_out, block])

    block = block_out

    block = Conv2D(28, (1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)

    block = AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(block)

    block = Dropout(drop_rate)(block)
    embedded = Flatten()(block)

    out_class = 4
    out = Dense(out_class, activation='softmax', kernel_constraint=MaxNorm(0.25))(embedded)

    model = Model(inputs=input_block, outputs=out)
    return model


def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()


def train(s1_train, s2_train, s2_test, s3_train, s3_test, c_s1_train_label, c_s2_train_label,
          c_s2_test_label, c_s3_train_label, c_s3_test_label, participant):
    chans = 62
    samples = 700

    model1 = inception_temporal_convolutional_net(chans, samples)
    model2 = inception_temporal_convolutional_net(chans, samples)
    for layer in model2.layers[-4:]:
        layer.trainable = False

    model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=600, mode='min', verbose=1, restore_best_weights=True
    )

    model_filename = f"model1_participant_{participant}.keras"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_filename, monitor='val_loss', save_best_only=True, mode='min'
    )

    callbacks = [early_stopping, model_checkpoint]

    print("model1.summary()", model1.summary())

    print("--------------- model1 training ----------------")
    print(s1_train.shape, c_s1_train_label.shape)
    _ = model1.fit(
        s1_train, c_s1_train_label, epochs=300, validation_split=0.1, callbacks=callbacks, batch_size=128
    )

    print("--------------- model2 training ----------------")
    model2.set_weights(model1.get_weights())

    categorical_labels_train = np.append(c_s2_train_label, c_s3_train_label, axis=0)
    print("categorical_labels_train shape: ", categorical_labels_train.shape)

    train_set = np.append(s2_train, s3_train, axis=0)
    print("ChunkData_train shape: ", train_set.shape)

    print("model2.summary()", model2.summary())
    fitted2 = model2.fit(train_set, categorical_labels_train, epochs=300, validation_split=0.1,
                         callbacks=callbacks,
                         batch_size=128)
    plot_training_history(fitted2)

    # Define the EarlyStopping callback and mode save
    _ = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True)
    # Define the ModelCheckpoint callback to save the best model during training
    model_filename = f"trained_model2_participant_{participant}.keras"
    _ = tf.keras.callbacks.ModelCheckpoint(model_filename, monitor='val_loss', save_best_only=True, mode='min')

    categorical_labels_tune = np.append(c_s2_test_label, c_s3_test_label, axis=0)
    print("categorical_labels_tune shape: ", categorical_labels_tune.shape)
    test_set = np.append(s2_test, s3_test, axis=0)
    print("test_set shape: ", test_set.shape)

    predicted = model2.predict(x=test_set)

    true_labels_multiclass = np.argmax(categorical_labels_tune, axis=1)
    predicted_labels_multiclass = np.argmax(predicted, axis=1)

    accuracy = metrics.accuracy_score(true_labels_multiclass, predicted_labels_multiclass)
    print("Accuracy:", accuracy)

    f1_score = metrics.f1_score(true_labels_multiclass, predicted_labels_multiclass, average='weighted')
    print("F1-score:", f1_score)

    classification_report = metrics.classification_report(true_labels_multiclass, predicted_labels_multiclass)
    print("Classification Report:", classification_report)
