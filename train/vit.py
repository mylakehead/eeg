import numpy as np

import scipy.io as sio

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LSTM, MaxPooling3D, BatchNormalization, Reshape

from data.seed_iv import subject_file_map, FeatureMethod, session_label, Session, Subject


def split(array, block_size, index):
    current_length = array.shape[index]

    pad_length = block_size - (current_length % block_size) if current_length % block_size != 0 else 0
    total_length = current_length + pad_length

    if pad_length > 0:
        if index == 0:
            pad_values = array[:pad_length, :, :]
        elif index == 1:
            pad_values = array[:, :pad_length, :]
        elif index == 2:
            pad_values = array[:, :, :pad_length]
        else:
            raise IndexError
        padded_array = np.concatenate((pad_values, array), axis=index)
    else:
        padded_array = array

    if index == 0:
        blocks = np.split(padded_array[:total_length, :, :], total_length // block_size, axis=index)
    elif index == 1:
        blocks = np.split(padded_array[:, :total_length, :], total_length // block_size, axis=index)
    elif index == 2:
        blocks = np.split(padded_array[:, :, :total_length], total_length // block_size, axis=index)
    else:
        raise IndexError

    return blocks


def collect_data(data_path, subjects, sessions, trails, method, sample_length):
    label_dict = dict()
    for subject in subjects:
        for session in sessions:
            subject_file_mapping = subject_file_map(data_path)
            files = subject_file_mapping[subject]
            file = files[session]

            data = sio.loadmat(file)
            data.pop('__header__', None)
            data.pop('__version__', None)
            data.pop('__globals__', None)

            for trail in trails:
                if method == FeatureMethod.DE_LDS:
                    pattern = f'de_LDS{trail+1}'
                elif method == FeatureMethod.DE_MOVING_AVE:
                    pattern = f'de_movingAve{trail+1}'
                elif method == FeatureMethod.PSD_LDS:
                    pattern = f'psd_LDS{trail+1}'
                elif method == FeatureMethod.PSD_MOVING_AVE:
                    pattern = f'psd_movingAve{trail+1}'
                else:
                    raise Exception("feature method error")

                found = False
                for key, trail_data in data.items():
                    if key != pattern:
                        continue

                    label = session_label[session][trail]
                    if label not in label_dict:
                        label_dict[label] = trail_data
                    else:
                        label_dict[label] = np.concatenate((label_dict[label], trail_data), axis=1)

                    found = True

                if not found:
                    raise ModuleNotFoundError

    dataset = []
    labels = []
    for k, v in label_dict.items():
        chunks = split(v, sample_length, 1)
        dataset.extend(chunks)
        labels.extend([k.value] * len(chunks))

    return np.array(dataset), np.array(labels)


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def vit_model():
    image_size = [15, 15]
    patch_size = 3
    num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
    projection_dim = 64
    transformer_layers = 8
    num_heads = 8
    num_classes = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]
    mlp_head_units = [2048, 1024]

    conv_block = keras.models.Sequential([
        Input(shape=(62, 250, 5)),
        Conv2D(filters=64, kernel_size=3, padding='same', strides=(1, 2)),
        Conv2D(filters=64, kernel_size=3, padding='same', strides=(1, 2)),
        Conv2D(filters=128, kernel_size=3, padding='same'),
        MaxPooling2D(pool_size=2, strides=2),
        Dropout(rate=0.3),
        Conv2D(filters=128, kernel_size=3, padding='same'),
        Conv2D(filters=256, kernel_size=3, padding='same'),
        Conv2D(filters=512, kernel_size=3, padding='same'),
        MaxPooling2D(pool_size=2, strides=2),
        Dropout(rate=0.2),
    ])

    inputs = layers.Input(shape=(62, 250, 5))
    features = conv_block(inputs)

    patches = Patches(patch_size)(features)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)

        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    logit = layers.Dense(num_classes)(features)
    outputs = layers.Softmax()(logit)

    model_vit = keras.Model(inputs=inputs, outputs=outputs)

    return model_vit


def start(config):
    subjects = [
        Subject.ONE, Subject.TWO, Subject.THREE, Subject.FOUR, Subject.FIVE, Subject.SIX, Subject.SEVEN,
        Subject.EIGHT, Subject.NINE, Subject.TEN, Subject.ELEVEN, Subject.TWELVE, Subject.THIRTEEN,
        Subject.FOURTEEN, Subject.FIFTEEN
    ]
    sessions = [
        Session.ONE, Session.TWO, Session.THREE
    ]
    sample_length = 250
    train_trials = list(range(0, 16))
    test_trails = list(range(16, 24))
    method = FeatureMethod.DE_LDS
    batch_size = 200

    train_dataset, train_labels = collect_data(
        config.dataset['eeg_feature_smooth_abs_path'],
        subjects,
        sessions,
        train_trials,
        method,
        sample_length
    )
    test_dataset, test_labels = collect_data(
        config.dataset['eeg_feature_smooth_abs_path'],
        subjects,
        sessions,
        test_trails,
        method,
        sample_length
    )

    train_labels_reshaped = train_labels.reshape(-1, 1)
    test_labels_reshaped = test_labels.reshape(-1, 1)
    train_labels_reshaped = keras.utils.to_categorical(train_labels_reshaped, 4)
    test_labels_reshaped = keras.utils.to_categorical(test_labels_reshaped, 4)

    model = keras.models.Sequential([
        Conv2D(filters=64, kernel_size=5, input_shape=(62, 250, 5), padding='same'),
        # kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)),
        Conv2D(filters=64, kernel_size=3, padding='same'),
        Conv2D(filters=64, kernel_size=3, padding='same'),
        MaxPooling2D(pool_size=2, strides=2),
        # BatchNormalization(),
        Dropout(rate=0.3),
        Conv2D(filters=128, kernel_size=3, padding='same'),
        # kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)),
        Conv2D(filters=128, kernel_size=3, padding='same'),
        MaxPooling2D(pool_size=2, strides=2),
        # BatchNormalization(),
        Dropout(rate=0.2),

        Conv2D(filters=256, kernel_size=3, padding='same'),
        # kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)),
        MaxPooling2D(pool_size=2, strides=2),
        # BatchNormalization(),
        Dropout(rate=0.25),

        Conv2D(filters=512, kernel_size=3, padding='same'),
        MaxPooling2D(pool_size=2, strides=2),
        # BatchNormalization(),
        Dropout(rate=0.3),

        Flatten(),

        Dense(512, activation='relu'),
        # kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01)),
        # BatchNormalization(),
        Dropout(rate=0.4),
        Dense(256, activation='relu'),
        # BatchNormalization(),
        Dropout(rate=0.2),
        Dense(64, activation='relu'),
        # BatchNormalization(),
        Dense(4, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.002), loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy', tf.keras.metrics.RootMeanSquaredError()])
    model.summary()

    reduce_lr_cnn = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5,
                                      min_lr=1e-7)  # patience = 5 and factor = 0.9

    history = model.fit(
        train_dataset,
        train_labels_reshaped,
        batch_size=64,
        epochs=100,
        validation_data=(test_dataset, test_labels_reshaped),
        callbacks=[]
    )

    return
    model_rnn = keras.models.Sequential([
        LSTM(units=64, activation='tanh', input_shape=[1250, 62], return_sequences=True),
        Dropout(0.25),
        LSTM(units=128, activation='tanh', return_sequences=True),
        Dropout(0.3),
        LSTM(units=128, activation='tanh', return_sequences=True),
        Dropout(0.35),
        LSTM(units=256, activation='tanh', return_sequences=True),
        Flatten(),

        Dense(256, activation='relu'),
        Dropout(0.25),
        Dense(128, activation='relu'),
        Dropout(0.35),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax'),

    ])

    model_rnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003), loss=keras.losses.categorical_crossentropy,
                      metrics=['accuracy', tf.keras.metrics.RootMeanSquaredError()])
    model_rnn.summary()

    rnn_train = train_dataset.reshape(106, 62, -1)  # Formatting for RNN input
    rnn_test = test_dataset.reshape(48, 62, -1)
    rnn_train = np.transpose(rnn_train, (0, 2, 1))
    rnn_test = np.transpose(rnn_test, (0, 2, 1))
    reduce_lr_rnn = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5,
                                      min_lr=1e-6)  # patience = 5 and factor = 0.9
    history_rnn = model_rnn.fit(
        rnn_train,
        train_labels_reshaped,
        batch_size=64,
        epochs=60,
        validation_data=(rnn_test, test_labels_reshaped),
        callbacks=[]
    )

    return

    model_hybrid = keras.models.Sequential([
        Conv2D(filters=64, kernel_size=3, input_shape=(62, 250, 5), padding='same'),
        Conv2D(filters=128, kernel_size=3, padding='same'),
        Conv2D(filters=128, kernel_size=3, padding='same'),
        MaxPooling2D(pool_size=2, strides=2),
        BatchNormalization(),

        Conv2D(filters=256, kernel_size=3, padding='same'),
        Conv2D(filters=256, kernel_size=3, padding='same'),
        MaxPooling2D(pool_size=2, strides=2),
        BatchNormalization(),

        Conv2D(filters=512, kernel_size=3, padding='same'),
        BatchNormalization(),

        Reshape((62, 15 * 512), input_shape=(15, 62, 512)),

        LSTM(units=128, activation='tanh', return_sequences=True),
        BatchNormalization(),
        LSTM(units=256, activation='tanh', return_sequences=True),
        BatchNormalization(),
        LSTM(units=256, activation='tanh', return_sequences=True),
        BatchNormalization(),
        LSTM(units=512, activation='tanh', return_sequences=True),
        BatchNormalization(),

        Flatten(),

        Dense(512, activation='relu'),
        Dropout(0.25),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model_hybrid.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                         loss=keras.losses.categorical_crossentropy,
                         metrics=['accuracy', tf.keras.metrics.RootMeanSquaredError()])
    model_hybrid.summary()

    reduce_lr_hybrid = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6)

    history_hybrid = model_hybrid.fit(
        train_dataset,
        train_labels_reshaped,
        batch_size=32,
        epochs=60,
        validation_data=(test_dataset, test_labels_reshaped),
        callbacks=[reduce_lr_hybrid]
    )

    return


    vit = vit_model()
    vit.summary()

    learning_rate = 1e-3
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    vit.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            metrics.CategoricalAccuracy(name="accuracy"),
        ],
    )

    '''
    checkpoint_filepath = "./checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    '''
    reduce_lr_exp = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=4, min_lr=1e-6)

    num_epochs = 150
    history = vit.fit(
        x=train_dataset,
        y=train_labels_reshaped,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[], # callbacks=[checkpoint_callback, reduce_lr_exp],
    )

    # vit.load_weights(checkpoint_filepath)
    _, accuracy = vit.evaluate(test_dataset, test_labels_reshaped)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history
