import os

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

import tensorflow as tf

from data.seed_iv import Subject
from pre.eeg_itnet import process
from model.eeg_itnet import inception_temporal_convolutional_net


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


def start(config):
    eeg_raw_data_path = os.path.join(
        config.dataset['root_path'],
        config.dataset['eeg_raw_data_path']
    )

    (
        s1_train, s2_train, s2_test, s3_train, s3_test, c_s1_train_label,
        c_s2_train_label, c_s2_test_label, c_s3_train_label, c_s3_test_label
    ) = process(eeg_raw_data_path, Subject.THREE)

    train(
        s1_train, s2_train, s2_test, s3_train, s3_test, c_s1_train_label,
        c_s2_train_label, c_s2_test_label, c_s3_train_label, c_s3_test_label, Subject.THREE.value
    )

    raw_processed_path = config.dataset['raw_processed_path']

    subjects = [
        Subject.ONE, Subject.TWO, Subject.THREE, Subject.FOUR, Subject.FIVE, Subject.SIX, Subject.SEVEN,
        Subject.EIGHT, Subject.NINE, Subject.TEN, Subject.ELEVEN, Subject.TWELVE, Subject.THIRTEEN,
        Subject.FOURTEEN, Subject.FIFTEEN
    ]
    sessions = [
        Session.ONE, Session.TWO, Session.THREE
    ]
    train_trials = list(range(0, 16))
    test_trails = list(range(16, 24))
    batch_size = 200

    best_accuracy = 0.8

    model = Conformer(emb_size=40, inner_channels=40, heads=10, depth=6, n_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    model.to(device)

    train_dataset, train_labels = get_raw_dataset(
        raw_processed_path,
        subjects,
        sessions,
        train_trials
    )
    test_dataset, test_labels = get_raw_dataset(
        raw_processed_path,
        subjects,
        sessions,
        test_trails
    )

