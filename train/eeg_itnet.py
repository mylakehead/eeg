import os

from data.seed_iv import Subject
from pre.eeg_itnet import process
from model.eeg_itnet import train


def start(config):
    eeg_raw_data_path = os.path.join(
        config.eeg_it_net['dataset']['root'],
        config.eeg_it_net['dataset']['eeg_raw_data']
    )

    (
        s1_train, s2_train, s2_test, s3_train, s3_test, c_s1_train_label,
        c_s2_train_label, c_s2_test_label, c_s3_train_label, c_s3_test_label
    ) = process(eeg_raw_data_path, Subject.THREE)

    train(
        s1_train, s2_train, s2_test, s3_train, s3_test, c_s1_train_label,
        c_s2_train_label, c_s2_test_label, c_s3_train_label, c_s3_test_label, Subject.THREE.value
    )
