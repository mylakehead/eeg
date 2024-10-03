import os
import sys
import json
import getopt

from data.seed_iv import Subject
from pre.seed_iv import pre_process
from model.eeg_itnet import train


class Config:
    def __init__(self, eeg_it_net):
        self.eeg_it_net = eeg_it_net


def parse_opt(argv):
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config="])
        for opt, arg in opts:
            if opt == '-h':
                print('main.py -c <config_file>')
                sys.exit()
            elif opt in ("-c", "--config"):
                return arg
            else:
                print('main.py -c <config_file>')
                sys.exit()
    except getopt.GetoptError:
        print('main.py -c <config_file>')
        sys.exit(2)


def parse_config(config_file):
    with open(config_file, 'r') as f:
        data = json.load(f)
        config = Config(data['EEG-ITNet'])
        return config


def main(argv):
    config_file = parse_opt(argv)
    config = parse_config(config_file)

    eeg_raw_data_path = os.path.join(config.eeg_it_net['dataset']['root'], config.eeg_it_net['dataset']['eeg_raw_data'])

    (
        s1_train, s2_train, s2_test, s3_train, s3_test, c_s1_train_label,
        c_s2_train_label, c_s2_test_label, c_s3_train_label, c_s3_test_label
    ) = pre_process(eeg_raw_data_path, Subject.THREE)

    train(
        s1_train, s2_train, s2_test, s3_train, s3_test, c_s1_train_label,
        c_s2_train_label, c_s2_test_label, c_s3_train_label, c_s3_test_label, Subject.THREE.value
    )


if __name__ == '__main__':
    main(sys.argv[1:])
