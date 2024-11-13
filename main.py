import os
import sys
import json
import getopt

import torch

from train.conformer import start


class Config:
    def __init__(self, data: dict):
        self.dataset = data['dataset']
        self.dataset['eeg_raw_data_abs_path'] = os.path.join(
            data['dataset']['root_path'],
            data['dataset']['eeg_raw_data_path']
        )
        self.dataset['eeg_feature_smooth_abs_path'] = os.path.join(
            data['dataset']['root_path'],
            data['dataset']['eeg_feature_smooth_path']
        )


def parse_opt(argv: list[str]) -> str:
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


def parse_config(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        data = json.load(f)
        return data


def main(argv):
    config_file = parse_opt(argv)
    data = parse_config(config_file)
    config = Config(data)

    start(config)


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    main(sys.argv[1:])
