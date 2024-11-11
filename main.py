import os
import sys
import json
import getopt

import torch


class Config:
    def __init__(self, data: dict):
        """
        Initializes the Config object with data from the provided dictionary.

        :param data: Dictionary containing configuration information.
        :type data: dict

        This method sets up the active model, processes dataset paths (e.g., absolute paths
        for raw EEG data and smoothed EEG features), and assigns specific configurations
        for EEG-ITNet and Conformer models.

        """
        self.active = data['active']

        self.dataset = data['dataset']
        self.dataset['eeg_raw_data_abs_path'] = os.path.join(
            data['dataset']['root_path'],
            data['dataset']['eeg_raw_data_path']
        )
        self.dataset['eeg_feature_smooth_abs_path'] = os.path.join(
            data['dataset']['root_path'],
            data['dataset']['eeg_feature_smooth_path']
        )

        self.eeg_it_net = data['EEG-ITNet']
        self.conformer = data['Conformer']


def parse_opt(argv: list[str]) -> str:
    """
    Parses command-line arguments to retrieve the configuration file path.

    :param argv: List of command-line arguments.
    :type argv: list[str]
    :return: Path to the configuration file specified by the '-c' or '--config' option.
    :rtype: str

    This function uses getopt to parse the command-line arguments. It accepts the following options:
      - '-h': Displays a help message and exits.
      - '-c' or '--config': Specifies the configuration file path.

    If no valid option is provided, or if an error occurs, the function prints usage instructions
    and exits the program with an error code.

    """
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

    active = data['active']
    if active == 'Conformer':
        from train.conformer import start
        start(config)
    elif active == 'Conformer-f':
        from train.conformer_f import start
        start(config)
    elif active == 'EEG-ITNet':
        from train.eeg_itnet import start
        start(config)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    main(sys.argv[1:])
