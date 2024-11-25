"""
This script serves as the main entry point for a project that performs either exploratory data analysis (EDA)
or training using a Conformer model for EEG data. It handles configuration parsing, mode selection,
and the execution of appropriate modules based on user input.

Execution Instructions:
1. Provide a JSON configuration file containing project settings.
2. Run the script with the `-c` or `--config` flag to specify the configuration file:

Copyright:
    MIT License

    Copyright © 2024 Lakehead University, Large Scale Data Analytics Group Project

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Authors:
    Kang Hong, XingJian Han, Minh Anh Nguyen
    hongkang@hongkang.name, xhan15@lakeheadu.ca, mnguyen9@lakeheadu.ca

Date:
    Created: 2024-10-02
    Last Modified: 2024-11-24
"""

import os
import sys
import json
import getopt

import torch

from train.conformer import start
from eda.f_test import analyze


class Config:
    """
    The Config class processes and organizes configuration data for the project.
    It sets up operational modes, dataset paths, and model-specific parameters.
    """
    def __init__(self, data: dict):
        self.mode = data['mode']
        self.dataset = data['dataset']
        self.dataset['eeg_raw_data_abs_path'] = os.path.join(
            data['dataset']['root_path'],
            data['dataset']['eeg_raw_data_path']
        )
        self.dataset['eeg_feature_smooth_abs_path'] = os.path.join(
            data['dataset']['root_path'],
            data['dataset']['eeg_feature_smooth_path']
        )
        self.conformer = data['Conformer']


def parse_opt(argv: list[str]) -> str:
    """
    Parses command-line arguments to extract the configuration file path.

    Behavior:
        - If the '-h' option is provided, displays the usage information and exits.
        - If the '-c' or '--config' option is used, returns the specified configuration file path.
        - Exits the script with an appropriate error message if no valid option is provided or there is an error.
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

    if config.mode == 'eda':
        analyze(config)
    else:
        start(config)


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    main(sys.argv[1:])
