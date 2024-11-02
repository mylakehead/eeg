"""
Module Name: Configurations for EEG and Conformer Models

Description:
    This module provides a configuration class to handle settings and paths for models such as
    EEG-ITNet and Conformer. It parses the configuration file and processes the necessary paths
    for datasets and model parameters.

    The module includes the following:
        - parse_opt: Parses command-line arguments to retrieve the configuration file path.
        - parse_config: Loads and parses a JSON configuration file.
        - Config: A class to organize and store configuration data for the active model, dataset paths,
          and model-specific configurations.

Usage:
    To use this module, provide the path to a configuration file when executing the main function,
    specifying the model to be loaded and trained.

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
    Last Modified: 2024-11-02

"""

import os
import sys
import json
import getopt


class Config:
    """
    Configuration class that processes and organizes configuration data for the model.

    This class initializes configuration attributes based on the provided data dictionary,
    including paths for EEG data, model selection, and specific configuration settings
    for EEG-ITNet and Conformer models.

    Attributes:
        active (str): The active model name.
        dataset (dict): Dictionary containing dataset paths and configuration.
        eeg_it_net (dict): Configuration specific to the EEG-ITNet model.
        conformer (dict): Configuration specific to the Conformer model.
    """
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
    """
    Main function to parse command-line arguments, load configuration, and start the training process
    for a specified model.

    :param argv: List of command-line arguments. Expects the configuration file path as an argument.
    :type argv: list

    Note: TensorFlow import is placed within the function scope to reduce initial load time.
    """
    config_file = parse_opt(argv)
    data = parse_config(config_file)
    config = Config(data)

    active = data['active']
    if active == 'Conformer':
        from train.conformer import start
        start(config)
    elif active == 'EEG-ITNet':
        from train.eeg_itnet import start
        start(config)


if __name__ == '__main__':
    main(sys.argv[1:])
