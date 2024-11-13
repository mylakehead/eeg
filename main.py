import os
import sys
import json
import getopt


class Config:
    """Configuration class for the EEG-ITNet, Conformer, and ViT models."""
    def __init__(self, data: dict):
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
    """Parse command line arguments.
    
    Args:
        argv (list[str]): List of command line arguments
        
    Returns:
        str: Path to config file specified by -c/--config argument
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
    """Parse the JSON configuration file.
    
    Args:
        config_file (str): Path to the JSON configuration file
        
    Returns:
        dict: Parsed configuration data
    """
    with open(config_file, 'r') as f:
        data = json.load(f)
        return data


def main(argv):
    config_file = parse_opt(argv)
    data = parse_config(config_file)
    config = Config(data)

    active = data['active']
    if active == 'Conformer':
        from train.conformer_raw import start
        start(config)
    elif active == 'ViT':
        from train.conformer_feature import start
        start(config)
    elif active == 'EEG-ITNet':
        from train.eeg_itnet import start
        start(config)


if __name__ == '__main__':
    main(sys.argv[1:])
