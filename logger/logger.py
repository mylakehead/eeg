import logging
import os
import json
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="results/logs", log_filename="training.log"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_filename)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.params = None  # Store parameters
        self.results = None  # Store results

    def log_parameters(self, params: dict):
        self.params = params
        self.logger.info("Training Parameters:")
        self.logger.info(json.dumps(params, indent=4))

    def log_result(self, result: dict):
        self.results = result
        self.logger.info("Training Results:")
        self.logger.info(json.dumps(result, indent=4))

    def save_to_file(self):
        if self.params is None or self.results is None:
            self.logger.error("Parameters or results are not set. Unable to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(os.path.dirname(self.log_path), f"training_{timestamp}.json")

        with open(output_file, 'w') as f:
            json.dump({"parameters": self.params, "results": self.results}, f, indent=4)

        self.logger.info(f"Parameters and results saved to {output_file}")
