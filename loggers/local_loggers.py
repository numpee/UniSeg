import os
import pickle

from loggers import AbstractBaseLogger


class SimpleLoggerExample(AbstractBaseLogger):
    """
    Simple Example of a custom logger.
    The log() function is called every epoch with loss and IOU information.
    The complete() function is called at the end of training. In this case, it saves the dictionary with all the logs
    into a pickle file. Obviously, this is a very bare example, so it you should modify it to your needs.
    """

    def __init__(self, prefix, export_folder):
        self.prefix = prefix
        self.log_data = {}
        self.export_folder = export_folder

    def log(self, log_data, step, commit=False):
        log_metrics = {self.prefix + k: v for k, v in log_data.items() if not isinstance(v, dict)}
        self.log_data[step] = log_metrics

    def complete(self, log_data, step, commit=True):
        with open(os.path.join(self.export_folder, "your_log_data.pkl"), "wb") as f:
            pickle.dump(self.log_data, f)
