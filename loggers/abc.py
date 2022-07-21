from typing import Sequence

import abc
import wandb

from abc import ABC


class AbstractBaseLogger(ABC):
    @abc.abstractmethod
    def log(self, log_data: dict, step: int, commit=False) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, log_data: dict, step: int) -> None:
        raise NotImplementedError


class LoggingService(object):
    def __init__(self, loggers: Sequence[AbstractBaseLogger]):
        self.loggers = loggers

    def log(self, log_data: dict, step: int, commit=False):
        for logger in self.loggers:
            logger.log(log_data, step, commit=commit)

    def complete(self, log_data: dict, step: int):
        for logger in self.loggers:
            logger.complete(log_data, step)

    @staticmethod
    def commit(step: int):
        wandb.log({}, step=step, commit=True)
