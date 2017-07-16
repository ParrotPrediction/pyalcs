from abc import ABCMeta, abstractmethod
from collections import defaultdict

from alcs.environment import Environment
from alcs.helpers.metrics import Metric


class Agent(metaclass=ABCMeta):

    def __init__(self):
        self.metrics = defaultdict(list)
        self.metrics_handlers = []

    def add_metrics_handlers(self, metrics_handlers: list):
        """
        Add metrics that will be captured at the algorithm execution

        :param metrics_handlers: list of metrics
        """
        self.metrics_handlers = metrics_handlers

    def acquire_metrics(self, *args, **kwargs) -> None:
        """
        Function evaluates all handlers for calculating metrics
        """
        for handler in self.metrics_handlers:
            if not isinstance(handler, Metric):
                raise ValueError('Incorrect metric handler passed')

            name, metric = handler.get(*args, **kwargs)

            self.metrics[name].append(metric)

    @abstractmethod
    def evaluate(self,
                 environment: Environment,
                 experiments: int,
                 generations: int) -> None:
        """
        Evaluates selected algorithm.

        :param environment: environment to operate on
        :param experiments: number of experiments
        :param generations: number of generations
        """
        raise NotImplementedError()
