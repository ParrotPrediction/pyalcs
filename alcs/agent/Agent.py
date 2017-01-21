from abc import ABCMeta, abstractmethod
from alcs.environment import Environment


class Agent(metaclass=ABCMeta):

    def __init__(self, environment: Environment):
        """
        Initialize the environment
        :param environment: environment in which an animat is placed.
        """
        self.env = environment

    @abstractmethod
    def evaluate(self, generations, **kwargs) -> None:
        """
        Evaluates selected algorithm.

        :param generations: numer of generations
        :param kwargs: additional arguments
        """
        raise NotImplementedError()
