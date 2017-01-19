from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):

    @abstractmethod
    def insert_animat(self):
        """
        Sets animat coordinates in fixed/random position inside a maze
        """
        raise NotImplementedError()

    @abstractmethod
    def reset_animat_state(self):
        raise NotImplementedError

    @abstractmethod
    def get_animat_perception(self, **position):
        """
        Sets the animat perception of desired directions from given position.

        :param position: position of the animat in the environment
        :return: animat perception (namedtuple)
        """
        raise NotImplementedError()

    @abstractmethod
    def execute_action(self, action: int, perception) -> int:
        """
        Executes an action in the environment

        :param action: action to be executed
        :param perception: perception of the environment
        :return: reward
        """
        raise NotImplementedError()
