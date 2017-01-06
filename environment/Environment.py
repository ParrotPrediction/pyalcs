from abc import ABCMeta, abstractmethod

class Environment(metaclass=ABCMeta):

    @abstractmethod
    def insert_animat(self):
        """
        Puts animat into suitable random position inside a maze

        :return: animat starting position
        """
        raise NotImplementedError()

    @abstractmethod
    def get_animat_perception(self, **position):
        """
        Return the perception of the animat in the given position.

        :param position: position of the animat in the environment
        :return: an array of possible values is returned.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_animat_position_value(self):
        """
        Gets the value of the current position

        :return: the value of the current position
        """
        raise NotImplementedError()

    @abstractmethod
    def execute_action(self, action):
        """
        Executes an action in the environment
        :param action: action to be executed
        :return: ???
        """
        raise NotImplementedError()

    @abstractmethod
    def get_reward(self):
        """Return the reward from the environment"""
        raise NotImplementedError()

