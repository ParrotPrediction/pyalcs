from abc import ABCMeta, abstractmethod


class Environment(metaclass=ABCMeta):

    @abstractmethod
    def insert_animat(self) -> None:
        """
        Sets animat coordinates in fixed/random position inside a maze
        """
        raise NotImplementedError()

    @abstractmethod
    def move_was_successful(self) -> bool:
        """
        Returns information whether the animat has proceeded to another
        situation (perception).
        """
        raise NotImplementedError

    @abstractmethod
    def trial_finished(self) -> bool:
        """
        Returns information whether an animat has accomplished his task.
        Based on this the environment might get reloaded.

        :return: True is animat accomplished task, false otherwise
        """
        raise NotImplementedError()

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
