from typing import List, Optional, Tuple

from lcs import Perception
from lcs.agents.acs2 import Classifier
from lcs.agents.acs2.ClassifiersList import ClassifiersList


class GoalSequenceSearcher:

    def __init__(self):
        self.forward_classifiers = []
        self.backward_classifiers = []
        self.forward_perceptions = []
        self.backward_perceptions = []

    def search_goal_sequence(self,
                             reliable_classifiers: ClassifiersList,
                             start: Perception,
                             goal: Perception) -> list:
        """
        Searches a path from start to goal using a bidirectional method in the
        environmental model (i.e. the list of reliable classifiers).

        Parameters
        ----------
        reliable_classifiers
            list of reliable classifiers
        start: Perception
        goal: Perception

        Returns
        -------
        list
            sequence of actions
        """
        if len(reliable_classifiers) < 1:
            return []

        max_depth = 6
        forward_size = 1
        backward_size = 1
        forward_point = 0
        backward_point = 0
        action_sequence = None

        self.forward_classifiers.clear()
        self.backward_classifiers.clear()
        self.forward_perceptions.clear()
        self.backward_perceptions.clear()

        self.forward_perceptions.append(Perception(start))
        self.backward_perceptions.append(Perception(goal))

        for depth in range(0, max_depth):
            # forward step
            action_sequence, forward_size_new = \
                self._search_one_forward_step(reliable_classifiers,
                                              forward_size, forward_point)
            forward_point = forward_size
            forward_size = forward_size_new

            if action_sequence is not None:
                return action_sequence

            # backwards step
            action_sequence, backward_size_new = \
                self._search_one_backward_step(reliable_classifiers,
                                               backward_size, backward_point)
            backward_point = backward_size
            backward_size = backward_size_new

            if action_sequence is not None:
                return action_sequence

        # depth limit was reached -> return empty action sequence
        return []

    def _search_one_forward_step(self,
                                 reliable_classifiers: ClassifiersList,
                                 forward_size: int,
                                 forward_point: int) -> Tuple[Optional[list],
                                                              int]:
        """
        Searches one step forward in the reliable_classifiers classifier list.
        Returns None if nothing was found so far, a sequence with a -1 element
        if the search failed completely
        (which is the case if the allowed array size of 10000 is reached),
        or the sequence if one was found.
        :param reliable_classifiers: ClassifiersList
        :param forward_size: int
        :param forward_point: int
        :return: act sequence and new forward_size
        """
        size = forward_size
        for i in range(forward_point, forward_size):
            match_forward = reliable_classifiers. \
                form_match_set(situation=self.forward_perceptions[i])
            for match_set_element in match_forward:
                anticipation = match_set_element. \
                    get_best_anticipation(self.forward_perceptions[i])
                if self.get_state_idx(self.forward_perceptions,
                                      anticipation) is None:
                    # state not detected forward -> search in backwards
                    backward_sequence_idx = self. \
                        get_state_idx(self.backward_perceptions,
                                      anticipation)
                    if backward_sequence_idx is None:
                        # state neither detected backwards
                        self.forward_perceptions.append(anticipation)
                        self.forward_classifiers.append(
                            self._form_new_classifiers(
                                self.forward_classifiers, i,
                                match_set_element))
                        size += 1
                        if size > 10001:
                            # logging.debug("Arrays are full")
                            return [], size
                    else:
                        # sequence found
                        return self._form_sequence_forwards(
                            i, backward_sequence_idx, match_set_element), size
        return None, size

    def _search_one_backward_step(self,
                                  reliable_classifiers: ClassifiersList,
                                  backward_size: int,
                                  backward_point: int) -> Tuple[Optional[list],
                                                                int]:
        """
        Searches one step backward in the reliable_classifiers classifiers list
        Returns None if nothing was found so far, a sequence with a -1 element
        if the search failed completely
        (which is the case if the allowed array size of 10000 is reached),
        or the sequence if one was found.
        :param reliable_classifiers: ClassifiersList
        :param backward_size: int
        :param backward_point: int
        :return: act sequence and new backward_size
        """
        size = backward_size
        for i in range(backward_point, backward_size):
            match_backward = reliable_classifiers.form_match_set_backwards(
                situation=self.backward_perceptions[i])
            for match_set_el in match_backward:
                anticipation = match_set_el. \
                    get_backwards_anticipation(self.backward_perceptions[i])
                if anticipation is not None and self. \
                        get_state_idx(self.backward_perceptions,
                                      anticipation) is None:
                    # Backwards anticipation was formable but
                    # not detected backwards
                    forward_sequence_idx = self.\
                        get_state_idx(self.forward_perceptions,
                                      anticipation)
                    if forward_sequence_idx is None:
                        self.backward_perceptions.append(anticipation)
                        self.backward_classifiers.append(
                            self._form_new_classifiers(
                                self.backward_classifiers, i, match_set_el))
                        size += 1
                        if size > 10001:
                            # logging.debug("Arrays are full")
                            return [], size
                    else:
                        return self._form_sequence_backwards(
                            i, forward_sequence_idx, match_set_el), size
        return None, size

    @staticmethod
    def _form_new_classifiers(classifiers_lists: List[ClassifiersList],
                              i: int,
                              match_set_el: Classifier) -> ClassifiersList:
        """
        Executes actions after sequence was not detected.
        :param classifiers_lists: list of ClassifiersLists
        :param i: int
        :param match_set_el: Classifier
        :return: new size of classifiers
        """
        if i > 0:
            new_classifiers = ClassifiersList()
            new_classifiers.extend(classifiers_lists[i - 1])
        else:
            new_classifiers = ClassifiersList()
        new_classifiers.append(match_set_el)
        return new_classifiers

    def _form_sequence_forwards(self, i: int,
                                backward_sequence_idx: int,
                                match_set_el: Classifier) -> list:
        """
        Forms sequence when it was found forwards.
        :param i:
        :param backward_sequence_idx:
        :param match_set_el: Classifier
        :return: act sequence
        """
        # count sequence size
        sequence_size = 0
        if i > 0:
            sequence_size += len(self.forward_classifiers[i - 1])
        if backward_sequence_idx > 0:
            sequence_size += len(self.backward_classifiers[
                                 backward_sequence_idx - 1])
        sequence_size += 1

        # construct sequence
        act_seq: list = [-1] * sequence_size
        j = 0
        if i > 0:
            for j, cl in enumerate(self.forward_classifiers[i - 1]):
                act_seq[len(self.forward_classifiers[i - 1]) - j - 1] \
                    = cl.action
            j += 1

        act_seq[j] = match_set_el.action
        j += 1
        if backward_sequence_idx > 0:
            for k, cl in enumerate(self.backward_classifiers[
                                   backward_sequence_idx - 1]):
                act_seq[k + j] = cl.action
        return act_seq

    def _form_sequence_backwards(self, i: int,
                                 forward_sequence_idx: int,
                                 match_set_el: Classifier) -> list:
        """
        Forms sequence when it was found backwards.
        :param i: int
        :param forward_sequence_idx: int
        :param match_set_el: Classifier
        :return: act sequence
        """
        # count sequence size
        sequence_size = 0
        if i > 0:
            sequence_size += len(self.backward_classifiers[i - 1])
        if forward_sequence_idx > 0:
            sequence_size += len(
                self.forward_classifiers[forward_sequence_idx - 1])
        sequence_size += 1

        # construct sequence
        act_seq: list = [-1] * sequence_size
        j = 0
        if forward_sequence_idx > 0:
            for j, cl in enumerate(self.forward_classifiers[
                                   forward_sequence_idx - 1]):
                act_seq[len(self.forward_classifiers[
                            forward_sequence_idx - 1]) - j - 1] = cl.action
            j += 1
        act_seq[j] = match_set_el.action
        j += 1
        if i > 0:
            for k, cl in enumerate(self.backward_classifiers[i - 1]):
                act_seq[k + j] = cl.action
        return act_seq

    @staticmethod
    def get_state_idx(perceptions: List[Perception],
                      state: Perception) -> Optional[int]:
        """
        Returns the position of state in list of perception.

        Parameters
        ----------
        perceptions: List[Perception]
            list of perceptions
        state: Perception
            sought perception

        Returns
        -------
        Optional[int]
            Position of perception, None if perception was not found
        """
        try:
            return perceptions.index(state)
        except ValueError:
            return None
