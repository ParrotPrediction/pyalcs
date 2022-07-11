from __future__ import annotations

from lcs import Perception
import lcs.agents.acs as acs


class Condition(acs.Condition):

    def specialize_with_condition(self, other: Condition) -> None:
        for idx, new_el in enumerate(other):
            if new_el != self.WILDCARD:
                if new_el != '0.0':
                    self[idx] = '1.0'
                else:
                    self[idx] = '0.0'

    def does_match(self, p: Perception) -> bool:
        """
        Check if condition match given observations

        Parameters
        ----------
        p: Union[Perception, Condition]
            perception or condition object

        Returns
        -------
        bool
            True if condition match given list, False otherwise
        """
        j = 0
        for ci, oi in zip(self, p):
            i = j
            check = False
            for obs in oi:
                if str(ci) != '0.0' and str(obs) != '0.0':
                    check = True
                    i += 1
                    break
                i += 1
            if '1.0' not in ci[j:i]:
                j += len(oi)
                continue
            j += len(oi)
            if not check:
                return False

        return True
