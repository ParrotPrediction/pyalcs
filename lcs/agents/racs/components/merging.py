from typing import List

from lcs.agents.racs import Classifier, ClassifierList, Condition, Effect
from lcs.representations import UBR


def merge(cl: Classifier, pop: ClassifierList) -> List[Classifier]:
    extendable = [e_cl for e_cl in pop if _can_be_extended(cl, e_cl)]
    offspring = [_extend(cl, e_cl) for e_cl in extendable]

    # offspring should be unique (set)
    return offspring


def _can_be_extended(new_cl: Classifier, existing_cl: Classifier) -> bool:
    # Check if ranges are overlapping.
    # Should be used alongside with subsumption (or)

    # Validate the assumption that the ranges are the same for condition
    # and effect parts.
    # assert new_cl.condition == new_cl.effect
    # assert existing_cl.condition == existing_cl.effect

    # since condition == effect we can check only the condition string
    return all(c1_ubr.can_be_merged(c2_ubr)
               for (c1_ubr, c2_ubr)
               in zip(new_cl.condition, existing_cl.condition))


def _extend(new_cl: Classifier, existing_cl: Classifier) -> Classifier:
    assert new_cl.action == existing_cl.action

    ps = []  # new perception string

    for idx, (e_ubr, n_ubr) in enumerate(zip(existing_cl.condition,
                                             new_cl.condition)):
        lb = min(e_ubr.lower_bound, n_ubr.lower_bound)
        ub = max(e_ubr.upper_bound, n_ubr.upper_bound)
        ps.append(UBR(lb, ub))

    return Classifier(
        condition=Condition(ps, existing_cl.cfg),
        action=existing_cl.action,
        effect=Effect(ps, existing_cl.cfg),
        quality=(existing_cl.q + new_cl.q) / 2,
        reward=(existing_cl.r + new_cl.r) / 2,
        immediate_reward=(existing_cl.ir + new_cl.ir) / 2,
        talp=(existing_cl.talp + new_cl.talp) / 2,
        cfg=existing_cl.cfg)
