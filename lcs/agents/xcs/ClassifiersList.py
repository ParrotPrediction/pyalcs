from dataclasses import dataclass
from typing import Union, Optional, Generator, List, Dict
from typing import Callable, List, Tuple

from lcs import TypedList, Perception
from lcs.agents.xcs import Classifier


class ClassifiersList(TypedList):
    def __init__(self, *args, oktypes=(Classifier,)) -> None:
        super().__init__(*args, oktypes=oktypes)

    def form_match_set(self, situation: Perception):
        matching_ls = [cl for cl in self if cl.does_match(situation)]
        # TODO: COVERING
        return ClassifiersList(*matching_ls)
