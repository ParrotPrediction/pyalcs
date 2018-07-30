from lcs import TypedList
from . import Classifier


class ClassifierList(TypedList):

    def __init__(self, *args):
        super().__init__((Classifier,), *args)
