import pytest
import random

from lcs import Perception
from lcs.agents.xcs import Configuration, Condition, Classifier, ClassifiersList


class TestXCS:

    @pytest.fixture
    def cfg(self):
        return Configuration(4, 4)


