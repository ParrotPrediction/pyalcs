import itertools
from dataclasses import dataclass

from lcs.strategies.genetic_algorithms import roulette_wheel_selection


@dataclass(unsafe_hash=True)
class IdClassifier:
    id: int
    q: float


class TestGeneticAlgorithms:

    def test_should_return_parents(self):
        # given
        cl1 = IdClassifier(1, 0.7)
        cl2 = IdClassifier(2, 0.3)
        cl3 = IdClassifier(3, 0.1)
        pop = [cl1, cl2, cl3]

        def fitnessfcn(cl):
            return pow(cl.q, 3)

        # when
        n = 1000
        results = []
        for _ in range(n):
            results.extend(roulette_wheel_selection(pop, fitnessfcn))

        # then
        results = sorted(results, key=lambda el: el.id)
        stats = {k: len(list(g)) for k, g in itertools.groupby(
            results, key=lambda el: el.id)}

        assert stats[cl1.id] > stats[cl2.id] * 10 > stats[cl3.id] * 10
