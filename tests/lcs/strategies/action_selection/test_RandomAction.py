from lcs.strategies.action_selection import RandomAction
import lcs.agents.acs2 as acs2


class TestRandomAction:

    def test_random_action_selection(self):
        # given
        all_actions = 4
        population = acs2.ClassifiersList(*[])

        strategy = RandomAction(all_actions)

        # when & then
        for _ in range(100):
            assert 0 <= strategy(population) < all_actions
