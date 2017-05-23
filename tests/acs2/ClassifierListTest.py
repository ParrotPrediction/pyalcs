import unittest

from alcs.agent.Perception import Perception
from alcs.agent.acs2 import ClassifiersList, Classifier, Condition, Effect

from random import randint


class ClassifierListTest(unittest.TestCase):

    def setUp(self):
        self.population = ClassifiersList()

    def test_should_insert_classifier_1(self):
        # Try to insert an integer instead of classifier object
        self.assertRaises(TypeError, self.population.append, 4)

    def test_should_insert_classifier_2(self):
        self.population.append(Classifier())
        self.assertEqual(1, len(self.population))

    def test_should_form_match_set(self):
        situation = Perception(['1', '1', '1', '1', '0', '0', '0', '0'])

        # C1 - general condition
        c1 = Classifier()

        # C2 - matching condition
        c2 = Classifier(condition=['1', '#', '#', '#', '0', '#', '#', '#'])

        # C3 - non-matching condition
        c3 = Classifier(condition=['0', '#', '#', '#', '1', '#', '#', '#'])

        self.population.append(c1)
        self.population.append(c2)
        self.population.append(c3)

        match_set = ClassifiersList.form_match_set(self.population, situation)

        self.assertEqual(2, len(match_set))
        self.assertIn(c1, match_set)
        self.assertIn(c2, match_set)

    def test_should_form_action_set(self):
        c0 = Classifier(action=0)
        c01 = Classifier(action=0)
        c1 = Classifier(action=1)

        self.population.append(c0)
        self.population.append(c01)
        self.population.append(c1)

        action_set = ClassifiersList.form_action_set(self.population, 0)
        self.assertEqual(2, len(action_set))
        self.assertIn(c0, action_set)
        self.assertIn(c01, action_set)

        action_set = ClassifiersList.form_action_set(self.population, 1)
        self.assertEqual(1, len(action_set))
        self.assertIn(c1, action_set)

    def test_should_calculate_maximum_fitness(self):
        # C1 - does not anticipate change
        c1 = Classifier()

        self.population.append(c1)
        self.assertEqual(0.0, self.population.get_maximum_fitness())

        # C2 - does anticipate some change
        c2 = Classifier(effect=['1', '#', '#', '#', '0', '#', '#', '#'],
                        reward=0.25)

        self.population.append(c2)
        self.assertEqual(0.125, self.population.get_maximum_fitness())

        # C3 - does anticipate change and is quite good
        c3 = Classifier(effect=['1', '#', '#', '#', '#', '#', '#', '#'],
                        quality=0.8,
                        reward=5)

        self.population.append(c3)
        self.assertEqual(4, self.population.get_maximum_fitness())

    def test_should_return_best_fitness_action(self):
        # C1 - does not anticipate change
        c1 = Classifier(action=1)
        self.population.append(c1)

        # Some random action should be selected here
        best_action = self.population.choose_best_fitness_action()
        self.assertIsNotNone(best_action)

        # C2 - does anticipate some change
        c2 = Classifier(action=2,
                        effect=['1', '#', '#', '#', '0', '#', '#', '#'],
                        reward=0.25)
        self.population.append(c2)

        # Here C2 action should be selected
        best_action = self.population.choose_best_fitness_action()
        self.assertEqual(2, best_action)

        # C3 - does anticipate change and is quite good
        c3 = Classifier(action=3,
                        effect=['1', '#', '#', '#', '#', '#', '#', '#'],
                        quality=0.8,
                        reward=5)
        self.population.append(c3)

        # Here C3 has the biggest fitness score
        best_action = self.population.choose_best_fitness_action()
        self.assertEqual(3, best_action)

    def test_should_return_random_action(self):
        random_actions = []

        for _ in range(0, 500):
            random_actions.append(self.population.choose_random_action())

        min_action = min(random_actions)
        max_action = max(random_actions)

        self.assertEqual(0, min_action)
        self.assertEqual(7, max_action)

    def test_should_return_latest_action(self):
        c0 = Classifier(action=0)
        c0.talp = 1

        self.population.append(c0)

        # Should return first action with no classifiers
        self.assertEqual(1, self.population.choose_latest_action())

        # Add rest of classifiers
        self.population.append(Classifier(action=3))
        self.population.append(Classifier(action=7))
        self.population.append(Classifier(action=5))
        self.population.append(Classifier(action=1))
        self.population.append(Classifier(action=4))
        self.population.append(Classifier(action=2))
        self.population.append(Classifier(action=6))

        # Assign each classifier random talp from certain range
        for cl in self.population:
            cl.talp = randint(70, 100)

        # But third classifier (action 7) will be the executed long time ago
        self.population[2].talp = randint(10, 20)
        self.assertEqual(7, self.population.choose_latest_action())

    def test_should_return_worst_quality_action(self):
        c0 = Classifier(action=0)
        self.population.append(c0)

        # Should return C1 (because it's first not mentioned)
        self.assertEqual(1,
                         self.population.choose_action_from_knowledge_array())

        # Add rest of classifiers
        c1 = Classifier(action=1, numerosity=31, quality=0.72)
        self.population.append(c1)

        c2 = Classifier(action=2, numerosity=2, quality=0.6)
        self.population.append(c2)

        c3 = Classifier(action=3, numerosity=2, quality=0.63)
        self.population.append(c3)

        c4 = Classifier(action=4, numerosity=7, quality=0.75)
        self.population.append(c4)

        c5 = Classifier(action=5, numerosity=1, quality=0.63)
        self.population.append(c5)

        c6 = Classifier(action=6, numerosity=6, quality=0.52)
        self.population.append(c6)

        c7 = Classifier(action=7, numerosity=10, quality=0.36)
        self.population.append(c7)

        # Classifier C7 should be the worst here
        self.assertEqual(7,
                         self.population.choose_action_from_knowledge_array())

    def test_should_get_similar_classifier(self):
        self.population.append(Classifier(action=1))
        self.population.append(Classifier(action=2))
        self.population.append(Classifier(action=3))

        # No similar classifiers exist
        self.assertIsNone(self.population.get_similar(Classifier(action=4)))

        # Should find similar classifier
        self.assertIsNotNone(self.population.get_similar(Classifier(action=2)))

    def test_should_apply_reinforcement_learning(self):
        c1 = Classifier()
        c1.r = 34.29
        c1.ir = 11.29
        self.population.append(c1)

        self.population.apply_reinforcement_learning(0, 28.79)

        self.assertAlmostEqual(33.94, self.population[0].r, 2)
        self.assertAlmostEqual(10.73, self.population[0].ir, 2)

    def test_should_insert_alp_offspring_1(self):
        # Given
        new_list = ClassifiersList()

        # *##*#O*O N O####*O* (########) q: 0.5	r: 8.96245	i: 0	exp: 1 tga: 423	talp: 423 tav: 27.3182 num: 1
        child = Classifier(
            condition=Condition(['1', '#', '#', '1', '#', '0', '1', '0']),
            action=0,
            effect=Effect(['0', '#', '#', '#', '#', '1', '0', '1']),
            quality=0.5,
            reward=8.96245,
            intermediate_reward=0,
            experience=1,
            tga=423,
            talp=423,
            tav=27.3182
        )

        # *##*#O*O N O####*O* (########) q: 0.571313	r: 7.67011	i: 0	exp: 3 tga: 225	talp: 423 tav: 70.881 num: 1
        c1 = Classifier(
            condition=Condition(['1', '#', '#', '1', '#', '0', '1', '0']),
            action=0,
            effect=Effect(['0', '#', '#', '#', '#', '1', '0', '1']),
            quality=0.571313,
            reward=7.67011,
            intermediate_reward=0,
            experience=3,
            tga=225,
            talp=423,
            tav=70.881
        )

        # *####O*O N O####*O* (#{*,O}{O,*}O{O,*}###) q: 0.462151	r: 8.96245	i: 0	exp: 11 tga: 143	talp: 423 tav: 27.3182 num: 1
        c2 = Classifier(
            condition=Condition(['1', '#', '#', '#', '#', '0', '1', '0']),
            action=0,
            effect=Effect(['0', '#', '#', '#', '#', '1', '0', '1']),
            quality=0.462151,
            reward=8.96245,
            intermediate_reward=0,
            experience=11,
            tga=143,
            talp=423,
            tav=27.3182
        )

        # *####O## N O####*## (#{O,*}{*,O}{O,*}{O,*}#{*,O}{O,*}) q: 0.31452	r: 9.04305	i: 0	exp: 19 tga: 49	talp: 423 tav: 19.125 num: 1
        c3 = Classifier(
            condition=Condition(['1', '#', '#', '#', '#', '0', '#', '#']),
            action=0,
            effect=Effect(['0', '#', '#', '#', '#', '1', '#', '#']),
            quality=0.31452,
            reward=9.04305,
            intermediate_reward=0,
            experience=19,
            tga=49,
            talp=423,
            tav=19.125
        )

        # Add classifiers into current ClassifierList
        self.population.extend([c1, c2, c3])

        # When
        self.population.add_alp_classifier(child, new_list)

        # Then
        self.assertEqual(3, len(self.population))
        self.assertIn(c1, self.population)
        self.assertIn(c2, self.population)
        self.assertIn(c3, self.population)
        self.assertAlmostEqual(0.592747, c1.q, places=1)

    def test_should_insert_alp_offspring_2(self):
        # Given
        new_list = ClassifiersList()

        # #*O##O## S ######## (########) q: 0.5	r: 18.206	i: 0	exp: 1 tga: 747	talp: 747 tav: 22.0755 num: 1
        child = Classifier(
            condition=Condition('#1O##O##'),
            action=0,
            quality=0.5,
            reward=18.206,
            intermediate_reward=0,
            experience=1,
            tga=747,
            talp=747,
            tav=22.0755
        )

        # #*O#O### S ######## (########) q: 0.650831	r: 14.8323	i: 0	exp: 5 tga: 531	talp: 747 tav: 48.3562 num: 1
        c1 = Classifier(
            condition=Condition('#1O#O###'),
            action=0,
            quality=0.650831,
            reward=14.8323,
            intermediate_reward=0,
            experience=5,
            tga=531,
            talp=747,
            tav=48.3562
        )

        # ##O#O### S ######## (########) q: 0.79094	r: 9.97782	i: 0	exp: 10 tga: 330	talp: 747 tav: 43.7171 num: 1
        c2 = Classifier(
            condition=Condition('##O#O###'),
            action=0,
            quality=0.79094,
            reward=9.97782,
            intermediate_reward=0,
            experience=10,
            tga=330,
            talp=747,
            tav=43.7171
        )

        # #*O###*O S #O*####* (*##OOO##) q: 0.515369	r: 8.3284	i: 0	exp: 8 tga: 316	talp: 747 tav: 57.8883 num: 1
        c3 = Classifier(
            condition=Condition('#1O###1O'),
            action=0,
            effect=Effect('#O1####1'),
            quality=0.515369,
            reward=8.3284,
            intermediate_reward=0,
            experience=8,
            tga=316,
            talp=747,
            tav=57.8883
        )
        # TODO p2: Maybe checking wheter marked sometimes does not work..
        # I mean type incompability
        c3.mark[0].update(['1'])
        c3.mark[3].update(['0'])
        c3.mark[4].update(['0'])
        c3.mark[5].update(['0'])

        # ####O### S ######## (########) q: 0.903144	r: 14.8722	i: 0	exp: 25 tga: 187	talp: 747 tav: 23.0038 num: 1
        c4 = Classifier(
            condition=Condition('####O###'),
            action=0,
            quality=0.903144,
            reward=14.8722,
            intermediate_reward=0,
            experience=25,
            tga=187,
            talp=747,
            tav=23.0038
        )

        # #*O####O S #O*####* (*##{*,O}{*,O}{*,O}{O,*}#) q: 0.647915	r: 9.24712	i: 0	exp: 14 tga: 154	talp: 747 tav: 44.5457 num: 1
        c5 = Classifier(
            condition=Condition('#1O####O'),
            action=0,
            effect=Effect('#O1####1'),
            quality=0.647915,
            reward=9.24712,
            intermediate_reward=0,
            experience=14,
            tga=154,
            talp=747,
            tav=44.5457
        )
        c5.mark[0].update(['1'])
        c5.mark[3].update(['0', '1'])
        c5.mark[4].update(['0', '1'])
        c5.mark[5].update(['0', '1'])
        c5.mark[6].update(['0', '1'])

        # #*O##### S ######## (*##***{*,O}{O,*}) q: 0.179243	r: 18.206	i: 0	exp: 29 tga: 104	talp: 747 tav: 22.0755 num: 1
        c6 = Classifier(
            condition=Condition('#1O#####'),
            action=0,
            quality=0.179243,
            reward=18.206,
            intermediate_reward=0,
            experience=29,
            tga=104,
            talp=747,
            tav=22.0755
        )
        c6.mark[0].update(['1'])
        c6.mark[3].update(['1'])
        c6.mark[4].update(['1'])
        c6.mark[5].update(['1'])
        c6.mark[6].update(['0', '1'])
        c6.mark[7].update(['0', '1'])

        # ##O##### S ######## ({*,O}{O,*}#{O,*}*{O,*}{*,O}{*,O}) q: 0.100984	r: 15.91	i: 0	exp: 44 tga: 58	talp: 747 tav: 14.4171 num: 1
        c7 = Classifier(
            condition=Condition('##O#####'),
            action=0,
            quality=0.100984,
            reward=15.91,
            intermediate_reward=0,
            experience=44,
            tga=58,
            talp=747,
            tav=14.4171
        )
        c7.mark[0].update(['0', '1'])
        c7.mark[1].update(['0', '1'])
        c7.mark[3].update(['0', '1'])
        c7.mark[5].update(['0', '1'])
        c7.mark[6].update(['0', '1'])
        c7.mark[7].update(['0', '1'])

        # Add classifiers into current ClassifierList
        self.population.extend([c1, c2, c3, c4, c5, c6, c7])

        # When
        self.population.add_alp_classifier(child, new_list)

        # Then
        self.assertEqual(7, len(self.population))
        self.assertEqual(0, len(new_list))
        self.assertIn(c1, self.population)
        self.assertIn(c2, self.population)
        self.assertIn(c3, self.population)
        self.assertIn(c4, self.population)
        self.assertIn(c5, self.population)
        self.assertIn(c6, self.population)
        self.assertIn(c7, self.population)

        # `C4` should be subsumer of `child`
        self.assertAlmostEqual(0.907987, c4.q, places=1)
