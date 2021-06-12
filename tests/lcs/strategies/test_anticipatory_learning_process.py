import lcs.agents.acs2 as acs2

import lcs.strategies.anticipatory_learning_process as alp


class TestAnticipatoryLearningProcess:

    def test_should_insert_alp_offspring_1(self):
        # given
        cfg = acs2.Configuration(
            classifier_length=8,
            number_of_possible_actions=8)

        population = acs2.ClassifiersList()
        new_list = acs2.ClassifiersList()

        child = acs2.Classifier(
            condition='1##1#010',
            action=0,
            effect='0####101',
            quality=0.5,
            reward=8.96245,
            immediate_reward=0,
            experience=1,
            tga=423,
            talp=423,
            tav=27.3182,
            cfg=cfg
        )

        c1 = acs2.Classifier(
            condition='1##1#010',
            action=0,
            effect='0####101',
            quality=0.571313,
            reward=7.67011,
            immediate_reward=0,
            experience=3,
            tga=225,
            talp=423,
            tav=70.881,
            cfg=cfg
        )

        c2 = acs2.Classifier(
            condition='1####010',
            action=0,
            effect='0####101',
            quality=0.462151,
            reward=8.96245,
            immediate_reward=0,
            experience=11,
            tga=143,
            talp=423,
            tav=27.3182,
            cfg=cfg
        )

        c3 = acs2.Classifier(
            condition='1####0##',
            action=0,
            effect='0####1##',
            quality=0.31452,
            reward=9.04305,
            immediate_reward=0,
            experience=19,
            tga=49,
            talp=423,
            tav=19.125,
            cfg=cfg
        )

        population.extend([c1, c2, c3])

        # when
        alp.add_classifier(child, population, new_list, cfg.theta_exp)

        # then
        assert len(population) == 3
        assert c1 in population
        assert c2 in population
        assert c3 in population
        assert abs(0.592747 - c1.q) < 0.01

    def test_should_insert_alp_offspring_2(self):
        # given
        cfg = acs2.Configuration(
            classifier_length=8,
            number_of_possible_actions=8)
        population = acs2.ClassifiersList()
        new_list = acs2.ClassifiersList()

        child = acs2.Classifier(
            condition='#1O##O##',
            action=0,
            reward=18.206,
            immediate_reward=0,
            experience=1,
            tga=747,
            talp=747,
            tav=22.0755,
            cfg=cfg
        )

        c1 = acs2.Classifier(
            condition='#1O#O###',
            action=0,
            quality=0.650831,
            reward=14.8323,
            immediate_reward=0,
            experience=5,
            tga=531,
            talp=747,
            tav=48.3562,
            cfg=cfg
        )

        c2 = acs2.Classifier(
            condition='##O#O###',
            action=0,
            quality=0.79094,
            reward=9.97782,
            immediate_reward=0,
            experience=10,
            tga=330,
            talp=747,
            tav=43.7171,
            cfg=cfg
        )

        c3 = acs2.Classifier(
            condition='#1O###1O',
            action=0,
            effect='#O1####1',
            quality=0.515369,
            reward=8.3284,
            immediate_reward=0,
            experience=8,
            tga=316,
            talp=747,
            tav=57.8883,
            cfg=cfg
        )

        c3.mark[0].update(['1'])
        c3.mark[3].update(['0'])
        c3.mark[4].update(['0'])
        c3.mark[5].update(['0'])

        c4 = acs2.Classifier(
            condition='####O###',
            action=0,
            quality=0.903144,
            reward=14.8722,
            immediate_reward=0,
            experience=25,
            tga=187,
            talp=747,
            tav=23.0038,
            cfg=cfg
        )

        c5 = acs2.Classifier(
            condition='#1O####O',
            action=0,
            effect='#O1####1',
            quality=0.647915,
            reward=9.24712,
            immediate_reward=0,
            experience=14,
            tga=154,
            talp=747,
            tav=44.5457,
            cfg=cfg
        )
        c5.mark[0].update(['1'])
        c5.mark[3].update(['0', '1'])
        c5.mark[4].update(['0', '1'])
        c5.mark[5].update(['0', '1'])
        c5.mark[6].update(['0', '1'])

        c6 = acs2.Classifier(
            condition='#1O#####',
            action=0,
            quality=0.179243,
            reward=18.206,
            immediate_reward=0,
            experience=29,
            tga=104,
            talp=747,
            tav=22.0755,
            cfg=cfg
        )
        c6.mark[0].update(['1'])
        c6.mark[3].update(['1'])
        c6.mark[4].update(['1'])
        c6.mark[5].update(['1'])
        c6.mark[6].update(['0', '1'])
        c6.mark[7].update(['0', '1'])

        c7 = acs2.Classifier(
            condition='##O#####',
            action=0,
            quality=0.100984,
            reward=15.91,
            immediate_reward=0,
            experience=44,
            tga=58,
            talp=747,
            tav=14.4171,
            cfg=cfg
        )
        c7.mark[0].update(['0', '1'])
        c7.mark[1].update(['0', '1'])
        c7.mark[3].update(['0', '1'])
        c7.mark[5].update(['0', '1'])
        c7.mark[6].update(['0', '1'])
        c7.mark[7].update(['0', '1'])

        # Add classifiers into current ClassifierList
        population.extend([c1, c2, c3, c4, c5, c6, c7])

        # When
        alp.add_classifier(child, population, new_list, cfg.theta_exp)

        # Then
        assert 7 == len(population)
        assert 0 == len(new_list)
        assert c1 in population
        assert c2 in population
        assert c3 in population
        assert c4 in population
        assert c5 in population
        assert c6 in population
        assert c7 in population

        # `C4` should be subsumer of `child`
        assert abs(0.907987 - c4.q) < 0.01
