def basic_metrics(trial: int, steps: int, reward: int):
    return {
        'trial': trial,
        'steps_in_trial': steps,
        'reward': reward
    }


def population_metrics(population, environment):
    metrics = {
        'population': 0,
        'numerosity': 0,
        'reliable': 0,
    }

    for cl in population:
        metrics['population'] += 1
        metrics['numerosity'] += cl.num
        if cl.is_reliable():
            metrics['reliable'] += 1

    return metrics


# def _collect_agent_metrics(self, trial, steps, total_steps) -> Metric:
#     # regions = self._count_averaged_regions()
#
#     return {
#         'population': len(self.population),
#         'numerosity': sum(cl.num for cl in self.population),
#         'reliable': len([cl for cl in
#                          self.population if cl.is_reliable()]),
#         'fitness': (sum(cl.fitness for cl in self.population) /
#                     len(self.population)),
#         # 'cover_ratio': (sum(cl.condition.cover_ratio for cl
#         #                     in self.population) / len(self.population)),
#         # 'region_1': regions[1],
#         # 'region_2': regions[2],
#         # 'region_3': regions[3],
#         # 'region_4': regions[4],
#         'trial': trial,
#         'steps': steps,
#         'total_steps': total_steps
#     }


# def _count_averaged_regions(self) -> Dict[int, float]:
#     region_counts = {1: 0, 2: 0, 3: 0, 4: 0}
#
#     for cl in self.population:
#         for region, counts in cl.get_interval_proportions().items():
#             region_counts[region] += counts
#
#     all_elems = sum(i for r, i in region_counts.items())
#
#     return {r: i / all_elems for r, i in region_counts.items()}
